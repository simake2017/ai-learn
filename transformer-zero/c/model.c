/* =============================================================================
 * model.c - GPT 模型：参数管理、内存分配、完整 forward
 *
 * 内存布局：
 *   params_memory: 一大块 float，按 offset 切片给每个张量
 *   acts_memory:   一大块 float，按 offset 切片给每个激活
 *
 * =============================================================================
 */

#include "transformer.h"

/* NUM_PER_TENSORS 已在 transformer.h 中定义（16 类参数张量） */

/* 计算每个张量大小
 *
 * 注意：vary with config，按层参数 (ln1w 等) 数量 = n_layer × C
 */
void fill_in_parameter_sizes(int *param_sizes, Config config) {
    int Vp = config.vocab_size;
    int C = config.n_embd;
    int maxT = config.block_size;
    int L = config.n_layer;
    param_sizes[0]  = Vp * C;       // wte
    param_sizes[1]  = maxT * C;     // wpe
    param_sizes[2]  = L * C;        // ln1w
    param_sizes[3]  = L * C;        // ln1b
    param_sizes[4]  = L * 3 * C * C; // qkvw
    param_sizes[5]  = L * 3 * C;    // qkvb
    param_sizes[6]  = L * C * C;    // projw
    param_sizes[7]  = L * C;        // projb
    param_sizes[8]  = L * C;        // ln2w
    param_sizes[9]  = L * C;        // ln2b
    param_sizes[10] = L * 4 * C * C; // fcw
    param_sizes[11] = L * 4 * C;    // fcb
    param_sizes[12] = L * C * 4 * C; // fcprojw
    param_sizes[13] = L * C;        // fcprojb
    param_sizes[14] = C;            // lnfw
    param_sizes[15] = C;            // lnfb
}

/* 给 16 个张量分配一片大内存，并设置指针 */
float* malloc_and_set_params(Params* params, int *param_sizes) {
    int num_parameters = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }

    float* mem = (float*)calloc(num_parameters, sizeof(float));
    if (!mem) { fprintf(stderr, "OOM: params\n"); exit(1); }

    float** ptrs[] = {
        &params->wte, &params->wpe,
        &params->ln1w, &params->ln1b,
        &params->qkvw, &params->qkvb,
        &params->projw, &params->projb,
        &params->ln2w, &params->ln2b,
        &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb,
        &params->lnfw, &params->lnfb,
    };

    float* cur = mem;
    for (int i = 0; i < NUM_PER_TENSORS; i++) {
        *(ptrs[i]) = cur;
        cur += param_sizes[i];
    }

    return mem;
}

/* 初始化参数：用小随机数 + 矩阵权重用 Xavier-like 缩放 */
static void init_params(GPT* model) {
    int C = model->config.n_embd;
    int V = model->config.vocab_size;
    int L = model->config.n_layer;
    int maxT = model->config.block_size;

    // 初始化所有参数为小随机数（正态分布近似）
    // 用 Box-Muller 转换生成
    unsigned int seed = 42;
    srand(seed);

    int total = model->num_parameters;
    for (int i = 0; i < total; i++) {
        float r1 = (float)rand() / RAND_MAX;
        float r2 = (float)rand() / RAND_MAX;
        float gaussian = sqrtf(-2.0f * logf(r1 + 1e-10f)) * cosf(2.0f * 3.14159f * r2);
        model->params_memory[i] = gaussian * 0.02f;  // 标准差 0.02
    }

    // bias 和 LN gamma/beta 初始化为 0/1
    // 简化，全部为 0 bias，1 gamma
    // （实际训练时需要细致初始化，这里为简化）

    (void)C; (void)V; (void)L; (void)maxT;
}

/* 分配并初始化梯度 */
float* malloc_grads(Grads* grads, int *param_sizes) {
    int num_parameters = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }

    float* mem = (float*)calloc(num_parameters, sizeof(float));
    if (!mem) { fprintf(stderr, "OOM: grads\n"); exit(1); }

    float** ptrs[] = {
        &grads->wte, &grads->wpe,
        &grads->ln1w, &grads->ln1b,
        &grads->qkvw, &grads->qkvb,
        &grads->projw, &grads->projb,
        &grads->ln2w, &grads->ln2b,
        &grads->fcw, &grads->fcb,
        &grads->fcprojw, &grads->fcprojb,
        &grads->lnfw, &grads->lnfb,
    };

    float* cur = mem;
    for (int i = 0; i < NUM_PER_TENSORS; i++) {
        *(ptrs[i]) = cur;
        cur += param_sizes[i];
    }

    return mem;
}

void zero_grads(Grads* grads, int n) {
    if (grads == NULL) return;
    // 简化：通过 grads->wte 等指针访问，假设都在一片连续内存里
    // 实际上 grads_memory 才是连续内存
    // 此函数只是占位
}

/* === 激活分配 === */
#define NUM_ACTIVATION_TENSORS 13

void fill_in_activation_sizes(int *act_sizes, Config config, int B, int T) {
    int V = config.vocab_size;
    int C = config.n_embd;
    int NH = config.n_head;
    int L = config.n_layer;
    act_sizes[0]  = B * T * C;            // x
    act_sizes[1]  = B * T * V;            // logits
    act_sizes[2]  = B * T * V;            // probs
    act_sizes[3]  = B * T;                // losses (额外)
    act_sizes[4]  = B * T * C;            // tok_emb
    // pos_emb 略
    act_sizes[5]  = L * B * T * C;        // ln1
    act_sizes[6]  = L * B * T * 3 * C;    // qkv
    act_sizes[7]  = L * B * NH * T * T;   // preatt
    act_sizes[8]  = L * B * NH * T * T;   // att
    act_sizes[9]  = L * B * T * C;        // att_out
    act_sizes[10] = L * B * T * C;        // residual2
    act_sizes[11] = L * B * T * C;        // ln2
    act_sizes[12] = L * B * T * 4 * C;    // fch
    // fch_gelu, fcproj, lnf 简化：不单独存储，复用内存
}

float* malloc_and_set_activations(Activations* acts, int *act_sizes) {
    int num_activations = 0;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }

    float* mem = (float*)calloc(num_activations, sizeof(float));
    if (!mem) { fprintf(stderr, "OOM: activations\n"); exit(1); }

    float** ptrs[] = {
        &acts->x, &acts->logits, &acts->probs, &acts->losses,
        &acts->tok_emb,
        &acts->ln1, &acts->qkv, &acts->preatt, &acts->att,
        &acts->att_out, &acts->residual2, &acts->ln2, &acts->fch,
    };

    float* cur = mem;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = cur;
        cur += act_sizes[i];
    }

    // fch_gelu 和 fcproj 复用 fch 的一部分（简化）
    acts->fch_gelu = acts->fch;
    acts->fcproj = acts->fch;  // 同样内存，用完 fch_gelu 后可以覆盖

    return mem;
}

/* === Embedding === */
void embedding_forward(float* out, const int* inputs,
                       const float* wte, const float* wpe,
                       int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token_id = inputs[b * T + t]; //wangyang 这里就是输入的token 向量,获取token id
            for (int c = 0; c < C; c++) {
                out[(b * T + t) * C + c] = // wte 矩阵会叠加 wpe
                    wte[token_id * C + c] + wpe[t * C + c];
            }
        }
    }
}

/* === 一个 Transformer Block === */
static void block_forward(GPT* model, int l, int B, int T, int C, int NH) {
    Params* p = &model->params;
    Activations* a = &model->acts;

    float* l_ln1w    = p->ln1w    + l * C;
    float* l_ln1b    = p->ln1b    + l * C;
    float* l_qkvw    = p->qkvw    + l * 3 * C * C;
    float* l_qkvb    = p->qkvb    + l * 3 * C;
    float* l_projw   = p->projw   + l * C * C;
    float* l_projb   = p->projb   + l * C;
    float* l_ln2w    = p->ln2w    + l * C;
    float* l_ln2b    = p->ln2b    + l * C;
    float* l_fcw     = p->fcw     + l * 4 * C * C;
    float* l_fcb     = p->fcb     + l * 4 * C;
    float* l_fcprojw = p->fcprojw + l * C * 4 * C;
    float* l_fcprojb = p->fcprojb + l * C;

    float* l_x        = a->x;
    float* l_ln1      = a->ln1      + l * B * T * C;
    float* l_qkv      = a->qkv      + l * B * T * 3 * C;
    float* l_preatt   = a->preatt   + l * B * NH * T * T;
    float* l_att      = a->att      + l * B * NH * T * T;
    float* l_att_out  = a->att_out  + l * B * T * C;
    float* l_residual2= a->residual2+ l * B * T * C;
    float* l_ln2      = a->ln2      + l * B * T * C;
    float* l_fch      = a->fch      + l * B * T * 4 * C;

    // ===== 子层 1: Attention =====
    layernorm_forward(l_ln1, l_x, l_ln1w, l_ln1b, B, T, C);
    matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
    attention_forward(l_att_out, l_preatt, l_att, l_qkv, B, T, C, NH);
    matmul_forward(l_att_out, l_att_out, l_projw, l_projb, B, T, C, C);
    residual_forward(l_residual2, l_x, l_att_out, B * T * C);

    // ===== 子层 2: MLP =====
    layernorm_forward(l_ln2, l_residual2, l_ln2w, l_ln2b, B, T, C);
    matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
    gelu_forward(l_fch, l_fch, B * T * 4 * C);  // in-place GELU
    matmul_forward(l_x, l_fch, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
    residual_forward(l_x, l_residual2, l_x, B * T * C);
}

/* === 完整 GPT forward === */
void gpt_forward(GPT* model, const int* inputs, const int* targets) {
    int B = model->batch_size;
    int T = model->seq_len;
    int C = model->config.n_embd;
    int V = model->config.vocab_size;
    int NH = model->config.n_head;
    int L = model->config.n_layer;

    // Step 1: Embedding
    embedding_forward(model->acts.tok_emb, inputs,
                      model->params.wte, model->params.wpe, B, T, C);
    // 复制到 x（后续操作 in-place）
    memcpy(model->acts.x, model->acts.tok_emb, B * T * C * sizeof(float));

    // Step 2: n_layer 个 Block
    for (int l = 0; l < L; l++) {
        block_forward(model, l, B, T, C, NH);
    }

    // Step 3: Final LayerNorm（用 lnfw, lnfb）
    layernorm_forward(model->acts.x, model->acts.x,
                      model->params.lnfw, model->params.lnfb, B, T, C);

    // Step 4: LM Head: C → V（用 wte，weight tying）
    matmul_forward(model->acts.logits, model->acts.x,
                   model->params.wte, NULL, B, T, C, V);

    // Step 5: Softmax
    softmax_forward(model->acts.probs, model->acts.logits, B, T, V);

    // Step 6: Loss
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs,
                              targets, B, T, V);
        float total = 0.0f;
        for (int i = 0; i < B * T; i++) total += model->acts.losses[i];
        model->mean_loss = total / (B * T);
    } else {
        model->mean_loss = -1.0f;
    }
}

/* === 初始化 GPT 模型 === */
void config_init(Config* c) {
    c->vocab_size = 65;     // 字符级 Shakespeare
    c->block_size = 64;     // 教学用，足够长
    c->n_layer = 6;
    c->n_head = 4;
    c->n_embd = 128;
}

void config_small(Config* c) {
    // 更小的模型用于快速测试（~120K 参数）
    c->vocab_size = 65;
    c->block_size = 16;
    c->n_layer = 2;
    c->n_head = 2;
    c->n_embd = 32;
}

void gpt_build_from_checkpoint(GPT* model, const char* checkpoint_path) {
    config_init(&model->config);

    // 分配参数
    int param_sizes[NUM_PER_TENSORS];
    fill_in_parameter_sizes(param_sizes, model->config);

    int total = 0;
    for (int i = 0; i < NUM_PER_TENSORS; i++) total += param_sizes[i];
    model->num_parameters = total;

    model->params_memory = malloc_and_set_params(&model->params, param_sizes);
    model->grads_memory = malloc_grads(&model->grads, param_sizes);
    model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
    model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));

    // 分配激活（懒分配：在第一次 forward 时）
    int act_sizes[NUM_ACTIVATION_TENSORS] = {0};
    fill_in_activation_sizes(act_sizes, model->config, 1, 1);  // 占位
    int total_acts = 0;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++) total_acts += act_sizes[i];
    model->num_activations = total_acts;

    // 初始化参数
    init_params(model);

    // 加载权重（如果文件存在）
    if (checkpoint_path != NULL) {
        FILE* f = fopen(checkpoint_path, "rb");
        if (f) {
            fread(model->params_memory, sizeof(float), model->num_parameters, f);
            fclose(f);
            printf("Loaded checkpoint: %s\n", checkpoint_path);
        } else {
            printf("No checkpoint found, using random init\n");
        }
    }
}

void gpt_save_checkpoint(GPT* model, const char* path_path) {
    FILE* f = fopen(path_path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot save to %s\n", path_path);
        return;
    }
    fwrite(model->params_memory, sizeof(float), model->num_parameters, f);
    fclose(f);
    printf("Saved checkpoint: %s\n", path_path);
}

void gpt_free(GPT* model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    if (model->acts_memory) free(model->acts_memory);
}
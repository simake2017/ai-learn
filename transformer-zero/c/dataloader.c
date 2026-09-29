/* =============================================================================
 * dataloader.c - 数据加载器（简化版）
 *
 * 复用 nanoGPT 的 train.bin (uint16 tokens)
 * =============================================================================
 */

#include "transformer.h"

void dataloader_init(DataLoader* loader, const char* path, int B, int T, int vocab_size) {
    loader->B = B;
    loader->T = T;
    loader->vocab_size = vocab_size;

    // 读取整个文件到内存
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    loader->data = (unsigned char*)malloc(size);
    fread(loader->data, 1, size, f);
    fclose(f);

    // 转为 int（避免溢出）
    int num_tokens = size / 2;  // uint16
    loader->inputs = (int*)malloc(B * T * sizeof(int));
    loader->targets = (int*)malloc(B * T * sizeof(int));
    loader->num_batches = num_tokens / (B * T);
    loader->current_idx = 0;

    // 保存原始 tokens 到 inputs (临时)
    // 实际我们每次取 batch 时再生成
    // 简化：用一个临时数组
    loader->current_idx = 0;
}

void dataloader_next_batch(DataLoader* loader) {
    // 每个 batch 取 B*T+1 个连续 tokens
    int B = loader->B;
    int T = loader->T;

    if (loader->current_idx + B * T + 1 > loader->num_batches * B * T) {
        loader->current_idx = 0;  // 重头开始
    }

    int offset = loader->current_idx * B * T;
    unsigned char* data = loader->data + offset * 2;  // 乘 2 因为 uint16

    for (int i = 0; i < B * T; i++) {
        // 小端序读 uint16
        int lo = data[i * 2];
        int hi = data[i * 2 + 1];
        int token = (hi << 8) | lo;
        loader->inputs[i] = token;
        loader->targets[i] = token;  // 暂时，下一行覆盖
    }
    // targets = inputs[1:]
    for (int i = 0; i < B * T - 1; i++) {
        loader->targets[i] = loader->inputs[i + 1];
    }
    loader->targets[B * T - 1] = 0;  // 最后一个 target 填 0（占位）

    loader->current_idx += 1;
}

void dataloader_free(DataLoader* loader) {
    free(loader->data);
    free(loader->inputs);
    free(loader->targets);
}
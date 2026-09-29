#!/usr/bin/env python3
"""
Gemini vs 千问 对比测试（走 102 new-api 的 OpenAI 兼容接口）
同一组测试题让两组模型作答，输出并排对比 + 用时 + token 消耗。

对比组:
    轻量档: gemini-3.5-flash-lite  vs  qwen3.7-flash
    主力档: gemini-3.5-flash       vs  qwen3-coder-plus

用法:
    python3 model_compare_gemini.py          # 跑全部用例
    python3 model_compare_gemini.py 1 4      # 只跑编号 1 和 4 的用例
"""
import json, sys, time
import urllib.request

API = "http://192.168.188.102:3001/v1/chat/completions"
KEY = "JBDRRudQpQq0Mo9n1eBb177b204244D28dD4714a1bF887B2"
MODELS = [
    "gemini-3.5-flash-lite",
    "qwen3.7-flash",
    "gemini-3.5-flash",
    "qwen3-coder-plus",
]

CASES = [
    {
        "id": 1,
        "name": "算法实现（考编程基本功）",
        "prompt": (
            "用 C++ 实现 LRU 缓存，要求 get/put 都是 O(1)，"
            "给出完整代码和关键思路说明。"
        ),
        "考点": "代码质量、数据结构选择、边界处理",
    },
    {
        "id": 2,
        "name": "代码 Debug（考找 bug 能力）",
        "prompt": (
            "下面这段 C++ 代码有问题，找出所有 bug 并给出修复：\n"
            "```cpp\n"
            "std::vector<int> v;\n"
            "for (auto it = v.begin(); it != v.end(); ++it) {\n"
            "    if (*it % 2 == 0) v.erase(it);\n"
            "}\n"
            "```\n"
        ),
        "考点": "迭代器失效这个经典坑，看谁能精准定位",
    },
    {
        "id": 3,
        "name": "复杂推理（考深度思考）",
        "prompt": (
            "一个农夫带狼、羊、白菜过河，船一次只能载农夫和一样东西，"
            "狼和羊独处会吃羊，羊和白菜独处会吃白菜。给出完整过河步骤，"
            "并解释为什么这是唯一的最优解结构。"
        ),
        "考点": "多步约束推理",
    },
    {
        "id": 4,
        "name": "简单问答（轻量档的主场）",
        "prompt": "Python 里怎么把字符串反转？一行代码回答。",
        "考点": "简单任务 → 轻量档又快又对",
    },
    {
        "id": 5,
        "name": "系统设计（考架构视野）",
        "prompt": (
            "设计一个短链接服务（类似 bit.ly），日写入 100 万、读取 1 亿。"
            "给出：ID 生成方案、存储选型、缓存策略、关键权衡。500 字以内。"
        ),
        "考点": "架构权衡能力",
    },
]


def ask(model, prompt, max_tokens=1200):
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    dt = time.time() - t0
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    return {
        "text": msg.get("content", ""),
        "secs": round(dt, 1),
        "in_tokens": usage.get("prompt_tokens", 0),
        "out_tokens": usage.get("completion_tokens", 0),
    }


def main():
    sel = set(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else None
    cases = [c for c in CASES if sel is None or c["id"] in sel]
    report = {"time": time.strftime("%F %T"), "models": MODELS, "cases": []}

    for c in cases:
        print(f"\n{'='*70}\n用例 {c['id']}: {c['name']}\n考点: {c['考点']}\n{'='*70}")
        entry = {"id": c["id"], "name": c["name"], "answers": {}}
        for m in MODELS:
            try:
                r = ask(m, c["prompt"])
                print(f"\n--- {m} ({r['secs']}s, 出 {r['out_tokens']} tok) ---")
                print(r["text"].strip()[:800])
                entry["answers"][m] = r
            except Exception as e:
                print(f"\n--- {m} 失败: {e} ---")
                entry["answers"][m] = {"error": str(e)}
        report["cases"].append(entry)

    # 汇总表
    print(f"\n{'='*70}\n汇总（用时 / 输出 token）\n{'='*70}")
    print(f"{'用例':<24}" + "".join(f"{m[:18]:>20}" for m in MODELS))
    for entry in report["cases"]:
        row = f"{entry['name'][:22]:<24}"
        for m in MODELS:
            a = entry["answers"][m]
            if "error" in a:
                row += f"{'失败':>20}"
            else:
                row += f"{a['secs']}s/{a['out_tokens']}tok".rjust(20)
        print(row)

    out = ("/home/magene/runtime/ai/ai-learn/ai-test/results/"
           f"gemini-report-{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()

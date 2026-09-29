#!/usr/bin/env python3
"""编程能力进阶实测：LFU 缓存（LeetCode 460 hard）编译+运行验证"""
import json, re, subprocess, time, urllib.request

API = "http://192.168.188.102:3001/v1/chat/completions"
KEY = "JBDRRudQpQq0Mo9n1eBb177b204244D28dD4714a1bF887B2"
MODELS = ["gemini-3.5-flash-lite", "qwen3.7-flash", "gemini-3.5-flash", "qwen3-coder-plus"]

PROMPT = (
    "用 C++ 实现 LFU 缓存类（LeetCode 460），严格使用以下接口（不要改动）：\n"
    "```cpp\n"
    "#include <iostream>\n"
    "#include <unordered_map>\n"
    "#include <list>\n"
    "class LFUCache {\n"
    "public:\n"
    "    LFUCache(int capacity);\n"
    "    int get(int key);              // 不存在返回 -1；每次访问频次+1\n"
    "    void put(int key, int value);  // 满时淘汰：频次最低者；频次相同淘汰最久未使用\n"
    "};\n"
    "```\n"
    "要求 get/put 都是 O(1)。只输出一个完整的 .cpp 代码块（含上面头文件），"
    "不要 main 函数，不要任何解释文字。"
)

DRIVER = '''
int main() {
    LFUCache c(2);
    c.put(1, 1); c.put(2, 2);
    if (c.get(1) != 1) { std::cout << "FAIL1\\n"; return 1; }   // 1 频次2, 2 频次1
    c.put(3, 3);                       // 淘汰频次最低的 2
    if (c.get(2) != -1) { std::cout << "FAIL2\\n"; return 1; }
    if (c.get(3) != 3) { std::cout << "FAIL3\\n"; return 1; }   // 3 频次2, 1 频次2
    c.put(4, 4);                       // 1、3 频次相同，淘汰最久未用的 1
    if (c.get(1) != -1) { std::cout << "FAIL4\\n"; return 1; }
    if (c.get(3) != 3) { std::cout << "FAIL5\\n"; return 1; }
    if (c.get(4) != 4) { std::cout << "FAIL6\\n"; return 1; }
    LFUCache d(0);                     // 容量 0 边界
    d.put(0, 0);
    if (d.get(0) != -1) { std::cout << "FAIL7\\n"; return 1; }
    LFUCache e(2);
    e.put(3, 1); e.put(2, 1); e.put(2, 2); e.put(4, 4);
    if (e.get(2) != 2) { std::cout << "FAIL8\\n"; return 1; }   // 覆盖更新后频次正确
    std::cout << "PASS\\n";
    return 0;
}
'''

def ask(model):
    body = json.dumps({"model": model, "max_tokens": 2500,
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            d = json.load(r)
        return d["choices"][0]["message"].get("content", ""), round(time.time()-t0, 1)
    except Exception as e:
        return f"ERROR: {e}", round(time.time()-t0, 1)

def extract(text):
    m = re.findall(r"```(?:cpp|c\+\+)?\n(.*?)```", text, re.S)
    return max(m, key=len) if m else None

results = {}
for model in MODELS:
    text, secs = ask(model)
    code = extract(text)
    r = {"secs": secs, "stage": "", "detail": ""}
    if not code:
        r["stage"], r["detail"] = "提取失败", text[:150].replace("\n", " ")
    else:
        src = f"/tmp/lfu_{MODELS.index(model)}.cpp"
        open(src, "w").write(code + "\n" + DRIVER)
        p = subprocess.run(["g++", "-std=c++17", "-O2", "-o", src+".out", src],
                           capture_output=True, text=True)
        if p.returncode != 0:
            r["stage"], r["detail"] = "编译失败", p.stderr.strip().split("\n")[0][:150]
        else:
            q = subprocess.run([src+".out"], capture_output=True, text=True, timeout=10)
            ok = q.stdout.strip() == "PASS"
            r["stage"] = "运行通过" if ok else "运行错误"
            r["detail"] = q.stdout.strip() or q.stderr[:100]
    results[model] = r
    print(f"{model:24} {secs:>5}s  [{r['stage']}]  {r['detail'][:80]}")

json.dump(results, open("/home/magene/runtime/ai/ai-learn/ai-test/results/coding-test-lfu.json", "w"),
          ensure_ascii=False, indent=1)

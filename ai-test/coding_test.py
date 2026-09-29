#!/usr/bin/env python3
"""编程能力实测：4 模型实现 LRU，提取代码真实编译运行验证"""
import json, re, subprocess, time, urllib.request

API = "http://192.168.188.102:3001/v1/chat/completions"
KEY = "JBDRRudQpQq0Mo9n1eBb177b204244D28dD4714a1bF887B2"
MODELS = ["gemini-3.5-flash-lite", "qwen3.7-flash", "gemini-3.5-flash", "qwen3-coder-plus"]

PROMPT = (
    "用 C++ 实现一个 LRU 缓存类，严格使用以下接口（不要改动）：\n"
    "```cpp\n"
    "#include <iostream>\n"
    "#include <unordered_map>\n"
    "#include <list>\n"
    "class LRUCache {\n"
    "public:\n"
    "    LRUCache(int capacity);\n"
    "    int get(int key);              // 不存在返回 -1\n"
    "    void put(int key, int value);  // 满了淘汰最久未使用\n"
    "};\n"
    "```\n"
    "只输出一个完整的 .cpp 代码块（含上面头文件），不要 main 函数，不要任何解释文字。"
)

DRIVER = '''
int main() {
    LRUCache c(2);
    c.put(1, 1); c.put(2, 2);
    if (c.get(1) != 1) { std::cout << "FAIL1\\n"; return 1; }
    c.put(3, 3);                       // 淘汰 2
    if (c.get(2) != -1) { std::cout << "FAIL2\\n"; return 1; }
    c.put(4, 4);                       // 淘汰 1（3 比 1 新）
    if (c.get(1) != -1) { std::cout << "FAIL3\\n"; return 1; }
    if (c.get(3) != 3) { std::cout << "FAIL4\\n"; return 1; }
    if (c.get(4) != 4) { std::cout << "FAIL5\\n"; return 1; }
    LRUCache d(2);
    d.put(2, 1); d.put(2, 2);          // 同 key 覆盖
    if (d.get(2) != 2) { std::cout << "FAIL6\\n"; return 1; }
    d.put(3, 3);                       // 淘汰的不应是刚用过的 2
    if (d.get(2) != 2) { std::cout << "FAIL7\\n"; return 1; }
    std::cout << "PASS\\n";
    return 0;
}
'''

def ask(model):
    body = json.dumps({"model": model, "max_tokens": 2000,
                       "messages": [{"role": "user", "content": PROMPT}]}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.load(r)
        return d["choices"][0]["message"].get("content", ""), round(time.time()-t0, 1)
    except Exception as e:
        return f"ERROR: {e}", round(time.time()-t0, 1)

def extract(text):
    m = re.findall(r"```(?:cpp|c\+\+)?\n(.*?)```", text, re.S)
    if not m: return None
    return max(m, key=len)  # 取最长的代码块

results = {}
for model in MODELS:
    text, secs = ask(model)
    code = extract(text)
    r = {"secs": secs, "stage": "", "detail": ""}
    if not code:
        r["stage"], r["detail"] = "提取失败", text[:150].replace("\n", " ")
    else:
        src = f"/tmp/lru_{MODELS.index(model)}.cpp"
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

json.dump(results, open("/home/magene/runtime/ai/ai-learn/ai-test/results/coding-test.json", "w"),
          ensure_ascii=False, indent=1)

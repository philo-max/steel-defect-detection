"""
实训3: 本地大模型部署与模型量化体验

使用 Ollama 在本地部署量化模型，体验边缘推理。
"""

import requests
import time
import json
import os

OLLAMA_API_URL = "http://localhost:11434"


def check_ollama_running() -> bool:
    """检查 Ollama 服务是否运行"""
    try:
        resp = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def list_models() -> list:
    """列出已下载的模型"""
    try:
        resp = requests.get(f"{OLLAMA_API_URL}/api/tags")
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return []


def call_local_llm(prompt: str, model: str = "qwen:0.5b-chat",
                   temperature: float = 0.7) -> tuple:
    """
    调用本地 Ollama 模型
    
    Returns:
        (response_text, elapsed_seconds)
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json=payload, timeout=120
        )
        resp.raise_for_status()
        elapsed = time.time() - start
        result = resp.json()
        return result.get("response", ""), elapsed
    except requests.ConnectionError:
        return ("❌ 无法连接到 Ollama。请确保 Ollama 已安装并运行。\n"
                "   下载: https://ollama.com/download\n"
                "   启动后运行: ollama pull qwen:0.5b-chat", 0)
    except Exception as e:
        return f"调用出错: {e}", 0


def benchmark_models(prompt: str, models: list):
    """对比不同模型的推理速度"""
    print("\n" + "=" * 60)
    print("模型推理速度对比")
    print("=" * 60)

    results = []
    for model in models:
        print(f"\n测试模型: {model} ...")
        answer, duration = call_local_llm(prompt, model=model)
        results.append((model, duration, len(answer)))
        print(f"  耗时: {duration:.1f}s | 输出长度: {len(answer)} 字符")

    print(f"\n{'─' * 50}")
    print(f"{'模型':<20} {'耗时':>8} {'输出长度':>8}")
    print(f"{'─' * 50}")
    for m, d, l in sorted(results, key=lambda x: x[1]):
        print(f"{m:<20} {d:>7.1f}s {l:>7}字")


def demo():
    """演示本地模型调用"""
    print("=" * 60)
    print("实训3: 本地大模型部署与模型量化体验 (Ollama)")
    print("=" * 60)

    # 检查Ollama状态
    if not check_ollama_running():
        print("\n⚠️  Ollama 服务未运行!")
        print("请按以下步骤操作:")
        print("  1. 下载 Ollama: https://ollama.com/download")
        print("  2. 安装并启动 Ollama")
        print("  3. 拉取模型: ollama pull qwen:0.5b-chat")
        print("  4. 重新运行本脚本")

        # 给出安装指导
        print("\n📋 手动安装指南 (PowerShell):")
        print("  # 下载并安装 Ollama")
        print("  # 然后运行:")
        print("  ollama serve          # 启动服务")
        print("  ollama pull qwen:0.5b-chat   # 下载轻量模型 (约400MB)")
        print("  ollama pull qwen:4b-chat     # (可选) 更大模型 (~2.5GB)")
        print("  ollama pull llama3.2:1b      # (可选) Llama模型")

        # 展示预期效果
        print("\n📊 预期效果 (参考数据):")
        print("  模型            量化    内存    推理速度")
        print("  qwen:0.5b-chat  Q4_0    ~1GB    ~2s/响应")
        print("  llama3.2:1b     Q4_0    ~1.5GB  ~3s/响应")
        print("  qwen:4b-chat    Q4_0    ~4GB    ~8s/响应")
        return

    # 列出已安装模型
    models = list_models()
    print(f"\n✅ Ollama 服务运行中")
    print(f"已安装模型: {', '.join(models) if models else '无'}")

    default_model = models[0] if models else "qwen:0.5b-chat"

    test_prompt = "请用中文简要介绍钢铁生产中常见的3种表面缺陷类型及其成因。"

    print(f"\n测试问题: {test_prompt}")
    print(f"使用模型: {default_model}")
    print("正在调用本地模型...\n")

    answer, duration = call_local_llm(test_prompt, model=default_model)
    print(f"模型回答:\n{answer}")
    print(f"\n⏱️  推理耗时: {duration:.1f} 秒")
    print(f"📝 输出长度: {len(answer)} 字符")

    # 如果有多模型，做对比
    if len(models) >= 2:
        benchmark_models(test_prompt, models[:3])

    print("\n" + "=" * 60)
    print("小结: 量化后的小模型可在普通CPU上运行。")
    print("工业场景推荐: 7B-14B量化模型 + GPU加速。")
    print("=" * 60)


def main():
    demo()


if __name__ == "__main__":
    main()

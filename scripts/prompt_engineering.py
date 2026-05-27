"""
实训1: 大模型API调用与提示词工程入门

掌握通过API调用大语言模型的基本流程。
理解提示词工程: 角色扮演、任务分解、少样本学习、思维链。

支持后端: Gemini (默认) / SiliconFlow Qwen
"""

from openai import OpenAI
import json
import os

# ==================== 配置 ====================
# 自动检测可用的 API 后端
def _detect_backend():
    """检测 Gemini 或 SiliconFlow 配置"""
    from dotenv import load_dotenv
    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        return {
            "api_key": gemini_key,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "model": "gemini-2.5-flash",
            "name": "Gemini"
        }

    sf_key = os.getenv("SILICONFLOW_API_KEY", "")
    if sf_key:
        return {
            "api_key": sf_key,
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
            "name": "SiliconFlow Qwen"
        }

    # 默认使用手册中的配置 (需用户自行填入)
    return {
        "api_key": "your-api-key-here",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "name": "SiliconFlow Qwen (请配置API Key)"
    }


BACKEND = _detect_backend()
client = OpenAI(api_key=BACKEND["api_key"], base_url=BACKEND["base_url"])
MODEL = BACKEND["model"]


def call_llm(messages, model=None, temperature=0.7):
    """调用大模型并返回回复内容"""
    try:
        response = client.chat.completions.create(
            model=model or MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API调用出错: {e}"


# ==================== 演示函数 ====================

def sentiment_classification_demo():
    """情感分类: 零样本 vs 少样本"""
    print("\n" + "=" * 50)
    print("--- 1. 情感分类: 零样本 vs 少样本 ---")

    # 零样本
    zero_shot = [
        {"role": "system", "content": "你是一个情感分析助手，只输出'正面'或'负面'。"},
        {"role": "user", "content": "这部电影真是无聊透顶。"}
    ]
    result_zero = call_llm(zero_shot, temperature=0)
    print(f"零样本: {result_zero}")

    # 少样本 (Few-shot)
    few_shot = [
        {"role": "system", "content": "你是一个情感分析助手。"},
        {"role": "user", "content": (
            "示例1: 文本='太棒了' -> 正面\n"
            "示例2: 文本='我讨厌这个' -> 负面\n\n"
            "现在请判断: 文本='还行吧，凑合' ->"
        )}
    ]
    result_few = call_llm(few_shot, temperature=0)
    print(f"少样本: {result_few}")


def information_extraction_demo():
    """信息抽取: 角色扮演 + JSON格式约束"""
    print("\n" + "=" * 50)
    print("--- 2. 信息抽取: 角色扮演 + JSON输出 ---")

    messages = [
        {"role": "system", "content": (
            "你是一个专业的信息抽取专家。请从用户输入的文本中提取"
            "'时间'、'地点'、'人物'三个字段，以严格的JSON格式输出，"
            "不要输出其他内容。"
        )},
        {"role": "user", "content": "昨天下午，张三在北京的会议室里向李四汇报了项目进展。"}
    ]
    result = call_llm(messages, temperature=0)
    print(f"抽取结果:\n{result}")

    # 尝试解析JSON
    try:
        # 清洗常见格式问题
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean.strip())
        print(f"✅ 成功解析JSON: {json.dumps(parsed, ensure_ascii=False, indent=2)}")
    except Exception:
        print("⚠️ 输出不是有效JSON，可能需要进一步优化提示词。")


def chain_of_thought_demo():
    """思维链推理: 数学问题"""
    print("\n" + "=" * 50)
    print("--- 3. 思维链(CoT)推理: 数学问题 ---")

    question = "一个商店先降价20%，再涨价20%，最终价格与原价相比是高了、低了还是不变？"

    # 不加CoT
    direct = call_llm([{"role": "user", "content": question}], temperature=0)
    print(f"不加CoT:\n{direct}")

    # 加CoT
    cot = call_llm(
        [{"role": "user", "content": f"{question}\n让我们一步步思考，最后给出结论。"}],
        temperature=0
    )
    print(f"\n加CoT:\n{cot}")


def steel_defect_prompt_demo():
    """钢铁缺陷检测提示词专项演示 (扩展)"""
    print("\n" + "=" * 50)
    print("--- 4. 钢铁缺陷检测VLM提示词 (角色扮演) ---")

    system_prompt = (
        "你是一名拥有20年经验的钢铁质量检测专家。"
        "请分析图像中的钢板表面，判断是否存在以下缺陷类型:\n"
        "- 裂纹(crack): 黑色锯齿状裂缝\n"
        "- 划痕(scratch): 白色/灰色直线或弯曲条痕\n"
        "- 压痕(indentation): 圆形凹陷\n"
        "- 氧化皮(scale): 不规则片状暗区\n"
        "- 气泡(blister): 圆形隆起\n\n"
        "以JSON格式返回: {\"detections\": [{\"type\": \"...\", "
        "\"severity\": \"SEVERE|MODERATE|MILD\", "
        "\"description\": \"位置和外观描述\", "
        "\"confidence\": 0.0-1.0}]}"
    )

    print("系统提示词 (角色扮演):")
    print(system_prompt)
    print("\n✅ 此提示词已集成到 VLM 检测引擎中 (src/vlm_engine.py)")


# ==================== 主入口 ====================

def main():
    print("=" * 50)
    print(f"实训1: 大模型API调用与提示词工程")
    print(f"后端: {BACKEND['name']} | 模型: {MODEL}")
    print("=" * 50)

    try:
        sentiment_classification_demo()
    except Exception as e:
        print(f"情感分类演示失败: {e}")

    try:
        information_extraction_demo()
    except Exception as e:
        print(f"信息抽取演示失败: {e}")

    try:
        chain_of_thought_demo()
    except Exception as e:
        print(f"思维链演示失败: {e}")

    steel_defect_prompt_demo()

    print("\n" + "=" * 50)
    print("实训1 完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()

"""
实训2: 基于RAG的质检知识库检索增强

构建钢铁缺陷知识库，通过检索增强生成实现根因分析。
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


# ==================== API 配置 ====================
def _get_client():
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return OpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        ), "gemini-2.5-flash", "Gemini"

    key = os.getenv("SILICONFLOW_API_KEY", "")
    if key:
        return OpenAI(
            api_key=key,
            base_url="https://api.siliconflow.cn/v1"
        ), "Qwen/Qwen3-235B-A22B-Instruct-2507", "SiliconFlow"

    return None, None, None


client, MODEL, BACKEND = _get_client()


# ==================== 知识库 ====================

KNOWLEDGE_BASE = [
    {
        "keywords": ["氧化铁皮", "氧化皮", "鱼鳞", "红色", "压入", "rolled-in_scale", "scale"],
        "defect_type": "氧化铁皮压入",
        "content": (
            "【缺陷类型】氧化铁皮压入 (Rolled-in Scale)\n"
            "【外观特征】呈鱼鳞状或片状，红棕色至暗灰色，嵌入钢板表面。\n"
            "【可能原因】\n"
            "  1. 除鳞系统压力不足 (<15MPa)，高压水未能有效清除表面氧化皮；\n"
            "  2. 轧辊表面粗糙度过高，将氧化皮压入钢基体；\n"
            "  3. 上游加热炉内氧化气氛过重，一次氧化皮过厚。\n"
            "【建议措施】\n"
            "  1. 检查并调整除鳞泵压力至18-22MPa；\n"
            "  2. 定期更换或修磨轧辊，保持表面光洁度Ra<1.6μm；\n"
            "  3. 优化加热炉空燃比，减少氧化烧损。\n"
            "【严重程度分级】\n"
            "  轻度: 零星分布，面积<5%；中度: 条带状，面积5-15%；重度: 大面积覆盖，面积>15%"
        )
    },
    {
        "keywords": ["划痕", "scratch", "直线", "沟槽", "机械损伤", "条痕"],
        "defect_type": "划痕",
        "content": (
            "【缺陷类型】划痕 (Scratch)\n"
            "【外观特征】沿轧制方向或随机方向的直线/弯曲沟槽，颜色较基体更亮。\n"
            "【可能原因】\n"
            "  1. 导卫板磨损严重或粘附异物，与带钢表面摩擦；\n"
            "  2. 轧辊边部有毛刺或崩裂；\n"
            "  3. 运输辊道速度不匹配，板间滑动摩擦。\n"
            "【建议措施】\n"
            "  1. 检查并更换磨损导卫板，确保表面光滑无毛刺；\n"
            "  2. 清理轧辊边部，必要时修磨或更换；\n"
            "  3. 优化辊道速度同步控制，避免板间相对滑动。\n"
            "【严重程度分级】\n"
            "  轻度: 深度<0.05mm；中度: 深度0.05-0.15mm；重度: 深度>0.15mm"
        )
    },
    {
        "keywords": ["点蚀", "麻点", "pitted_surface", "坑", "腐蚀", "凹坑"],
        "defect_type": "麻点/点蚀",
        "content": (
            "【缺陷类型】麻点/点蚀 (Pitted Surface)\n"
            "【外观特征】表面密集分布的微小凹坑，直径0.5-3mm，深度较浅。\n"
            "【可能原因】\n"
            "  1. 冷却水水质不佳（Cl⁻含量>50ppm），导致局部电化学腐蚀；\n"
            "  2. 钢中夹杂物（MnS、Al₂O₃）暴露于表面后被腐蚀脱落；\n"
            "  3. 酸洗过度或酸液残留，造成过腐蚀。\n"
            "【建议措施】\n"
            "  1. 改善冷却水质，控制Cl⁻<30ppm，添加缓蚀剂；\n"
            "  2. 优化炼钢脱氧工艺，减少非金属夹杂物；\n"
            "  3. 控制酸洗时间和温度，确保充分冲洗。\n"
            "【严重程度分级】\n"
            "  轻度: 零星分布；中度: 局部密集；重度: 大面积密集分布"
        )
    },
    {
        "keywords": ["裂纹", "crack", "crazing", "裂缝", "锯齿", "开裂"],
        "defect_type": "裂纹/龟裂",
        "content": (
            "【缺陷类型】裂纹/龟裂 (Crack/Crazing)\n"
            "【外观特征】不规则网状或线状开裂，呈黑色锯齿状，深度较大。\n"
            "【可能原因】\n"
            "  1. 轧制温度过低 (<Ar₃相变点)，材料塑性不足产生应力裂纹；\n"
            "  2. 冷却速率过快，产生过大的热应力；\n"
            "  3. 钢中氢含量过高，产生氢致裂纹；\n"
            "  4. 铸坯原有皮下气泡或缩孔在轧制中扩展。\n"
            "【建议措施】\n"
            "  1. 严格控制终轧温度在Ar₃以上30-50℃；\n"
            "  2. 优化层流冷却制度，避免急冷；\n"
            "  3. 加强炼钢脱气处理，控制[H]<2ppm；\n"
            "  4. 对铸坯进行表面检查和修磨。\n"
            "【严重程度分级】\n"
            "  轻度: 微裂纹，长度<5mm；中度: 可见裂纹，5-20mm；重度: 贯穿性裂纹，>20mm"
        )
    },
    {
        "keywords": ["斑块", "patches", "色差", "黑斑", "白斑", "不均匀"],
        "defect_type": "表面斑块",
        "content": (
            "【缺陷类型】表面斑块 (Patches)\n"
            "【外观特征】局部颜色异常区域，呈暗色或亮色不规则斑块。\n"
            "【可能原因】\n"
            "  1. 表面局部氧化不均，与冷却水分布不均有关；\n"
            "  2. 轧制油或乳化液残留，在退火过程中碳化；\n"
            "  3. 来料表面原始锈蚀未清除干净。\n"
            "【建议措施】\n"
            "  1. 检查冷却水喷嘴，确保均匀覆盖；\n"
            "  2. 优化轧制润滑液配比和吹扫系统；\n"
            "  3. 加强酸洗工序管理，确保表面清洁。\n"
            "【严重程度分级】\n"
            "  轻度: 淡色斑；中度: 明显色差；重度: 深色碳化斑"
        )
    },
    {
        "keywords": ["夹杂", "inclusion", "夹渣", "异物", "亮点", "白点"],
        "defect_type": "非金属夹杂",
        "content": (
            "【缺陷类型】非金属夹杂 (Inclusion)\n"
            "【外观特征】表面可见的亮点、白点或不规则异物嵌入。\n"
            "【可能原因】\n"
            "  1. 炼钢过程中脱氧产物（Al₂O₃、SiO₂）未充分上浮；\n"
            "  2. 连铸过程中中间包覆盖剂或结晶器保护渣卷入；\n"
            "  3. 钢包或中间包耐火材料侵蚀脱落。\n"
            "【建议措施】\n"
            "  1. 优化脱氧工艺，采用Ca处理改性夹杂物；\n"
            "  2. 加强中间包冶金效果，保证足够停留时间（>8min）；\n"
            "  3. 选用优质耐火材料，定期检查和更换。\n"
            "【严重程度分级】\n"
            "  轻度: 零星微细夹杂；中度: 可见夹杂物；重度: 大型夹杂或聚集"
        )
    },
    {
        "keywords": ["压痕", "indentation", "凹痕", "凹陷", "圆形坑", "撞击"],
        "defect_type": "压痕/凹陷",
        "content": (
            "【缺陷类型】压痕/凹陷 (Indentation)\n"
            "【外观特征】局部圆形或椭圆形凹陷，边缘光滑，深度明显。\n"
            "【可能原因】\n"
            "  1. 异物（铁屑、焊渣）掉落在带钢表面后被轧辊压入；\n"
            "  2. 轧辊表面有凹坑或剥落，周期性复制到带钢表面；\n"
            "  3. 卷取时张力不当导致层间压痕。\n"
            "【建议措施】\n"
            "  1. 加强轧线清洁，定期清理轧辊及导卫间的异物；\n"
            "  2. 检查轧辊表面状况，发现凹坑及时修磨或更换；\n"
            "  3. 优化卷取张力控制，避免层间滑移。\n"
            "【严重程度分级】\n"
            "  轻度: 深度<0.1mm；中度: 深度0.1-0.3mm；重度: 深度>0.3mm"
        )
    },
    {
        "keywords": ["气泡", "blister", "隆起", "鼓包", "凸起"],
        "defect_type": "气泡/鼓包",
        "content": (
            "【缺陷类型】气泡/鼓包 (Blister)\n"
            "【外观特征】表面圆形隆起，内部中空，轻敲有空响声。\n"
            "【可能原因】\n"
            "  1. 钢中氢含量过高，在冷却过程中析出形成皮下气泡；\n"
            "  2. 铸坯皮下气孔在轧制中未能焊合；\n"
            "  3. 酸洗时氢原子渗入钢基体，退火时聚集膨胀。\n"
            "【建议措施】\n"
            "  1. 加强炼钢脱气处理，控制[H]<2ppm；\n"
            "  2. 检查连铸保护浇注效果，防止二次氧化吸气；\n"
            "  3. 优化酸洗工艺参数，减少氢渗入。\n"
            "【严重程度分级】\n"
            "  轻度: 零星微气泡；中度: 局部密集气泡；重度: 大面积气泡群"
        )
    },
]


# ==================== 检索与生成 ====================

def retrieve_context(query: str, kb: list = None) -> tuple:
    """基于关键词匹配检索最相关的知识条目"""
    if kb is None:
        kb = KNOWLEDGE_BASE

    query_lower = query.lower()
    best_match = None
    best_score = 0

    for item in kb:
        score = sum(1 for kw in item["keywords"] if kw.lower() in query_lower)
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score > 0:
        return best_match["content"], best_match["defect_type"], best_score
    return None, None, 0


def call_llm(messages, temperature=0):
    """调用大模型"""
    if client is None:
        return "API未配置，请在.env中设置GEMINI_API_KEY或SILICONFLOW_API_KEY"
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API调用出错: {e}"


def rag_analyze(defect_type: str, description: str = "") -> str:
    """
    完整的RAG流程: 检索知识库 → 增强生成根因分析
    
    Args:
        defect_type: 缺陷类型 (如 'crack', 'scratch')
        description: 缺陷外观描述
    Returns:
        根因分析报告
    """
    # 中英文类型映射
    type_map = {
        "crack": "裂纹", "crazing": "裂纹",
        "scratch": "划痕", "scratches": "划痕",
        "scale": "氧化铁皮", "rolled-in_scale": "氧化铁皮",
        "indentation": "压痕", "pitted_surface": "麻点",
        "blister": "气泡", "patches": "斑块",
        "inclusion": "夹杂",
    }
    cn_type = type_map.get(defect_type.lower(), defect_type)

    # 检索
    query = f"{cn_type} {description}"
    context, matched_type, score = retrieve_context(query)

    if context is None:
        # 尝试只用中文名检索
        context, matched_type, score = retrieve_context(cn_type)

    if context is None:
        return f"## {cn_type} 根因分析\n\n⚠️ 知识库中未找到 '{cn_type}' 的相关知识。\n建议: 请补充知识库或咨询工艺专家。"

    # 生成
    prompt = (
        "你是一名钢铁质量分析专家。请基于以下参考知识，"
        "对检测到的缺陷进行专业的根因分析，给出具体的改进建议。\n\n"
        f"【检测到的缺陷】\n"
        f"  类型: {cn_type}\n"
        f"  描述: {description or '详见检测图像'}\n\n"
        f"【参考知识】\n{context}\n\n"
        "请以简洁的要点形式输出：\n"
        "1. 缺陷确认\n"
        "2. 最可能的原因 (1-2条)\n"
        "3. 建议的检查/处理措施\n"
        "4. 是否需要停机处理"
    )

    messages = [{"role": "user", "content": prompt}]
    answer = call_llm(messages)

    return (
        f"## {cn_type} 根因分析报告\n\n"
        f"📚 知识库匹配: {matched_type} (关键词匹配度: {score})\n\n"
        f"---\n\n{answer}"
    )


# ==================== 演示 ====================

def demo():
    """演示RAG vs 无RAG的差异"""
    print("=" * 60)
    print(f"实训2: 基于RAG的质检知识库检索增强")
    print(f"后端: {BACKEND or '未配置'} | 知识库条目: {len(KNOWLEDGE_BASE)}")
    print("=" * 60)

    test_queries = [
        ("scratch", "右上角白色倾斜条痕，深度较浅"),
        ("crack", "中部偏左黑色锯齿状裂缝，长度约15mm"),
        ("pitted_surface", "左下角密集微小凹坑"),
    ]

    for dtype, desc in test_queries:
        print(f"\n{'─' * 50}")
        report = rag_analyze(dtype, desc)
        print(report)

    print(f"\n{'=' * 60}")
    print("实训2 完成!")
    print("=" * 60)


def main():
    # 可以直接调用 rag_analyze 进行根因分析
    demo()


if __name__ == "__main__":
    main()

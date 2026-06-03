"""
RAG 根因分析模块 — 结合国家标准（GB/T）与大模型生成专业处置报告。
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional
from openai import OpenAI

# 获取项目根目录及数据库路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "inspection.db"

# 缺陷类别名称映射
DEFECT_MAP = {
    "crazing": "裂纹/龟裂",
    "crack": "裂纹",
    "inclusion": "非金属夹杂",
    "patches": "色差斑块",
    "pitted_surface": "麻点凹坑",
    "rolled-in_scale": "轧制氧化皮",
    "rolled_in_scale": "轧制氧化皮",
    "scale": "氧化皮",
    "scratches": "划痕/擦伤",
    "scratch": "划痕",
    "rust": "铁皮锈蚀",
    "blister": "起皮气泡",
}

def _get_api_client() -> tuple[Optional[OpenAI], str]:
    """探测 API 客户端与模型"""
    # 尝试读取环境变量
    api_key = os.getenv("VLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or ""
    if not api_key:
        return None, ""
        
    # 自定义或默认 Gemini Endpoint
    base_url = os.getenv("VLM_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
    model = os.getenv("VLM_MODEL") or "gemini-2.5-flash"
    
    if os.getenv("DASHSCOPE_API_KEY"):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen-vl-max"
        
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=15,
            max_retries=2
        )
        return client, model
    except Exception:
        return None, ""

def query_knowledge_base(defect_type: str, vlm_desc: str = "") -> list[dict]:
    """从数据库中检索匹配的标准和成因"""
    results = []
    if not os.path.exists(str(DB_PATH)):
        return results
        
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. 尝试缺陷类别精准匹配
    mapped_type = defect_type.lower().strip()
    cursor.execute(
        "SELECT defect_type, title, standard_code, content FROM knowledge_base WHERE defect_type = ?",
        (mapped_type,)
    )
    rows = cursor.fetchall()
    
    # 2. 如果精准匹配无结果，尝试模糊搜索或关键字搜索
    if not rows:
        keywords = [mapped_type]
        if vlm_desc:
            # 提取简单的中文或英文关键字
            for word in ["裂纹", "夹杂", "斑块", "麻点", "氧化皮", "划痕", "锈蚀", "气泡", "crack", "scratch", "rust", "scale"]:
                if word in vlm_desc:
                    keywords.append(word)
                    
        for kw in set(keywords):
            cursor.execute(
                "SELECT defect_type, title, standard_code, content FROM knowledge_base WHERE defect_type LIKE ? OR title LIKE ? OR content LIKE ?",
                (f"%{kw}%", f"%{kw}%", f"%{kw}%")
            )
            rows.extend(cursor.fetchall())
            
    # 去重
    seen = set()
    for row in rows:
        r_dict = dict(row)
        key = (r_dict["standard_code"], r_dict["title"])
        if key not in seen:
            seen.add(key)
            results.append(r_dict)
            
    conn.close()
    return results

def rag_analyze(defect_type: str, vlm_desc: str = "") -> str:
    """RAG 根因分析入口函数"""
    defect_cn = DEFECT_MAP.get(defect_type.lower(), defect_type)
    
    # 1. 从 SQLite 检索知识库条目
    knowledge_items = query_knowledge_base(defect_type, vlm_desc)
    
    if not knowledge_items:
        # 兜底返回静态的分析
        return f"""<div style="border-left: 4px solid #e63946; padding: 12px; background: #fff5f5; border-radius: 6px; font-family: system-ui; font-size: 13px; line-height: 1.6">
            <h4 style="margin: 0 0 6px 0; color: #e63946; font-weight: bold">⚠️ 未匹配到国标规范</h4>
            <p style="margin: 0; color: #555">系统检测到缺陷类型为 <b>{defect_cn}</b> ({defect_type})。当前本地知识库中未找到完全匹配的国家生产标准规范。</p>
            <p style="margin: 6px 0 0 0; color: #777">建议人工核实缺陷等级并参考常规工艺进行纠偏。</p>
        </div>"""
        
    # 拼接参考标准文本
    references_text = ""
    for item in knowledge_items:
        references_text += f"标准名称: {item['title']} ({item['standard_code']})\n内容:\n{item['content']}\n\n"
        
    # 2. 尝试调用大模型生成高级 RAG 报告
    client, model = _get_api_client()
    if client:
        try:
            prompt = f"""你是一名资深的钢铁冶金及轧钢工艺专家。你的任务是根据图像智能分析出的缺陷信息，结合提供的国家标准（GB/T）规范，为生产车间出具一份极具工业级专业度、严谨的【钢板表面缺陷根因与工艺处置分析报告】。

## 待分析数据
- 检出缺陷类型: {defect_cn} ({defect_type})
- 视觉特征描述: {vlm_desc if vlm_desc else "待查，请参考国标做典型诊断"}

## 参考国家标准及工业机理
{references_text}

## 报告输出规范
你必须生成结构化的 HTML，包含以下板块：
1. **📑 规范比对 (GB/T Alignment)**: 明确引用哪项国标，判定缺陷是否超标，给出判等结论（例如：A级合格 / 降级接收 / 判废切除）。
2. **🔬 根因溯源 (Root Cause)**: 从炼钢、连铸、热轧/冷轧工艺参数（如温度、下压量、保护渣、轧辊粗糙度等）全链路细致推导产生此缺陷的原因。
3. **🛠️ 工艺纠偏建议 (Process Optimization)**: 给出3条可在实际车间落地的具体工艺优化动作，必须带有具体的工程参数（如：温度、压力、配料比例等）。

输出要求：
- 直接返回 HTML，不要用 ```html ``` 代码块包裹。
- 样式必须符合工业精装风，使用扁平卡片、微阴影和高对比度边框。
- 语气必须严谨、专业、科学。
"""
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一名严谨的钢铁制造质检专家，只输出漂亮的 HTML 报告，不需要任何多余的前言或后记。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.2
            )
            report_content = response.choices[0].message.content or ""
            if report_content.strip():
                return report_content.strip()
        except Exception as e:
            # 接口异常时回退到本地模板生成
            pass
            
    # 3. 本地 RAG 模板拼装（Offline Fallback）
    # 当没有网络或 API 调用失败时，将数据库中的内容拼装成极其高档的 HTML 报告
    fallback_htmls = []
    for item in knowledge_items:
        content_lines = item["content"].split("\n")
        gb_clause = ""
        cause_analysis = ""
        harm_eval = ""
        action_plan = ""
        
        for line in content_lines:
            if "【国标规范】" in line:
                gb_clause = line.replace("【国标规范】", "").strip()
            elif "【成因分析】" in line:
                cause_analysis = line.replace("【成因分析】", "").strip()
            elif "【危害评估】" in line:
                harm_eval = line.replace("【危害评估】", "").strip()
            elif "【工艺建议】" in line:
                action_plan = line.replace("【工艺建议】", "").strip()
                
        # 兜底填充
        if not gb_clause: gb_clause = item["content"]
        
        # 拼装专业 HTML 卡片
        card = f"""
        <div style="font-family: 'Segoe UI', system-ui, sans-serif; border: 1px solid #e2e8f0; border-radius: 12px; 
                    background: #ffffff; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.03); overflow: hidden">
            <!-- 头部 -->
            <div style="background: linear-gradient(135deg, #1565c0, #0d47a1); padding: 12px 18px; color: #ffffff; 
                        display: flex; justify-content: space-between; align-items: center">
                <span style="font-size: 15px; font-weight: 800; letter-spacing: 0.5px">📑 工业标准比对报告</span>
                <span style="background: rgba(255,255,255,0.22); color: #ffffff; font-size: 11px; font-weight: 700; 
                             padding: 3px 10px; border-radius: 20px">{item['standard_code']}</span>
            </div>
            
            <div style="padding: 18px">
                <!-- 1. 标准条文 -->
                <div style="margin-bottom: 14px">
                    <div style="font-weight: 800; font-size: 13px; color: #1565c0; margin-bottom: 4px">【对应国标】{item['title']}</div>
                    <div style="font-size: 13px; color: #2c3e50; line-height: 1.6; background: #f8fafc; padding: 10px 14px; 
                                border-left: 3px solid #1565c0; border-radius: 4px">
                        {gb_clause}
                    </div>
                </div>
                
                <!-- 2. 原因分析 -->
                {f'''<div style="margin-bottom: 14px">
                    <div style="font-weight: 800; font-size: 13px; color: #ff6b35; margin-bottom: 4px">🔬 物理根因分析</div>
                    <div style="font-size: 13px; color: #475569; line-height: 1.6">
                        {cause_analysis}
                    </div>
                </div>''' if cause_analysis else ''}
                
                <!-- 3. 危害评估 -->
                {f'''<div style="margin-bottom: 14px">
                    <div style="font-weight: 800; font-size: 13px; color: #e63946; margin-bottom: 4px">⚡ 质量与结构危害</div>
                    <div style="font-size: 13px; color: #475569; line-height: 1.6">
                        {harm_eval}
                    </div>
                </div>''' if harm_eval else ''}
                
                <!-- 4. 工艺动作 -->
                {f'''<div style="margin-top: 16px; border-top: 1px dashed #e2e8f0; padding-top: 14px">
                    <div style="font-weight: 800; font-size: 13px; color: #2a9d8f; margin-bottom: 6px">🛠️ 车间工艺纠偏动作</div>
                    <div style="font-size: 13px; color: #2c3e50; line-height: 1.6; background: #f0fdf4; padding: 10px 14px; 
                                border-left: 3px solid #2a9d8f; border-radius: 4px">
                        {action_plan}
                    </div>
                </div>''' if action_plan else ''}
            </div>
        </div>
        """
        fallback_htmls.append(card)
        
    return "\n".join(fallback_htmls)

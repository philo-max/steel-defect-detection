/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import express from "express";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Set up larger JSON body limits to support raw base64 structural uploading of 2K steel plate photos
app.use(express.json({ limit: "15mb" }));

// Initialize Google GenAI client lazily to prevent crash on startup if key is missing
let aiClient: GoogleGenAI | null = null;
function getAiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
    return null;
  }
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        }
      }
    });
  }
  return aiClient;
}

// 1. Defect Response Schema for Gemini JSON alignment
const responseSchema = {
  type: Type.OBJECT,
  properties: {
    overallStatus: {
      type: Type.STRING,
      description: "Overall steel coil batch quality status. MUST be 'Pass', 'Fail' or 'Marginal'."
    },
    severityIndex: {
      type: Type.INTEGER,
      description: "Severity index representing product hazard, from 0 (clean steel) to 100 (critical damage/scrap)."
    },
    defectDensity: {
      type: Type.NUMBER,
      description: "Estimated percentage area of defect density over total visible area (0.0% to 100.0%)."
    },
    defects: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          type: {
            type: Type.STRING,
            description: "Defect class: 'Scratches' | 'Cracks' | 'Pitting' | 'Inclusions' | 'Scale' | 'Patches'."
          },
          typeName: {
            type: Type.STRING,
            description: "Detailed description of defect taxonomy in Chinese, e.g. 纵向贯穿划痕, 辊面点蚀, 局部氧化皮残留."
          },
          description: {
            type: Type.STRING,
            description: "Detailed technical inspection remarks in Chinese, specifying size, spread, and critical evaluation."
          },
          severity: {
            type: Type.STRING,
            description: "Hazard evaluation. MUST be 'Low', 'Medium', or 'High'."
          },
          bbox: {
            type: Type.ARRAY,
            items: { type: Type.INTEGER },
            description: "Bounding box of defect as four normalized integers [ymin, xmin, ymax, xmax] scaled 0 to 100."
          },
          confidence: {
            type: Type.NUMBER,
            description: "AI confidence score of identification from 0.00 to 1.00."
          }
        },
        required: ["type", "typeName", "description", "severity", "bbox", "confidence"]
      },
      description: "List of all scanned defect regions detected inside the frame."
    },
    chemicalExplanation: {
      type: Type.STRING,
      description: "Metallurgical mechanics and thermal-chemical analysis in Chinese, explaining precisely why and how this thermal/physical defect originated in casting, reheating, or rolling steps."
    },
    recommendedAction: {
      type: Type.STRING,
      description: "Operational reprocessing workflow recommendation in Chinese, e.g. 降级接收并作为内部件, 离线酸洗抛磨修复, 物理剪切切除缺陷段, or 直接废品报废重熔."
    }
  },
  required: ["overallStatus", "severityIndex", "defectDensity", "defects", "chemicalExplanation", "recommendedAction"]
};

// Simulated Fallback Metallurgy Analytics Database
const MOCKED_ANSWERS: Record<string, any> = {
  clean: {
    overallStatus: "Pass",
    severityIndex: 4,
    defectDensity: 0.0,
    defects: [],
    chemicalExplanation: "钢胚精炼脱硫工艺到位，连铸结晶器液位控制平稳。热轧及冷轧阶段辊面光洁度保持优良，乳化液循环过滤和流速控制精密，轧制力分配均匀，未发生机械擦伤或咬入杂物，防锈钝化膜致密无缺陷。",
    recommendedAction: "产品表面质量评定为 A 级，符合一类卷材合格标准。无需修复，建议立即挂牌入库并发货。"
  },
  scratch: {
    overallStatus: "Fail",
    severityIndex: 78,
    defectDensity: 6.8,
    defects: [
      {
        id: "s1",
        type: "Scratches",
        typeName: "机械辊印拉应力纵向划痕",
        description: "板材上表面中部可见明显的纵向摩擦划痕，走势平行于轧制方向。沟槽剖面呈V型，伴有微弱卷边金属撕裂凸起。",
        severity: "High",
        bbox: [15, 20, 28, 85],
        confidence: 0.94
      },
      {
        id: "s2",
        type: "Scratches",
        typeName: "侧边冷校直微划痕",
        description: "板带下方靠近边缘有轻微摩擦亮条纹，属于卷取机侧导板开口度过窄导致的机械碰伤。",
        severity: "Low",
        bbox: [70, 50, 78, 88],
        confidence: 0.81
      }
    ],
    chemicalExplanation: "主因是轧机工作辊或张力辊表面粘附了超硬的氧化铁皮硬质小颗粒，或者导向滑板松动错位，导致高速运动的钢带在连续滑擦中产生贯穿性的表面拉伤。属于典型热变形机械损伤。",
    recommendedAction: "考虑到严重的贯穿性拉伤影响抗拉强度与深冲性能。建议：① 对表面进行打磨评级；② 若深度超标，对该部分缺陷段进行物理剪切切除分段；③ 立即停机检查各道轧辊面及清扫器清洁度。"
  },
  crack: {
    overallStatus: "Fail",
    severityIndex: 92,
    defectDensity: 11.4,
    defects: [
      {
        id: "c1",
        type: "Cracks",
        typeName: "板材边部热应力晶间龟裂纹",
        description: "板坯右侧边缘发生严重的锯齿撕裂裂口，并沿晶界向内陆延伸呈现树枝状微裂纹分支，缝隙深度较大。",
        severity: "High",
        bbox: [22, 38, 62, 95],
        confidence: 0.97
      }
    ],
    chemicalExplanation: "此种边裂多由于加热炉内温度不均或边缘冷却速度过快产生极高的内应力梯度。在粗轧机大压下量轧制时，边缘拉应力超过了钢种的极限塑性变形阈值，导致金属原子在晶界处撕离并沿应力最大断面迅速扩张。",
    recommendedAction: "产品质量严重超标，评定为 C 等废品。建议：① 立即调送剪切线，进行宽边切除（两侧切边不低于100mm）；② 切边后对中心残余钢带重新进行探伤评估；③ 若断口有空洞或分层，直接废品回装电炉重融。"
  },
  pitting: {
    overallStatus: "Marginal",
    severityIndex: 58,
    defectDensity: 8.5,
    defects: [
      {
        id: "p1",
        type: "Pitting",
        typeName: "酸洗过度腐蚀性聚集点蚀麻面",
        description: "全板面随机密集分布呈黑色凹坑斑点，手感粗糙，并呈现部分鳞状剥落腐蚀层。剥落层底部伴有铁锈氧化残留。",
        severity: "Medium",
        bbox: [12, 18, 88, 82],
        confidence: 0.89
      }
    ],
    chemicalExplanation: "板材在经过连续酸洗线时，因中途意外停机或带钢速度不足，导致其在酸槽中浸泡时间显著超额（过酸洗）。高温强酸介质优先攻击板带内的晶界交界处和夹杂区，造成金属氧化层下出现局部不均匀凹坑，严重削弱抗指纹及表面电镀附着力。",
    recommendedAction: "属于表面装饰性与涂敷失效级缺陷。建议：① 严禁直接送高要求家电卷或冷轨底板生产线；② 调拨至次级包装带或经表面刷棉球高速抛光除锈降级处理；③ 纠正酸洗线联锁速比控制程序。"
  },
  scale: {
    overallStatus: "Fail",
    severityIndex: 82,
    defectDensity: 14.2,
    defects: [
      {
        id: "sc1",
        type: "Scale",
        typeName: "热轧残留原生态铁素体氧化皮",
        description: "表面夹杂块状暗黑色氧化层，与基体金属界限分明，面积较大，部分区域已呈现铁锈红层碳化，在冷轧碾压后脱位边缘明显。",
        severity: "High",
        bbox: [15, 32, 58, 82],
        confidence: 0.95
      },
      {
        id: "sc2",
        type: "Inclusions",
        typeName: "保护 slag 精炼非金属夹杂物痕",
        description: "在钢板中下侧点状散落几处黄褐色或暗黄色熔渣形变细条条，系轧制后拉长变形的伴生斑疤。",
        severity: "Medium",
        bbox: [65, 12, 75, 48],
        confidence: 0.88
      }
    ],
    chemicalExplanation: "前者产生于热连轧粗轧之前的除鳞工艺异常。高压水嘴局部堵塞或喷水压力缺失导致生长的氧化铁皮（FeO/Fe3O4）未能被高压水剥离剥尽。后者乃是连铸期间精炼炉脱氧不足形成的脱氧产物或保护渣，卷入铸坯表层并在连铸弯曲段冷却收缩时硬化固缩，轧制时在钢基中挤压形成异相沉积物。",
    recommendedAction: "该批次表面硬质夹渣和大面积氧化皮已破坏组织连续性，后续冷弯容易产生拉应力突变爆开。建议：① 进行二次物理抛光拉拔试样；② 直接分流作底结构板或中厚无外观要求的粗管钢；③ 严格检修高压除鳞泵组水压阀。"
  }
};

// User authentication accounts configuration
const DEFAULT_PW = process.env.APP_DEFAULT_PASSWORD || crypto.randomBytes(12).toString("base64url");
const USER_ACCOUNTS: Record<string, [string, string]> = {
  admin: [process.env.ACCOUNT_PW_admin || DEFAULT_PW, "admin"],
  inspector: [process.env.ACCOUNT_PW_inspector || DEFAULT_PW, "inspector"],
  supervisor: [process.env.ACCOUNT_PW_supervisor || DEFAULT_PW, "supervisor"],
  ai_engineer: [process.env.ACCOUNT_PW_ai_engineer || DEFAULT_PW, "ai_engineer"],
  process_engineer: [process.env.ACCOUNT_PW_process_engineer || DEFAULT_PW, "process_engineer"],
};

const TOKEN_SECRET = process.env.TOKEN_SECRET || crypto.randomBytes(32).toString("hex");

function createToken(username: string): string {
  const ts = Math.floor(Date.now() / 1000);
  const payload = `${username}:${ts}`;
  const sig = crypto.createHmac("sha256", TOKEN_SECRET).update(payload).digest("hex").slice(0, 16);
  return `${username}:${ts}:${sig}`;
}

function verifyToken(token: string): boolean {
  try {
    const parts = token.split(":");
    if (parts.length !== 3) return false;
    const [username, tsStr, sig] = parts;
    const ts = parseInt(tsStr, 10);
    const payload = `${username}:${ts}`;
    const expected = crypto.createHmac("sha256", TOKEN_SECRET).update(payload).digest("hex").slice(0, 16);
    if (sig !== expected) return false;
    if (Math.abs(Date.now() / 1000 - ts) > 86400) return false;
    return true;
  } catch {
    return false;
  }
}

// 1.5 LOGIN & LOGOUT APIs
app.post("/api/login", (req, res) => {
  const { username, password } = req.body;
  const u = username?.trim();
  const p = password?.trim();
  if (USER_ACCOUNTS[u] && USER_ACCOUNTS[u][0] === p) {
    const role = USER_ACCOUNTS[u][1];
    const token = createToken(u);
    return res.json({
      success: true,
      token,
      role,
      username: u
    });
  } else {
    return res.json({
      success: false,
      error: "用户名或密码错误"
    });
  }
});

app.post("/api/logout", (req, res) => {
  res.json({ success: true });
});

// 2. HEALTH CHECK API
app.get("/api/health", (req, res) => {
  const isKeyActive = !!process.env.GEMINI_API_KEY && process.env.GEMINI_API_KEY !== "MY_GEMINI_API_KEY";
  res.json({
    status: "healthy",
    timestamp: new Date().toISOString(),
    aiEngine: isKeyActive ? "Gemini 3.5 Active" : "Local Standby Metallurgy Engine Active",
  });
});

// 3. CORE CV DEFECT DETECT API
app.post("/api/detect", async (req, res) => {
  // Session Authorization check
  const authHeader = req.headers["authorization"];
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Unauthorized session. Please log in first." });
  }
  const token = authHeader.slice(7);
  if (!verifyToken(token)) {
    return res.status(401).json({ error: "Invalid or expired token." });
  }

  try {
    const { image, selectedSampleId, filename } = req.body;

    if (!image) {
      return res.status(400).json({ error: "Missing uploaded image base64 data" });
    }

    // Determine sample fallback key if the image represents one of our custom templates
    let detectionResult: any = null;
    let fallbackKey: string | null = null;

    if (selectedSampleId) {
      if (selectedSampleId.includes("clean")) fallbackKey = "clean";
      else if (selectedSampleId.includes("scratch")) fallbackKey = "scratch";
      else if (selectedSampleId.includes("crack")) fallbackKey = "crack";
      else if (selectedSampleId.includes("pitting")) fallbackKey = "pitting";
      else if (selectedSampleId.includes("inclusion") || selectedSampleId.includes("scale")) fallbackKey = "scale";
    }

    const ai = getAiClient();

    if (ai) {
      // 1. Process via Real Google GenAI Gemini 3.5 Flash Model
      try {
        console.log(`[AI INSPECTOR] Booting Gemini-3.5-Flash for visual analysis...`);
        // Clean base64 header
        const base64Clean = image.replace(/^data:image\/\w+;base64,/, "");
        const mimeMatch = image.match(/^data:(image\/\w+);base64,/);
        const mimeType = mimeMatch ? mimeMatch[1] : "image/jpeg";

        const systemMessage = `
          You are a highly professional, state-of-the-art Metallurgical Machine Vision Inspecting AI.
          You analyze surface photos of hot-rolled and cold-rolled steel coils, plates, and slabs.
          Identify and annotate defective areas. Return 100% valid JSON matching the responseSchema exactly.
          
          Category Reference details:
          - Scratches (划痕): linear groove, scoring. Chinese label: 划痕 / 划伤
          - Cracks (裂纹): hairline cracks, fissures. Chinese label: 龟裂 / 裂纹
          - Pitting (麻面): clusters of tiny depressions, corrosion. Chinese label: 辊面拉深凹坑 / 晶间腐蚀麻面
          - Inclusions (夹杂物): non-metallic inclusions, slag. Chinese label: 熔渣夹杂物
          - Scale (氧化皮): dark iron-oxide scale, iron oxide slate. Chinese label: 残留金属氧化皮
          - Patches (斑块): oil stains, uneven passivation spots. Chinese label: 平衡色斑 / 乳化液油污斑块

          Ensure to output bounding boxes (bbox) using normalized [ymin, xmin, ymax, xmax] coordinates scaling from 0 to 100.
          Keep defect coordinates precise so they overlay exactly over the areas that display these defects in the image.
          Write all technical explanation and recommendations in comprehensive, ultra-professional Chinese metallurgical jargon.
        `;

        const response = await ai.models.generateContent({
          model: "gemini-3.5-flash",
          contents: [
            {
              inlineData: {
                mimeType: mimeType,
                data: base64Clean
              }
            },
            {
              text: `Please run a micro-defect structural scan on this steel surface visual print. Return response in strict responseSchema format.`
            }
          ],
          config: {
            systemInstruction: systemMessage,
            responseMimeType: "application/json",
            responseSchema: responseSchema,
            temperature: 0.1, // low temperature for precise factual classifications
          }
        });

        const parsedText = response.text || "{}";
        console.log(`[AI INSPECTOR] Received strict JSON output from Gemini.`);
        const geminiResult = JSON.parse(parsedText);

        // Map defects with IDs
        if (geminiResult && Array.isArray(geminiResult.defects)) {
          geminiResult.defects = geminiResult.defects.map((d: any, index: number) => ({
            ...d,
            id: `gemini_def_${index}_${Date.now()}`
          }));
        }

        detectionResult = geminiResult;
      } catch (geminiError: any) {
        console.error(`[AI INSPECTOR] Gemini actual execution error, falling back to metallurgical heuristics:`, geminiError);
        // Fallback to MOCKED data corresponding to image preset
        if (fallbackKey && MOCKED_ANSWERS[fallbackKey]) {
          detectionResult = JSON.parse(JSON.stringify(MOCKED_ANSWERS[fallbackKey]));
        } else {
          // If uploaded custom photo and Gemini errored out (e.g. rate limit, content block)
          detectionResult = generateDynamicAIHeuristicFallback(filename || "custom_upload.jpg");
        }
        detectionResult.isSimulated = true;
        detectionResult.simulatedReason = `Gemini API 调用异常 (${geminiError.message || "请求过载"})，启用边缘侧专家诊断机制。`;
      }
    } else {
      // 2. Process via metallurgical fallback database
      console.log(`[AI INSPECTOR] No GEMINI_API_KEY environment variable provided. Invoking local metallurgical standby heuristics...`);
      if (fallbackKey && MOCKED_ANSWERS[fallbackKey]) {
        detectionResult = JSON.parse(JSON.stringify(MOCKED_ANSWERS[fallbackKey]));
      } else {
        detectionResult = generateDynamicAIHeuristicFallback(filename || "custom_upload.jpg");
      }
      detectionResult.isSimulated = true;
      detectionResult.simulatedReason = "系统当前运行在「离线专家模式」（如需接入 Gemini 视觉大模型进行实时真实分析，请在后台配置您的 API 密钥）。";
    }

    return res.json({
      success: true,
      timestamp: new Date().toISOString(),
      analyzer: getAiClient() ? "Google Gemini 3.5" : "Standby Physics Expert Core",
      data: detectionResult
    });

  } catch (err: any) {
    console.error("Defect detector critical route server error:", err);
    res.status(500).json({ error: "Metallurgical defect analyzer internal error", details: err.message });
  }
});

// Heuristics generator for custom user photos when offline
function generateDynamicAIHeuristicFallback(filename: string): any {
  // Generate a plausible mock response for custom pictures
  const isCrackWord = /crack|crack|裂纹|破裂/i.test(filename);
  const isScratchWord = /scratch|scratch|划痕|擦伤/i.test(filename);
  const isRustWord = /rust|pitting|腐蚀|麻面/i.test(filename);

  if (isCrackWord) {
    return MOCKED_ANSWERS.crack;
  } else if (isScratchWord) {
    return MOCKED_ANSWERS.scratch;
  } else if (isRustWord) {
    return MOCKED_ANSWERS.pitting;
  } else {
    // Generate a default mild defect report
    return {
      overallStatus: "Marginal",
      severityIndex: 42,
      defectDensity: 4.5,
      defects: [
        {
          id: `h_${Date.now()}_1`,
          type: "Patches",
          typeName: "表面不规则乳化液斑疤",
          description: "带钢中心区域检测到由于乳化液吹扫吹扫不净残留的水油不均斑，轧制形变后呈放射状灰色晕染。",
          severity: "Low",
          bbox: [35, 40, 60, 70],
          confidence: 0.77
        },
        {
          id: `h_${Date.now()}_2`,
          type: "Scratches",
          typeName: "辊面轻微微区微磨损线",
          description: "部分可见非常薄弱的拉丝机械细丝，深度甚微，尚未贯通基底组织。",
          severity: "Low",
          bbox: [10, 5, 20, 45],
          confidence: 0.65
        }
      ],
      chemicalExplanation: "此斑块主要是带钢退火或炉冷却出炉端吹风嘴分布风压微弱失调，导致挥发性防锈油或冷却煤液未能均匀脱附。轧制力未异常，化学腐蚀程度较低。",
      recommendedAction: "外观不敏感，板面强度硬度测试合格。建议：作为次级家电内配隔层板或结构垫圈使用。建议班组定时清洗除鳞水洗风压系统。"
    };
  }
}

// 4. INTEGRATE CLIENT-SIDE VITE ASSETS & MIDDLEWARE
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("[SERVER] Dev Mode detected. Starting Vite development multiplexer...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("[SERVER] Production Mode detected. Serving built static content...");
    const distPath = path.join(process.cwd(), "dist");
    
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`===========================================================`);
    console.log(`  STEEL DEFECT DETECT WORKSTATION HUB RUNNING AT PORT ${PORT}`);
    console.log(`  Access URL: http://localhost:${PORT}`);
    console.log(`  Target binding strictly mapped to 0.0.0.0`);
    console.log(`===========================================================`);
  });
}

startServer();

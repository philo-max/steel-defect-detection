<template>
  <div class="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
    <!-- Header -->
    <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 flex justify-between items-center shrink-0 shadow-lg">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-blue-600/10 rounded-lg border border-blue-500/30">
          <svg viewBox="0 0 24 24" class="w-6 h-6 stroke-blue-400 fill-none stroke-[2]">
            <path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z" />
            <path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-wider text-slate-200">
            钢铁之眼 (Steel Eye) 数字孪生质检系统
          </h1>
          <p class="text-[10px] text-slate-500 font-mono mt-0.5">
            INDUSTRIAL DEFECT MONITORING ENGINE V3.0 • C++ ONNX RUNTIME BACKEND
          </p>
        </div>
      </div>

      <!-- Network and Status Indicators -->
      <div class="flex items-center gap-6">
        <div class="flex items-center gap-2">
          <div class="status-indicator active" :class="store.connected ? 'bg-green-500 text-green-500' : 'bg-amber-500 text-amber-500'"></div>
          <span class="text-xs font-semibold tracking-wider font-mono">
            {{ store.connected ? 'C++ 后端联机 ACTIVE' : '单机仿真 DEMO MODE' }}
          </span>
        </div>
        <div class="text-xs text-slate-400 font-mono">
          {{ currentTime }}
        </div>
      </div>
    </header>

    <!-- Main Content Area -->
    <main class="flex-1 overflow-hidden p-6 flex gap-6">
      <!-- 1. Left Column: Camera Feed & YOLO Overlay -->
      <div class="w-1/3 flex flex-col gap-6 h-full min-w-[360px]">
        <div class="industrial-panel flex-1 p-5 flex flex-col relative min-h-[400px]">
          <div class="flex justify-between items-center mb-3 shrink-0">
            <h3 class="text-sm font-semibold tracking-wide text-slate-300 uppercase flex items-center gap-2">
              <span class="w-1.5 h-3 bg-blue-500 rounded-sm"></span>
              工业相机实时扫描流
            </h3>
            <span class="text-xs text-slate-500 font-mono">FRAME_HEIGHT: 4096px</span>
          </div>

          <!-- Video Stream Box -->
          <div class="relative flex-1 bg-slate-900 border border-slate-800 rounded-lg overflow-hidden flex items-center justify-center">
            <!-- Simulated Camera Grid Background -->
            <div class="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_50%,rgba(30,41,59,0.5),rgba(9,13,22,0.9))]"></div>
            
            <!-- Laser scanning beam -->
            <div class="laser-scanner z-20"></div>

            <!-- YOLO Camera View canvas -->
            <div class="relative w-full h-full max-h-full flex items-center justify-center p-3">
              <img 
                v-if="store.selectedRecord" 
                ref="defectImg"
                src="https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=640&q=80" 
                class="max-w-full max-h-full rounded border border-slate-700 object-contain shadow-2xl"
                @load="drawBoundingBoxes"
              />
              <div v-else class="text-slate-500 text-xs flex flex-col items-center gap-2">
                <svg viewBox="0 0 24 24" class="w-10 h-10 stroke-slate-600 fill-none stroke-[1.5] animate-pulse">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M6.827 6.175A2.31 2.31 0 0110 6h4a2.305 2.305 0 012.227 1.83l.89 3.668A2.298 2.298 0 0019.35 13.5H20.5c.983 0 1.879.37 2.5 1.025v3.475a2.5 2.5 0 01-2.5 2.5H3.5a2.5 2.5 0 01-2.5-2.5v-3.475C1.621 13.87 2.517 13.5 3.5 13.5h1.15a2.298 2.298 0 002.228-2.027l.89-3.668z" />
                  <path stroke-linecap="round" stroke-linejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span>等待带钢缺陷触发...</span>
              </div>
              
              <!-- Draw canvas bounding box overlays -->
              <canvas ref="boxCanvas" class="absolute inset-0 pointer-events-none z-10 w-full h-full object-contain"></canvas>
            </div>

            <!-- YOLO Details Tag on screen -->
            <div v-if="store.selectedRecord && store.selectedRecord.defect_count > 0" class="absolute bottom-4 left-4 bg-rose-950/90 border border-rose-500/50 px-3 py-1.5 rounded text-[10px] z-20 font-mono shadow-lg">
              <div class="text-rose-400 font-bold">检测到表面缺陷: {{ store.selectedRecord.defect_types }}</div>
              <div class="text-slate-300 mt-0.5">置信度: {{ (store.selectedRecord.confidence * 100).toFixed(0) }}% | 时延: {{ store.selectedRecord.yolo_result?.inference_time_ms?.toFixed(2) }} ms</div>
            </div>
          </div>
        </div>
      </div>

      <!-- 2. Middle Column: Conveyor Belt & Gemini Consultation -->
      <div class="w-5/12 flex flex-col gap-6 h-full">
        <!-- Conveyor Belt -->
        <ConveyorBelt class="shrink-0" />

        <!-- Gemini Consultation & RAG Standard -->
        <div class="industrial-panel flex-1 p-5 flex flex-col overflow-hidden min-h-[300px]">
          <div class="flex justify-between items-center mb-3 shrink-0">
            <h3 class="text-sm font-semibold tracking-wide text-slate-300 uppercase flex items-center gap-2">
              <span class="w-1.5 h-3 bg-orange-500 rounded-sm"></span>
              Gemini VLM & GB/T 标准联合专家系统
            </h3>
            <button 
              v-if="store.selectedRecord && store.selectedRecord.defect_count > 0 && !store.selectedRecord.vlm_result?.analysis"
              @click="store.requestConsultation(store.selectedRecord.id)"
              :disabled="store.consulting"
              class="px-3 py-1 bg-gradient-to-r from-orange-600 to-amber-500 hover:from-orange-500 hover:to-amber-400 text-slate-100 font-medium text-xs rounded transition-all flex items-center gap-1.5 shadow shadow-orange-500/20 disabled:opacity-50"
            >
              <svg viewBox="0 0 24 24" class="w-3.5 h-3.5 fill-none stroke-current stroke-[2]" :class="store.consulting ? 'animate-spin' : ''">
                <path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904L9 21l3.59-1.353 3.59 1.353-.813-5.096A9.037 9.037 0 0119.5 12c0-5.247-4.253-9.5-9.5-9.5S.5 6.753.5 12c0 1.543.388 2.996 1.072 4.272L.5 21l3.59-1.353 3.59 1.353-.813-5.096A9.037 9.037 0 019.813 15.904z" />
              </svg>
              {{ store.consulting ? '专家诊断中...' : '发起会诊' }}
            </button>
          </div>

          <!-- RAG Results Container -->
          <div class="flex-1 overflow-y-auto flex flex-col gap-4 pr-1">
            <div v-if="!store.selectedRecord" class="flex-1 flex items-center justify-center text-xs text-slate-500">
              请点击虚拟传送带或右侧历史列表的“异常钢板”启动 RAG 大模型会诊面板。
            </div>
            
            <div v-else-if="store.selectedRecord.defect_count === 0" class="flex-1 flex items-center justify-center text-xs text-green-500/80 bg-green-950/20 border border-green-900/30 rounded-lg p-6">
              ✅ 钢卷 ID #{{ store.selectedRecord.id }} 表面无可见缺陷。已自动判定为合格。无需发起大模型诊断。
            </div>

            <div v-else class="flex flex-col gap-4">
              <!-- Gemini Diagnosis Block -->
              <div class="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4 relative overflow-hidden">
                <div class="flex justify-between items-center mb-2">
                  <span class="text-xs font-semibold text-orange-400 flex items-center gap-1.5">
                    💬 AI 视觉模型(VLM) 深度成因会诊报告
                  </span>
                  <span v-if="store.selectedRecord.vlm_result?.confidence" class="text-[10px] text-slate-500 font-mono">
                    会诊置信度: {{ (store.selectedRecord.vlm_result.confidence * 100).toFixed(0) }}%
                  </span>
                </div>
                <div v-if="store.consulting" class="py-6 flex flex-col items-center gap-2 text-slate-400 text-xs">
                  <div class="w-8 h-8 border-2 border-orange-500/20 border-t-orange-500 rounded-full animate-spin"></div>
                  <span>正在调取本地 SQLite3 向量数据库并注入 Gemini 大模型...</span>
                </div>
                <div v-else-if="!store.selectedRecord.vlm_result?.analysis" class="py-6 text-center text-xs text-slate-500">
                  尚未发起会诊。点击右上角“发起会诊”对异常缺陷成因做多维度分析。
                </div>
                <div v-else class="text-xs leading-relaxed text-slate-300 font-sans whitespace-pre-line">
                  {{ store.selectedRecord.vlm_result.analysis }}
                </div>
              </div>

              <!-- GB/T RAG Knowledge Block -->
              <div v-if="store.selectedRecord.rag_standard" class="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4">
                <div class="text-xs font-semibold text-blue-400 flex items-center gap-1.5 mb-2">
                  📘 国家标准（GB/T）质检规范依据
                </div>
                <div class="flex justify-between items-center bg-slate-950 px-3 py-1.5 rounded border border-slate-800/60 mb-2">
                  <span class="text-[11px] font-bold text-slate-300 font-mono">{{ store.selectedRecord.rag_standard.standard_code }}</span>
                  <span class="text-[10px] text-slate-500">{{ store.selectedRecord.rag_standard.title }}</span>
                </div>
                <p class="text-xs text-slate-400 leading-relaxed bg-slate-950/40 p-2.5 rounded italic">
                  {{ store.selectedRecord.rag_standard.content }}
                </p>
              </div>

              <!-- Human Audit Input -->
              <div class="bg-slate-900/80 border border-slate-800/80 rounded-lg p-4">
                <div class="text-xs font-semibold text-slate-300 mb-2.5">
                  ✍️ 人工审核判定 (Human-in-the-Loop)
                </div>
                <div class="flex gap-4 mb-3">
                  <div class="flex-1">
                    <label class="text-[10px] text-slate-500 block mb-1">审核员姓名</label>
                    <input 
                      v-model="auditForm.reviewer" 
                      type="text" 
                      placeholder="质检员甲"
                      class="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-blue-500/50"
                    />
                  </div>
                  <div class="flex-1.5">
                    <label class="text-[10px] text-slate-500 block mb-1">判批意见</label>
                    <input 
                      v-model="auditForm.note" 
                      type="text" 
                      placeholder="输入核实备注..."
                      class="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs focus:outline-none focus:border-blue-500/50"
                    />
                  </div>
                </div>
                <div class="flex gap-3">
                  <button 
                    @click="submitAudit('confirmed')"
                    class="flex-1 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-slate-100 font-bold text-xs rounded transition-all shadow shadow-emerald-700/20"
                  >
                    确认合格 (PASS)
                  </button>
                  <button 
                    @click="submitAudit('corrected')"
                    class="flex-1 py-1.5 bg-rose-700 hover:bg-rose-600 text-slate-100 font-bold text-xs rounded transition-all shadow shadow-rose-700/20"
                  >
                    确认不合格 (REJECT)
                  </button>
                </div>
                <!-- Audit State Footer -->
                <div v-if="store.selectedRecord.review_status !== 'pending'" class="mt-3 flex justify-between items-center text-[10px] text-slate-500 border-t border-slate-800/80 pt-2 font-mono">
                  <span>审核状态: <strong class="text-green-400">{{ store.selectedRecord.review_status === 'confirmed' ? '已确认合格' : '确认排除缺陷' }}</strong></span>
                  <span>审核员: {{ store.selectedRecord.reviewer }} | {{ new Date(store.selectedRecord.review_time).toLocaleTimeString() }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 3. Right Column: Defect Logs & Charts -->
      <div class="w-1/3 flex flex-col gap-6 h-full min-w-[340px]">
        <!-- 3.1 Defect Logs List -->
        <div class="industrial-panel h-[220px] p-4 flex flex-col overflow-hidden">
          <h3 class="text-xs font-semibold tracking-wider text-slate-400 uppercase mb-2 shrink-0 flex items-center gap-1.5">
            📋 生产质检记录表 (100% 实时流水)
          </h3>
          <div class="flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
            <div 
              v-for="record in store.records" 
              :key="record.id"
              class="bg-slate-900/60 border rounded px-3 py-2 flex items-center justify-between cursor-pointer transition-all hover:bg-slate-900"
              :class="store.selectedRecord?.id === record.id ? 'border-blue-500/70 bg-slate-900 shadow-glow-blue' : 'border-slate-800/80'"
              @click="store.selectedRecord = record"
            >
              <div class="flex items-center gap-2">
                <span 
                  class="w-2 h-2 rounded-full" 
                  :class="record.defect_count > 0 ? 'bg-rose-500 shadow-[0_0_5px_rgba(239,68,68,0.7)]' : 'bg-green-500'"
                ></span>
                <span class="text-xs font-mono font-bold text-slate-300">#{{ record.id }}</span>
              </div>
              <span class="text-[10px] text-slate-400 truncate max-w-[120px]">
                {{ record.defect_count > 0 ? `🚨 缺陷: ${record.defect_types}` : '✅ 正常合格' }}
              </span>
              <span class="text-[9px] text-slate-500 font-mono">{{ new Date(record.timestamp).toLocaleTimeString() }}</span>
            </div>
          </div>
        </div>

        <!-- 3.2 ECharts Graphs -->
        <div class="industrial-panel flex-1 p-4 flex flex-col overflow-hidden min-h-[180px]">
          <h3 class="text-xs font-semibold tracking-wider text-slate-400 uppercase mb-2 shrink-0">
            📊 缺陷品类占比与良品率看板
          </h3>
          <div class="flex-1 relative flex flex-col gap-2">
            <div ref="pieChart" class="flex-1 min-h-[110px] w-full"></div>
            <div ref="lineChart" class="flex-1 min-h-[110px] w-full"></div>
          </div>
        </div>

        <!-- 3.3 Hardware Metrics -->
        <div class="industrial-panel p-4 flex flex-col shrink-0 gap-3">
          <h3 class="text-xs font-semibold tracking-wider text-slate-400 uppercase">
            ⚡️ 工控系统健康指数 (SYSTEM HEALTH)
          </h3>
          <div class="grid grid-cols-3 gap-3">
            <!-- CPU card -->
            <div class="bg-slate-900/60 border border-slate-800/80 rounded p-2.5 flex flex-col">
              <span class="text-[9px] text-slate-500">CPU 使用率</span>
              <span class="text-sm font-bold font-mono text-slate-300 mt-1">{{ store.systemMetrics.cpu_usage.toFixed(0) }}%</span>
              <div class="text-[9px] text-slate-400 font-mono mt-0.5">🔥 {{ store.systemMetrics.cpu_temp.toFixed(0) }}℃</div>
            </div>
            <!-- GPU card -->
            <div class="bg-slate-900/60 border border-slate-800/80 rounded p-2.5 flex flex-col">
              <span class="text-[9px] text-slate-500">GPU 使用率</span>
              <span class="text-sm font-bold font-mono text-slate-300 mt-1">{{ store.systemMetrics.gpu_usage.toFixed(0) }}%</span>
              <div class="text-[9px] text-slate-400 font-mono mt-0.5">🔥 {{ store.systemMetrics.gpu_temp.toFixed(0) }}℃</div>
            </div>
            <!-- Latency card -->
            <div class="bg-slate-900/60 border border-slate-800/80 rounded p-2.5 flex flex-col">
              <span class="text-[9px] text-slate-500">端到端推理时延</span>
              <span class="text-sm font-bold font-mono text-blue-400 mt-1">{{ store.systemMetrics.inference_delay.toFixed(2) }}ms</span>
              <div class="text-[9px] text-slate-400 font-mono mt-0.5">⚡️ 500+ FPS</div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, onUnmounted, watch } from 'vue';
import { useDefectStore } from '../store/defect';
import ConveyorBelt from '../components/ConveyorBelt.vue';
import * as echarts from 'echarts';

const store = useDefectStore();
const currentTime = ref('');

const auditForm = reactive({
  reviewer: '',
  note: ''
});

// Update time tick
let timer: any = null;
const updateTime = () => {
  currentTime.value = new Date().toLocaleString();
};

// Canvas and image overlay logic
const boxCanvas = ref<HTMLCanvasElement | null>(null);
const defectImg = ref<HTMLImageElement | null>(null);

const drawBoundingBoxes = () => {
  const canvas = boxCanvas.value;
  const img = defectImg.value;
  if (!canvas || !img || !store.selectedRecord) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Set canvas dynamic resolution matching image source
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const defects = store.selectedRecord.final_result?.defects || [];
  
  // Real coordinates of YOLO are on standard 640x640 frame
  // Scale them to fits responsive img dimensions
  defects.forEach(defect => {
    const [x, y, w, h] = defect.box;
    
    // Scaling calculations
    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 640;
    
    const scaledX = x * scaleX;
    const scaledY = y * scaleY;
    const scaledW = w * scaleX;
    const scaledH = h * scaleY;

    // Draw box
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);

    // Label tag
    ctx.fillStyle = '#ef4444';
    ctx.font = '10px Outfit, Inter, sans-serif';
    const text = `${defect.cn} (${(defect.confidence * 100).toFixed(0)}%)`;
    const textWidth = ctx.measureText(text).width;
    
    ctx.fillRect(scaledX - 1, scaledY - 14, textWidth + 8, 14);
    
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, scaledX + 3, scaledY - 4);
  });
};

watch(() => store.selectedRecord, () => {
  // Redraw if selection changes
  setTimeout(drawBoundingBoxes, 150);
});

// Human audit submission
const submitAudit = (status: 'confirmed' | 'corrected') => {
  if (!store.selectedRecord) return;
  const reviewer = auditForm.reviewer.trim() || '操作员A';
  const note = auditForm.note.trim() || '无';
  
  store.auditRecord(store.selectedRecord.id, status, reviewer, note);
  
  auditForm.note = '';
};

// ECharts instantiation
const pieChart = ref<HTMLDivElement | null>(null);
const lineChart = ref<HTMLDivElement | null>(null);
let pieChartInstance: echarts.ECharts | null = null;
let lineChartInstance: echarts.ECharts | null = null;

const initCharts = () => {
  if (pieChart.value) {
    pieChartInstance = echarts.init(pieChart.value, 'dark');
    const option = {
      backgroundColor: 'transparent',
      title: {
        text: '缺陷分类占比统计',
        left: 'center',
        textStyle: { fontSize: 11, color: '#94a3b8' }
      },
      tooltip: { trigger: 'item', formatter: '{b} : {c} ({d}%)' },
      series: [
        {
          name: '缺陷类型',
          type: 'pie',
          radius: ['40%', '70%'],
          center: ['50%', '55%'],
          avoidLabelOverlap: false,
          itemStyle: { borderRadius: 4, borderColor: '#090d16', borderWidth: 2 },
          label: { show: false, position: 'center' },
          labelLine: { show: false },
          data: [
            { value: 12, name: '裂纹' },
            { value: 8, name: '划痕' },
            { value: 15, name: '氧化皮' },
            { value: 6, name: '压痕' },
            { value: 4, name: '气泡' }
          ]
        }
      ]
    };
    pieChartInstance.setOption(option);
  }

  if (lineChart.value) {
    lineChartInstance = echarts.init(lineChart.value, 'dark');
    const option = {
      backgroundColor: 'transparent',
      title: {
        text: '良品率趋势走势',
        left: 'center',
        textStyle: { fontSize: 11, color: '#94a3b8' }
      },
      grid: { top: 25, bottom: 20, left: 30, right: 10 },
      xAxis: {
        type: 'category',
        data: ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00'],
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { fontSize: 9 }
      },
      yAxis: {
        type: 'value',
        min: 90,
        max: 100,
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: { lineStyle: { color: '#1e293b' } },
        axisLabel: { fontSize: 9 }
      },
      series: [
        {
          data: [98.2, 97.5, 96.8, 98.4, 97.9, 99.1],
          type: 'line',
          smooth: true,
          lineStyle: { color: '#10b981', width: 2 },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(16,185,129,0.3)' },
              { offset: 1, color: 'rgba(16,185,129,0)' }
            ])
          }
        }
      ]
    };
    lineChartInstance.setOption(option);
  }
};

// Redraw chart stats when new defect records arrive
watch(() => store.records, () => {
  if (!pieChartInstance || !lineChartInstance) return;
  
  // Aggregate real defects statistics from Pinia records
  const counts: Record<string, number> = { '裂纹': 0, '划痕': 0, '氧化皮': 0, '压痕': 0, '气泡': 0 };
  let totalDefects = 0;
  
  store.records.forEach(r => {
    if (r.defect_count > 0 && r.defect_types) {
      const type = r.defect_types;
      if (counts[type] !== undefined) {
        counts[type]++;
        totalDefects++;
      }
    }
  });

  // Update Pie Chart Data
  const updatedPieData = Object.keys(counts).map(key => ({
    value: counts[key] + (Math.floor(Math.random() * 2)), // Keep mockup values slightly sliding
    name: key
  }));

  pieChartInstance.setOption({
    series: [{ data: updatedPieData }]
  });
}, { deep: true });

const handleResize = () => {
  pieChartInstance?.resize();
  lineChartInstance?.resize();
  drawBoundingBoxes();
};

onMounted(() => {
  updateTime();
  timer = setInterval(updateTime, 1000);
  
  // Init WebSocket + Mockup falls
  store.connectWebSocket();
  store.mockServerDetectionStream();
  
  setTimeout(() => {
    initCharts();
    window.addEventListener('resize', handleResize);
  }, 100);
});

onUnmounted(() => {
  clearInterval(timer);
  window.removeEventListener('resize', handleResize);
  pieChartInstance?.dispose();
  lineChartInstance?.dispose();
  if (store.ws) {
    store.ws.close();
  }
});
</script>

<style scoped>
.flex-1\.5 {
  flex: 1.5 1.5 0%;
}
.laser-scanner {
  animation: laser-scan 4s infinite linear;
}
@keyframes laser-scan {
  0% { top: 0%; opacity: 0.8; }
  50% { opacity: 0.2; }
  100% { top: 100%; opacity: 0.8; }
}
</style>

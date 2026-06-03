import { defineStore } from 'pinia';

export interface DefectItem {
  type: string;
  cn: string;
  confidence: number;
  box: [number, number, number, number]; // [x, y, w, h]
}

export interface InspectionRecord {
  id: number;
  timestamp: string;
  image_path: string;
  result_path: string;
  yolo_result: {
    defects?: DefectItem[];
    inference_time_ms?: number;
  };
  vlm_result: {
    status?: string;
    analysis?: string;
    confidence?: number;
  };
  final_result: {
    status?: 'pass' | 'defect' | 'review';
    defects?: DefectItem[];
  };
  defect_types: string;
  defect_count: number;
  confidence: number;
  reviewer: string;
  review_status: 'pending' | 'confirmed' | 'corrected';
  review_time: string;
  note: string;
  // RAG standard recommendation
  rag_standard?: {
    standard_code: string;
    title: string;
    content: string;
  };
}

export interface ConveyorItem {
  id: string;
  status: 'pass' | 'defect' | 'processing';
  time: string;
  details: string;
  position: number; // 0 to 100 representing percentage along conveyor
  recordId?: number;
}

export interface SystemMetrics {
  cpu_usage: number;
  cpu_temp: number;
  gpu_usage: number;
  gpu_temp: number;
  memory_usage: number;
  disk_usage: number;
  inference_delay: number;
  camera_fps: number;
  defect_rate: number;
}

export const useDefectStore = defineStore('defect', {
  state: () => ({
    connected: false,
    ws: null as WebSocket | null,
    systemMetrics: {
      cpu_usage: 12,
      cpu_temp: 45,
      gpu_usage: 28,
      gpu_temp: 52,
      memory_usage: 34,
      disk_usage: 48,
      inference_delay: 2.1,
      camera_fps: 30,
      defect_rate: 4.8
    } as SystemMetrics,
    records: [] as InspectionRecord[],
    conveyorItems: [] as ConveyorItem[],
    selectedRecord: null as InspectionRecord | null,
    consulting: false,
    consultationResult: null as string | null,
  }),

  actions: {
    connectWebSocket() {
      if (this.ws) {
        this.ws.close();
      }

      // Default Drogon Server WebSocket Address
      const wsUrl = `ws://${window.location.hostname}:8080/camera/stream`;
      console.log(`[WebSocket] Connecting to ${wsUrl}...`);
      
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('[WebSocket] Connected successfully!');
        this.connected = true;
      };

      this.ws.onclose = () => {
        console.log('[WebSocket] Disconnected! Reconnecting in 3 seconds...');
        this.connected = false;
        setTimeout(() => this.connectWebSocket(), 3000);
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'metrics') {
            this.systemMetrics = { ...this.systemMetrics, ...data.data };
          } 
          else if (data.type === 'detection') {
            const record = data.data as InspectionRecord;
            this.addInspectionRecord(record);
          }
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };
    },

    async fetchHistory() {
      try {
        const response = await fetch(`http://${window.location.hostname}:8080/api/records?limit=50`);
        const data = await response.json();
        if (Array.isArray(data)) {
          this.records = data.map((record: any) => {
            return {
              ...record,
              review_status: record.review_status || 'pending',
              yolo_result: typeof record.yolo_result === 'string' ? JSON.parse(record.yolo_result) : (record.yolo_result || {}),
              vlm_result: typeof record.vlm_result === 'string' ? JSON.parse(record.vlm_result) : (record.vlm_result || {}),
              final_result: typeof record.final_result === 'string' ? JSON.parse(record.final_result) : (record.final_result || {}),
            };
          });
          
          // Auto select the first record if any and none is currently selected
          if (this.records.length > 0 && !this.selectedRecord) {
            this.selectedRecord = this.records[0];
          }
          console.log(`[Store] Loaded ${this.records.length} history records from database.`);
        }
      } catch (err) {
        console.error('[Store] Failed to fetch history records:', err);
      }
    },

    addInspectionRecord(record: InspectionRecord) {
      // Robustly parse JSON properties if they come in as stringified JSON from database updates
      const formattedRecord: InspectionRecord = {
        ...record,
        review_status: record.review_status || 'pending',
        yolo_result: typeof record.yolo_result === 'string' ? JSON.parse(record.yolo_result) : (record.yolo_result || {}),
        vlm_result: typeof record.vlm_result === 'string' ? JSON.parse(record.vlm_result) : (record.vlm_result || {}),
        final_result: typeof record.final_result === 'string' ? JSON.parse(record.final_result) : (record.final_result || {}),
      };

      // Check if record already exists
      const existingIndex = this.records.findIndex(r => r.id === formattedRecord.id);
      if (existingIndex !== -1) {
        // Sync and update record in place
        this.records[existingIndex] = { ...this.records[existingIndex], ...formattedRecord };
        
        // Update selected record in place if it matches
        if (this.selectedRecord && this.selectedRecord.id === formattedRecord.id) {
          this.selectedRecord = { ...this.records[existingIndex] };
        }
        console.log(`[Store] Updated existing record #${formattedRecord.id} in-place.`);
        return;
      }

      // Add to history records list
      this.records.unshift(formattedRecord);
      if (this.records.length > 50) {
        this.records.pop();
      }

      // Add to rolling conveyor belt
      const conveyorId = `sheet-${Date.now()}`;
      const status = formattedRecord.defect_count > 0 ? 'defect' : 'pass';
      const details = formattedRecord.defect_count > 0 
        ? `⚠️ 检出缺陷: ${formattedRecord.defect_types}`
        : '✅ 钢板表面合格';

      this.conveyorItems.push({
        id: conveyorId,
        status,
        time: new Date(formattedRecord.timestamp || Date.now()).toLocaleTimeString(),
        details,
        position: 0,
        recordId: formattedRecord.id
      });

      // Maintain conveyor items list length
      if (this.conveyorItems.length > 15) {
        this.conveyorItems.shift();
      }

      // Automatically select if it is a defect to alert the operator
      if (status === 'defect') {
        this.selectedRecord = formattedRecord;
      }
    },

    // Trigger VLM & RAG Consultation via HTTP POST to C++ backend
    async requestConsultation(recordId: number) {
      this.consulting = true;
      this.consultationResult = null;
      try {
        const response = await fetch(`http://${window.location.hostname}:8080/api/consult?id=${recordId}`, {
          method: 'POST'
        });
        const data = await response.json();
        
        // Update the record in our local list
        const recordIndex = this.records.findIndex(r => r.id === recordId);
        if (recordIndex !== -1) {
          this.records[recordIndex].vlm_result = data.vlm_result;
          this.records[recordIndex].rag_standard = data.rag_standard;
          if (this.selectedRecord && this.selectedRecord.id === recordId) {
            this.selectedRecord = this.records[recordIndex];
          }
        }
        
        this.consultationResult = data.vlm_result?.analysis || '无会诊结论';
      } catch (err) {
        console.error('[Store] Consultation API failed:', err);
        // Fallback mockup in case backend isn't ready
        setTimeout(() => {
          this.mockConsultation(recordId);
        }, 1500);
      } finally {
        this.consulting = false;
      }
    },

    mockConsultation(recordId: number) {
      const record = this.records.find(r => r.id === recordId);
      if (!record) return;

      record.vlm_result = {
        status: 'completed',
        confidence: 0.94,
        analysis: '【大模型诊断结论】\n经过视觉大模型（VLM）与国家钢铁表面质量规范（GB/T 3280-2015）协同判定，该位置存在明显的机械划擦伤（Scratch）。缺陷沿带钢轧制方向呈纵向连续分布，触感深度约 0.06mm。符合 GB/T 3280-2015 中对 BA 板面“不允许存在宏观划伤”的限制，故该钢卷部分区域应当判定为【不合格】。\n\n【工艺改进建议】\n1. 检查精整机组夹送辊和擦拭器，定时吹扫，清除表层硬屑点。\n2. 在卷取两端加装精密对中系统，消除卷取错边塔形拉伤。'
      };

      record.rag_standard = {
        standard_code: 'GB/T 3280-2015',
        title: '高精度不锈钢板机械擦伤与滑痕允许公差标准',
        content: '根据 GB/T 3280 规定，精密冷轧面板（BA/No.4）绝对不允许存在宏观划痕。普通用途板（2B板）表面局部浅表擦伤，深度小于材料负公差之半（通常≤0.05mm）可降级使用，但不得呈贯穿性分布。'
      };

      if (this.selectedRecord && this.selectedRecord.id === recordId) {
        this.selectedRecord = { ...record };
      }
    },

    // Audit control: Pass, Reject, Correct
    async auditRecord(recordId: number, status: 'confirmed' | 'corrected', reviewer: string, note: string) {
      try {
        const response = await fetch(`http://${window.location.hostname}:8080/api/audit`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id: recordId, review_status: status, reviewer, note })
        });
        const data = await response.json();
        
        const recordIndex = this.records.findIndex(r => r.id === recordId);
        if (recordIndex !== -1 && data.success) {
          this.records[recordIndex].review_status = status;
          this.records[recordIndex].reviewer = reviewer;
          this.records[recordIndex].note = note;
          this.records[recordIndex].review_time = new Date().toISOString();
          
          if (this.selectedRecord && this.selectedRecord.id === recordId) {
            this.selectedRecord = { ...this.records[recordIndex] };
          }
        }
      } catch (err) {
        console.error('[Store] Audit API failed, applying local mock:', err);
        // Fallback local update
        const recordIndex = this.records.findIndex(r => r.id === recordId);
        if (recordIndex !== -1) {
          this.records[recordIndex].review_status = status;
          this.records[recordIndex].reviewer = reviewer;
          this.records[recordIndex].note = note;
          this.records[recordIndex].review_time = new Date().toISOString();
          
          if (this.selectedRecord && this.selectedRecord.id === recordId) {
            this.selectedRecord = { ...this.records[recordIndex] };
          }
        }
      }
    },

    mockServerDetectionStream() {
      // Mock detection stream in frontend for local standalone presentation
      setInterval(() => {
        if (this.connected) return; // Skip if live WebSocket is working
        
        const defectChance = Math.random() < 0.15; // 15% defect rate
        const mockTypes = ['crack', 'scratch', 'scale', 'indentation', 'blister'];
        const mockCns = ['裂纹', '划痕', '氧化皮', '压痕', '气泡'];
        const mockColors = ['#E53E3E', '#3182CE', '#D69E2E', '#805AD5', '#319795'];
        
        let record: InspectionRecord;
        const id = Math.floor(Math.random() * 89999) + 10000;
        
        if (defectChance) {
          const typeIndex = Math.floor(Math.random() * mockTypes.length);
          const type = mockTypes[typeIndex];
          const cn = mockCns[typeIndex];
          const confidence = 0.75 + Math.random() * 0.22;
          
          const x = 100 + Math.random() * 300;
          const y = 80 + Math.random() * 200;
          const w = 50 + Math.random() * 100;
          const h = 40 + Math.random() * 80;

          record = {
            id,
            timestamp: new Date().toISOString(),
            image_path: `/data/images/sheet_${id}.jpg`,
            result_path: '',
            yolo_result: {
              defects: [{ type, cn, confidence, box: [x, y, w, h] }],
              inference_time_ms: 1.8 + Math.random() * 0.8
            },
            vlm_result: {},
            final_result: {
              status: 'defect',
              defects: [{ type, cn, confidence, box: [x, y, w, h] }]
            },
            defect_types: cn,
            defect_count: 1,
            confidence: parseFloat(confidence.toFixed(2)),
            reviewer: '',
            review_status: 'pending',
            review_time: '',
            note: ''
          };
        } else {
          record = {
            id,
            timestamp: new Date().toISOString(),
            image_path: `/data/images/sheet_${id}.jpg`,
            result_path: '',
            yolo_result: {
              defects: [],
              inference_time_ms: 1.2 + Math.random() * 0.4
            },
            vlm_result: {},
            final_result: { status: 'pass' },
            defect_types: '',
            defect_count: 0,
            confidence: 0.99,
            reviewer: '',
            review_status: 'confirmed',
            review_time: '',
            note: ''
          };
        }
        
        this.addInspectionRecord(record);
      }, 4000);

      // System metrics fluctuation
      setInterval(() => {
        if (this.connected) return;
        this.systemMetrics.cpu_usage = Math.max(8, Math.min(45, this.systemMetrics.cpu_usage + (Math.random() - 0.5) * 4));
        this.systemMetrics.gpu_usage = Math.max(15, Math.min(85, this.systemMetrics.gpu_usage + (Math.random() - 0.5) * 6));
        this.systemMetrics.inference_delay = Math.max(1.1, Math.min(3.5, this.systemMetrics.inference_delay + (Math.random() - 0.5) * 0.3));
      }, 2000);
    }
  }
});

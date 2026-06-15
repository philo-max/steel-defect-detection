/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Sliders,
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Sparkles,
  RefreshCw,
  Play,
  Flame,
  Eye,
  Camera,
  Upload,
  Trash2,
  Info,
  ChevronRight,
  Database,
  History,
  HardDrive,
  ZoomIn,
  Layers,
  Settings,
  X,
  FileSpreadsheet,
  Check,
  AlertCircle,
  Maximize,
  Minimize,
  Lock,
  LogOut,
  MessageSquare,
  Send
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { drawSteelPlate, SAMPLE_PRESETS } from './data/samples';
import { DefectItem, DefectSample, DetectionResult, InspectionRecord } from './types';

export default function App() {
  // User Authentication state
  const [session, setSession] = useState<{ username: string; role: string; token: string } | null>(() => {
    const cached = localStorage.getItem('steel_user_session');
    if (cached) {
      try {
        return JSON.parse(cached);
      } catch (err) {
        return null;
      }
    }
    return null;
  });

  const [usernameInput, setUsernameInput] = useState<string>('');
  const [passwordInput, setPasswordInput] = useState<string>('');
  const [loginError, setLoginError] = useState<string>('');
  const [isLoggingIn, setIsLoggingIn] = useState<boolean>(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!usernameInput || !passwordInput) {
      setLoginError('请填写所有字段');
      return;
    }
    setIsLoggingIn(true);
    setLoginError('');
    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: usernameInput, password: passwordInput })
      });
      const data = await res.json();
      if (data.success) {
        const userSession = {
          username: data.username,
          role: data.role,
          token: data.token
        };
        setSession(userSession);
        localStorage.setItem('steel_user_session', JSON.stringify(userSession));
        setUsernameInput('');
        setPasswordInput('');
      } else {
        setLoginError(data.error || '登录失败，请检查账号密码');
      }
    } catch (err: any) {
      setLoginError(`无法连接登录服务器: ${err.message}`);
    } finally {
      setIsLoggingIn(false);
    }
  };

  const handleLogout = async () => {
    try {
      await fetch('/api/logout', { method: 'POST' });
    } catch (e) {}
    setSession(null);
    localStorage.removeItem('steel_user_session');
    setActiveResult(null);
  };

  const getRoleChinese = (role: string) => {
    switch (role) {
      case 'admin': return '系统管理员';
      case 'inspector': return '现场质检员';
      case 'supervisor': return '质检主管';
      case 'ai_engineer': return 'AI 工程师';
      case 'process_engineer': return '工艺工程师';
      default: return role;
    }
  };

  // Application UI states
  const [samples] = useState<DefectSample[]>(SAMPLE_PRESETS);
  const [selectedSample, setSelectedSample] = useState<DefectSample>(SAMPLE_PRESETS[1]); // scratch by default
  const [customImageBase64, setCustomImageBase64] = useState<string | null>(null);
  const [customFilename, setCustomFilename] = useState<string>('');
  
  // Viewer and Scan State
  const [viewMode, setViewMode] = useState<'standard' | 'infrared' | 'edge' | 'contrast'>('standard');
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [contrastThreshold, setContrastThreshold] = useState<number>(50);
  const [laserSpeed, setLaserSpeed] = useState<number>(3); // 1-5
  const [zoomScale, setZoomScale] = useState<number>(1);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [hoveredDefectId, setHoveredDefectId] = useState<string | null>(null);
  const [selectedDefectId, setSelectedDefectId] = useState<string | null>(null);
  
  // Historical Records & Active Detection results
  const [history, setHistory] = useState<InspectionRecord[]>([]);
  const [activeResult, setActiveResult] = useState<DetectionResult | null>(null);
  const [showHistoryModal, setShowHistoryModal] = useState<boolean>(false);
  const [isSystemHealthy, setIsSystemHealthy] = useState<boolean>(true);
  const [aiEngineStatus, setAiEngineStatus] = useState<string>('Init...');

  // Manual auditing modal / states (human in the loop override)
  const [auditingDefect, setAuditingDefect] = useState<DefectItem | null>(null);

  // Role based access permissions
  const userRole = session?.role || '';
  const isAdmin = userRole === 'admin';
  const isInspector = userRole === 'inspector';
  const isSupervisor = userRole === 'supervisor';
  const isAiEngineer = userRole === 'ai_engineer';
  const isProcessEngineer = userRole === 'process_engineer';

  const canAdjustSliders = isAdmin || isAiEngineer;
  const canSelectPresets = isAdmin || isInspector;
  const canScan = isAdmin || isInspector;
  const canAudit = isAdmin || isInspector;
  const canViewKPI = isAdmin || isSupervisor;
  const canViewRAG = isAdmin || isInspector || isProcessEngineer;

  const [historyModalTab, setHistoryModalTab] = useState<'list' | 'stats'>('list');
  const [auditClass, setAuditClass] = useState<string>('');
  const [auditSeverity, setAuditSeverity] = useState<'Low' | 'Medium' | 'High'>('Medium');
  const [auditComment, setAuditComment] = useState<string>('');

  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [kpiData, setKpiData] = useState<{
    leakageRate: number;
    overkillRate: number;
    avgDelayMs: number;
    totalInspections: number;
    yoloCount: number;
    vlmCount: number;
    hardware: {
      device: string;
      yoloModel: string;
      vlmProvider: string;
      dbType: string;
    }
  } | null>(null);

  const [filterReviewStatus, setFilterReviewStatus] = useState<string>('all');
  const [filterDefectType, setFilterDefectType] = useState<string>('all');

  // Training and Data Flywheel State
  const [trainStatus, setTrainStatus] = useState<{
    status: string;
    currentEpoch: number;
    totalEpochs: number;
    progress: number;
    logPreview: string;
    correctedCount: number;
  } | null>(null);

  const fetchTrainStatus = async () => {
    try {
      const res = await fetch('/api/train/status');
      const data = await res.json();
      if (data.success) {
        setTrainStatus(data);
      }
    } catch (err) {
      console.error('Failed fetching training status:', err);
    }
  };

  const handleStartTraining = async () => {
    if (!confirm('确认要导出已审核修正的数据并启动后台 YOLO 增量训练吗？\n该操作会合并 Bad Cases 进行自动迭代。')) {
      return;
    }
    try {
      const res = await fetch('/api/train/start', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        alert(data.message);
        fetchTrainStatus();
      } else {
        alert(data.error);
      }
    } catch (err: any) {
      alert(`启动训练失败: ${err.message}`);
    }
  };

  // RAG Chatbot Assistant State
  const [chatMessages, setChatMessages] = useState<Array<{ sender: 'user' | 'assistant'; text: string }>>([
    { sender: 'assistant', text: '您好！我是您的 SteelEye 智能冶金工艺助理。您可以向我咨询：\n1. 龟裂/裂纹的发生机理与调控方法\n2. 氧化皮压入对板面特性的影响与高压除鳞设置\n3. 机械拉伸划痕的安全防范与辊面磨削\n4. 非金属夹杂物与连铸工艺调节\n5. 酸洗麻面点蚀的成因与工艺参数纠偏' }
  ]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isSendingChat, setIsSendingChat] = useState<boolean>(false);
  const [isAssistantOpen, setIsAssistantOpen] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [chatMessages, isAssistantOpen]);

  const handleSendChat = async (e: React.FormEvent) => {
    e.preventDefault();
    const query = chatInput.trim();
    if (!query) return;

    setChatMessages((prev) => [...prev, { sender: 'user', text: query }]);
    setChatInput('');
    setIsSendingChat(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: query })
      });
      const data = await res.json();
      if (data.success && data.reply) {
        setChatMessages((prev) => [...prev, { sender: 'assistant', text: data.reply }]);
      } else {
        setChatMessages((prev) => [...prev, { sender: 'assistant', text: '抱歉，工艺助理服务遇到异常。' }]);
      }
    } catch (err) {
      setChatMessages((prev) => [...prev, { sender: 'assistant', text: '网络连接超时，无法呼叫大模型服务。' }]);
    } finally {
      setIsSendingChat(false);
    }
  };

  const workspaceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dropZoneRef = useRef<HTMLDivElement | null>(null);

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/records');
      const data = await res.json();
      if (data.success && data.records) {
        setHistory(data.records);
      }
    } catch (err) {
      console.error('Failed fetching history from server:', err);
      const storedData = localStorage.getItem('steel_inspections_v1');
      if (storedData) {
        try {
          setHistory(JSON.parse(storedData));
        } catch (e) {}
      }
    }
  };

  const fetchKPI = async () => {
    try {
      const res = await fetch('/api/kpi');
      const data = await res.json();
      if (data.success) {
        setKpiData(data);
      }
    } catch (err) {
      console.error('Failed fetching KPI:', err);
    }
  };

  // Initialize and load historical record database (API + Local Storage fallback)
  useEffect(() => {
    // Check Server Health & Mode
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => {
        setIsSystemHealthy(true);
        setAiEngineStatus(data.aiEngine || 'Standby Physics Expert Core');
      })
      .catch((err) => {
        console.error('Server status query issue:', err);
        setIsSystemHealthy(false);
        setAiEngineStatus('Offline Heuristics Standby');
      });

    fetchHistory();
    fetchKPI();
  }, []);

  // Poll training status if active or on mount
  useEffect(() => {
    fetchTrainStatus();
    const interval = setInterval(() => {
      fetchTrainStatus();
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  // Sync canvas redraw on preset selection or custom images upload
  useEffect(() => {
    redrawWorkspace();
  }, [selectedSample, customImageBase64]);

  const redrawWorkspace = () => {
    const canvas = workspaceCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (customImageBase64) {
      // Render uploaded base64 custom photo
      const img = new Image();
      img.onload = () => {
        // Uniform rendering scale fit to canvas dimensions
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // Add optional physical calibration overlay
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        // Centered crosshair
        ctx.beginPath();
        ctx.moveTo(canvas.width / 2, 0);
        ctx.lineTo(canvas.width / 2, canvas.height);
        ctx.moveTo(0, canvas.height / 2);
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.stroke();
        ctx.setLineDash([]);
      };
      img.src = customImageBase64;
    } else {
      // Procedurally draw high-fidelity metallurgical textured samples
      drawSteelPlate(canvas, selectedSample.renderType);
    }
  };

  useEffect(() => {
    if (showHeatmap && activeResult && activeResult.defects && heatmapCanvasRef.current) {
      const canvas = heatmapCanvasRef.current;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Draw a semi-transparent dark overlay first to focus on the heat spots
        ctx.fillStyle = 'rgba(7, 11, 25, 0.45)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        activeResult.defects.forEach((defect) => {
          const [ymin, xmin, ymax, xmax] = defect.bbox;
          // convert percent coordinates (0..100) to 640x360 pixels
          const cx = ((xmin + xmax) / 2) * 6.4;
          const cy = ((ymin + ymax) / 2) * 3.6;
          const radius = Math.max(50, ((xmax - xmin) * 6.4 + (ymax - ymin) * 3.6) * 0.75);
          
          const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, radius);
          grad.addColorStop(0, 'rgba(239, 68, 68, 0.95)'); // Vibrant red center
          grad.addColorStop(0.2, 'rgba(249, 115, 22, 0.65)'); // Orange
          grad.addColorStop(0.5, 'rgba(234, 179, 8, 0.3)'); // Yellow
          grad.addColorStop(0.8, 'rgba(59, 130, 246, 0.08)'); // Light blue edge
          grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
          
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    }
  }, [showHeatmap, activeResult]);

  // Convert current Canvas view to a data URL and execute analysis
  const handlePerformAnalysis = async () => {
    const canvas = workspaceCanvasRef.current;
    if (!canvas) return;

    setIsScanning(true);
    setActiveResult(null);
    setSelectedDefectId(null);
    
    // Simulate real visual light scanning laser sweep delay (for high-fidelity polish experience!)
    try {
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      
      const payload = {
        image: dataUrl,
        selectedSampleId: customImageBase64 ? undefined : selectedSample.id,
        filename: customImageBase64 ? customFilename : `${selectedSample.id}.jpg`
      };

      // Ensure minimal duration of laser animation to look amazing
      const requestPromise = fetch('/api/detect', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': session ? `Bearer ${session.token}` : ''
        },
        body: JSON.stringify(payload)
      });

      const delayPromise = new Promise((resolve) => setTimeout(resolve, 1500));
      
      const [response] = await Promise.all([requestPromise, delayPromise]);
      const data = await response.json();

      if (data.success && data.data) {
        setActiveResult(data.data);
        await fetchHistory();
        await fetchKPI();
      } else {
        alert(`缺陷诊断失败: ${data.error || '未知响应异常'}`);
      }
    } catch (err: any) {
      console.error('Inspect API request error:', err);
      alert(`无法连接至冶金大模型云服务器: ${err.message}`);
    } finally {
      setIsScanning(false);
    }
  };

  // Reset visual inputs, clear local custom cache
  const handleClearCustomUpload = () => {
    setCustomImageBase64(null);
    setCustomFilename('');
    setActiveResult(null);
    setSelectedDefectId(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Preset Selection Click handler
  const handleSelectPreset = (sample: DefectSample) => {
    setCustomImageBase64(null);
    setCustomFilename('');
    setSelectedSample(sample);
    setActiveResult(null);
    setSelectedDefectId(null);
  };

  // Process manual local PC photo upload
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      loadSelectedFile(files[0]);
    }
  };

  const loadSelectedFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('请载入有效的图像格式 (JPEG, PNG, WEBP)');
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target?.result) {
        setCustomImageBase64(event.target.result as string);
        setCustomFilename(file.name);
        setActiveResult(null);
        setSelectedDefectId(null);
      }
    };
    reader.readAsDataURL(file);
  };

  // HTML5 Drag Event handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (dropZoneRef.current) {
      dropZoneRef.current.classList.add('border-blue-500', 'bg-blue-950/20');
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    if (dropZoneRef.current) {
      dropZoneRef.current.classList.remove('border-blue-500', 'bg-blue-950/20');
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (dropZoneRef.current) {
      dropZoneRef.current.classList.remove('border-blue-500', 'bg-blue-950/20');
    }
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadSelectedFile(e.dataTransfer.files[0]);
    }
  };

  // Filter effect css generators for procedural custom polishing filters
  const getFilterCSS = () => {
    switch (viewMode) {
      case 'infrared':
        // Thermography simulated heat signature
        return 'brightness(0.9) contrast(1.6) saturate(1.8) hue-rotate(180deg) invert(1)';
      case 'edge':
        // High Contrast outline extraction representation
        return 'grayscale(1) contrast(3) invert(1) brightness(0.85)';
      case 'contrast':
        // Super Reticle Calibration view
        return 'contrast(1.95) saturate(1.1) brightness(1.05)';
      case 'standard':
      default:
        return 'none';
    }
  };

  // Delete specific history record item
  const handleDeleteHistoryItem = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm('确认清除此条钢铁缺陷检测历史记录吗?')) {
      try {
        const res = await fetch(`/api/records/${id}`, {
          method: 'DELETE'
        });
        const data = await res.json();
        if (data.success) {
          await fetchHistory();
        } else {
          const updated = history.filter((r) => r.id !== id);
          setHistory(updated);
          localStorage.setItem('steel_inspections_v1', JSON.stringify(updated));
        }
      } catch (err) {
        const updated = history.filter((r) => r.id !== id);
        setHistory(updated);
        localStorage.setItem('steel_inspections_v1', JSON.stringify(updated));
      }
    }
  };

  // Restore history record to inspection view
  const handleRecallHistoryRecord = (rec: InspectionRecord) => {
    if (rec.imageUrl) {
      setCustomImageBase64(rec.imageUrl);
      setCustomFilename(rec.imageName);
    } else {
      // Find preset by name or match
      const matched = samples.find(s => rec.imageName.includes(s.chineseName) || rec.imageName.includes(s.name));
      if (matched) {
        setSelectedSample(matched);
        setCustomImageBase64(null);
      }
    }
    setActiveResult(rec.result);
    setShowHistoryModal(false);
    setSelectedDefectId(null);
  };

  // CSV Export for Supervisor and Admin roles
  const handleExportCSV = () => {
    if (history.length === 0) {
      alert("没有检测记录可供导出！");
      return;
    }
    const headers = ["记录流水号", "检测时间", "标样名称", "板卷评级", "损伤指数(0-100)", "缺陷面积占比(%)", "缺陷统计", "理化成因分析", "排产处置建议"];
    const rows = history.map((rec) => {
      const defectsSummary = rec.result.defects?.map(d => `${d.typeName}(置信度:${(d.confidence*100).toFixed(0)}%)`).join(";") || "无缺陷";
      return [
        rec.id,
        rec.timestamp,
        rec.imageName,
        rec.result.overallStatus === 'Pass' ? '合格' : rec.result.overallStatus === 'Marginal' ? '降级' : '不合格',
        rec.result.severityIndex,
        rec.result.defectDensity.toFixed(1),
        defectsSummary,
        rec.result.chemicalExplanation.replace(/[\n\r,]/g, " "),
        rec.result.recommendedAction.replace(/[\n\r,]/g, " ")
      ];
    });
    const csvContent = "\uFEFF" + [headers.join(","), ...rows.map(r => r.map(val => `"${val}"`).join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `SteelEye_Defect_Report_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Human-in-the-loop manual override triggers
  const handleOpenAuditModal = (def: DefectItem) => {
    setAuditingDefect(def);
    setAuditClass(def.type);
    setAuditSeverity(def.severity);
    setAuditComment('');
  };

  const submitManualAudit = async () => {
    if (!activeResult || !auditingDefect) return;

    // Update active activeResult state with audited edits
    const typeNamesChinese: Record<string, string> = {
      Scratches: '手工标定-纵横划伤裂纹',
      Cracks: '人工校核-热拉力破晶微裂隙',
      Pitting: '人工校核-深度点状酸坑麻面',
      Inclusions: '手工标定-非金属杂质沉降物',
      Scale: '人工复核-表层粘连铁锈氧化皮',
      Patches: '手工标定-特异性异常浮渣斑块',
      None: '人工复验-优质合格轧段'
    };

    const updatedDefects = activeResult.defects.map((def) => {
      if (def.id === auditingDefect.id) {
        return {
          ...def,
          type: auditClass as any,
          typeName: typeNamesChinese[auditClass] || '人工重构缺陷标记',
          severity: auditSeverity,
          description: auditComment ? `${auditComment} (经工程师人工确权签发)` : `${def.description} (经校验重分类为 ${auditClass})`,
          confidence: 1.0 // Overridden by certified structural engineer
        };
      }
      return def;
    });

    // Recalculate severity indexes and densities on manually changed data
    let totalSeverityScore = 0;
    updatedDefects.forEach((d) => {
      const w = d.severity === 'High' ? 100 : d.severity === 'Medium' ? 60 : 30;
      totalSeverityScore += w;
    });
    
    // Smooth out updated stats
    const updatedSeverityIndex = Math.min(100, Math.max(5, Math.round(totalSeverityScore / (updatedDefects.length || 1))));
    const updatedAreaDensity = parseFloat((updatedDefects.length * 4.2).toFixed(1));
    const finalStatus = updatedSeverityIndex > 75 ? 'Fail' : updatedSeverityIndex > 30 ? 'Marginal' : 'Pass';

    const updatedResult = {
      ...activeResult,
      overallStatus: finalStatus as any,
      severityIndex: updatedSeverityIndex,
      defectDensity: updatedAreaDensity,
      defects: updatedDefects,
      chemicalExplanation: `${activeResult.chemicalExplanation} (添加工程师复核注记: 已修正为 [${auditClass}] 分类)`,
      recommendedAction: `[人工审计指令] 修正后建议: ${auditClass === 'None' ? '无需后处理，按合格放行。' : '根据各层工艺进行相应重型精整或去卷剪断。'}`
    };

    setActiveResult(updatedResult);

    try {
      await fetch(`/api/records/${activeResult.id}/audit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          final_result: updatedResult,
          reviewer: session?.username || 'engineer',
          review_status: 'corrected',
          note: auditComment || '人工标定修正分类'
        })
      });
      await fetchHistory();
      await fetchKPI();
    } catch (err) {
      console.error('Failed to submit manual audit to backend:', err);
    }

    setAuditingDefect(null);
  };

  // Convert coordinate ratio parameters nicely for rendering absolutely over workspace
  const getBBoxStyle = (bbox: [number, number, number, number]) => {
    const [ymin, xmin, ymax, xmax] = bbox;
    return {
      top: `${ymin}%`,
      left: `${xmin}%`,
      height: `${ymax - ymin}%`,
      width: `${xmax - xmin}%`,
    };
  };

  if (!session) {
    return (
      <div className="min-h-screen bg-[#070b19] flex items-center justify-center relative overflow-hidden select-none"
        style={{
          backgroundImage: 'radial-gradient(circle at 50% 50%, #1e293b 0%, #020617 100%)',
        }}
      >
        {/* Decorative Grid and Lights */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.015)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.015)_1px,transparent_1px)] bg-[size:30px_30px]" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-[100px]" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-[120px]" />

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.4 }}
          className="w-full max-w-md p-8 bg-slate-900/60 border border-slate-700/50 rounded-2xl shadow-2xl backdrop-blur-xl relative z-10 mx-4"
        >
          <div className="text-center mb-6">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center font-bold text-2xl text-white mx-auto shadow-[0_0_20px_rgba(37,99,235,0.4)] mb-3">S</div>
            <h2 className="text-lg font-bold text-white tracking-widest uppercase flex items-center justify-center gap-1.5">
              SteelEye <span className="font-normal text-xs font-mono text-slate-400">v4.2.0</span>
            </h2>
            <p className="text-[10px] text-blue-400 uppercase tracking-[0.25em] mt-1 text-center">
              SURFACE QUALITY INSPECTION SYSTEM
            </p>
          </div>

          <form onSubmit={handleLogin} className="space-y-4">
            {loginError && (
              <div className="p-3 bg-red-950/40 border border-red-800 text-red-300 rounded-lg text-xs flex items-center gap-2">
                <AlertCircle className="h-4 w-4 shrink-0 text-red-500" />
                <span>{loginError}</span>
              </div>
            )}

            <div className="space-y-1.5">
              <label className="text-[10px] font-mono font-bold tracking-widest text-slate-400 uppercase">USERNAME</label>
              <input
                type="text"
                value={usernameInput}
                onChange={(e) => setUsernameInput(e.target.value)}
                placeholder="请输入用户名 (如 admin/inspector/supervisor)"
                className="w-full text-xs font-medium bg-slate-950/40 border border-slate-700 text-white rounded-lg px-3 py-2.5 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] font-mono font-bold tracking-widest text-slate-400 uppercase">PASSWORD</label>
              <input
                type="password"
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
                placeholder="请输入密码"
                className="w-full text-xs font-medium bg-slate-950/40 border border-slate-700 text-white rounded-lg px-3 py-2.5 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              />
            </div>

            <button
              type="submit"
              disabled={isLoggingIn}
              className="w-full py-2.5 bg-blue-600 hover:bg-blue-550 text-white text-xs font-bold rounded-lg shadow-[0_0_15px_rgba(37,99,235,0.3)] transition hover:brightness-105 active:scale-[0.99] flex items-center justify-center gap-1.5"
            >
              {isLoggingIn ? (
                <>
                  <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                  验证中...
                </>
              ) : (
                '登录系统'
              )}
            </button>
          </form>

          <div className="mt-6 border-t border-slate-800 pt-4 text-[10px] text-slate-500 leading-relaxed font-sans text-center">
            <p className="font-semibold text-slate-400 mb-1">测试默认账户 (密码 123456):</p>
            <div className="flex flex-wrap gap-1.5 justify-center mt-1 text-[9px] font-mono">
              <span className="bg-slate-950/60 px-1.5 py-0.5 rounded text-slate-400">admin (管理员)</span>
              <span className="bg-slate-950/60 px-1.5 py-0.5 rounded text-slate-400">inspector (质检员)</span>
              <span className="bg-slate-950/60 px-1.5 py-0.5 rounded text-slate-400">supervisor (主管)</span>
            </div>
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#F1F5F9] text-slate-800 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* HEADER SECTION - Design Polish with beautiful Display Letterspacing */}
      <header className="h-16 bg-[#0F172A] text-white flex items-center justify-between px-6 shrink-0 border-b border-slate-700 shadow-md sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center font-bold text-lg text-white">S</div>
          <div>
            <h1 className="text-sm font-bold leading-none tracking-tight uppercase flex items-center gap-2">
              SteelEye <span className="font-normal text-slate-400 text-xs font-mono">v4.2.0-Pro</span>
              <span className="text-[9px] uppercase font-mono tracking-wider px-1.5 py-0.5 rounded bg-blue-900/60 text-blue-300 border border-blue-700">
                AI Vision Workstation
              </span>
            </h1>
            <p className="text-[10px] text-blue-400 uppercase tracking-widest mt-1">
              SURFACE QUALITY INSPECTION SYSTEM
            </p>
          </div>
        </div>

        {/* Real-time Telemetry Display Pairings - "一的字体不错" Monospaced Font Rhythm */}
        <div className="flex items-center gap-6 text-xs font-mono">
          <div className="hidden md:flex flex-col items-end">
            <span className="text-[9px] text-slate-400 uppercase font-bold tracking-wider">当前角色 / 登录人</span>
            <span className="text-blue-400 text-blue-400 flex items-center gap-1.5 font-semibold">
              <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse"></span>
              {getRoleChinese(session.role)}: {session.username}
            </span>
          </div>

          <span className="h-6 w-px bg-slate-700 hidden md:block"></span>

          <div className="flex items-center gap-3">
            {(session.role === 'admin' || session.role === 'supervisor') && (
              <button
                id="history_btn_id"
                onClick={() => setShowHistoryModal(true)}
                className="flex items-center gap-1.5 px-3 py-1 bg-slate-800 hover:bg-slate-700 rounded border border-slate-600 text-slate-200 text-xs font-semibold transition duration-150 shadow-sm"
              >
                <History className="h-3.5 w-3.5" />
                历史日志 ({history.length})
              </button>
            )}

            <button
              onClick={handleLogout}
              className="flex items-center gap-1.5 px-3 py-1 bg-red-950/60 hover:bg-red-900/60 rounded border border-red-800 text-red-200 text-xs font-semibold transition duration-150 shadow-sm"
            >
              <LogOut className="h-3.5 w-3.5" />
              退出登录
            </button>
            
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800/80 border border-slate-700 text-slate-300 font-bold text-xs">
              <span className={`h-2 w-2 rounded-full ${isSystemHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></span>
              <span>{isSystemHealthy ? 'System Live' : 'LATENCY_ERR'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* CORE WORKSPACE GRID SYSTEM - split grid with left controls panel, middle camera panel, right diagnostic report panel */}
      <main className="flex-grow flex flex-col xl:grid xl:grid-cols-4 overflow-hidden p-4 gap-4 max-h-[calc(100vh-64px)]">
        
        {/* PANEL 1: INGESTION & ADJUSTMENT (Side Panel Left) */}
        <section id="panel_controls_id" className="xl:col-span-1 p-4 space-y-4 flex flex-col bg-white border border-slate-200 rounded-xl shadow-sm overflow-y-auto">
          <div className="flex items-center gap-2 text-slate-800 border-b border-slate-100 pb-2">
            <Database className="h-4 w-4 text-blue-600" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">轧钢试样与材料录入</h2>
          </div>

          {/* Sample preset selector - "五的抛光不错" Clean custom steel preview indicators */}
          <div className="space-y-2.5">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-700">标样预设集 {!canSelectPresets && '🔒'}</span>
              <span className="text-blue-600 font-mono text-[11px] font-bold">{samples.length} 个基底选项</span>
            </div>

            <div className="grid grid-cols-1 gap-2">
              {samples.map((sample) => {
                const isSelected = selectedSample.id === sample.id && !customImageBase64;
                return (
                  <button
                    key={sample.id}
                    id={`preset_btn_${sample.id}`}
                    onClick={() => canSelectPresets && handleSelectPreset(sample)}
                    disabled={!canSelectPresets}
                    className={`text-left p-3 rounded-lg border text-xs transition duration-150 group relative overflow-hidden flex items-center justify-between ${
                      isSelected
                        ? 'bg-blue-50/70 border-blue-500 text-blue-900 shadow-sm font-semibold'
                        : 'bg-slate-50 border-slate-200 text-slate-700'
                    } ${!canSelectPresets ? 'opacity-65 cursor-not-allowed' : 'hover:bg-slate-100'}`}
                  >
                    <div className="space-y-0.5 z-10">
                      <div className="flex items-center gap-1.5">
                        <span className={`h-1.5 w-1.5 rounded-full ${
                          sample.type === 'None' ? 'bg-emerald-500' :
                          sample.type === 'Cracks' ? 'bg-red-500' :
                          sample.type === 'Scratches' ? 'bg-amber-500' : 'bg-blue-500'
                        }`}></span>
                        <h4 className="font-bold text-slate-800 tracking-wide">{sample.chineseName}</h4>
                      </div>
                      <p className="text-[11px] text-slate-500 leading-relaxed font-sans line-clamp-1">
                        {sample.description}
                      </p>
                    </div>
                    <ChevronRight className={`h-3.5 w-3.5 text-slate-400 group-hover:text-blue-600 transition-transform ${isSelected ? 'translate-x-1 text-blue-600' : ''}`} />
                    {/* Metal sheen overlay decorative inside buttons */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="border-t border-slate-150 my-1"></div>

          {/* Custom File Uploader Section with Touch/Drag-and-Drop Guidance */}
          <div className="space-y-2.5">
            <h3 className="text-xs font-semibold text-slate-700">本地生产相机快照上传</h3>
            <div
              ref={dropZoneRef}
              onDragOver={(e) => canScan && handleDragOver(e)}
              onDragLeave={(e) => canScan && handleDragLeave(e)}
              onDrop={(e) => canScan && handleDrop(e)}
              onClick={() => canScan && fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-4 text-center transition duration-150 group relative ${
                customImageBase64
                  ? 'border-blue-500 bg-blue-50/30'
                  : 'border-slate-200 bg-slate-50'
              } ${!canScan ? 'opacity-65 cursor-not-allowed' : 'hover:border-slate-350 cursor-pointer'}`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                className="hidden"
                disabled={!canScan}
              />
              
              <div className="flex flex-col items-center justify-center gap-1.5">
                <Upload className={`h-7 w-7 transition ${customImageBase64 ? 'text-blue-500' : 'text-slate-400'} ${canScan ? 'group-hover:text-slate-600' : ''}`} />
                <div className="text-xs">
                  {!canScan ? (
                    <div>
                      <p className="font-semibold text-slate-700">⚠️ 上传未授权 🔒</p>
                      <p className="text-[10px] text-slate-400 mt-0.5">仅限现场质检员与管理员操作</p>
                    </div>
                  ) : customImageBase64 ? (
                    <div className="space-y-1">
                      <p className="font-bold text-emerald-600">已载入自定义照片</p>
                      <p className="text-[10px] text-slate-500 font-mono select-all truncate max-w-[180px] mx-auto">{customFilename}</p>
                    </div>
                  ) : (
                    <div>
                      <p className="font-semibold text-slate-700">点击或拖拽文件至此</p>
                      <p className="text-[10px] text-slate-400 mt-0.5">支持高清冷/热轧钢板宽图</p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {customImageBase64 && canScan && (
              <button
                id="clear_upload_btn"
                onClick={(e) => { e.stopPropagation(); handleClearCustomUpload(); }}
                className="w-full flex items-center justify-center gap-2 py-1.5 border border-red-200 bg-red-50 hover:bg-red-100/80 text-red-600 rounded-lg text-xs font-semibold transition"
              >
                <Trash2 className="h-3.5 w-3.5" />
                清除上传，恢复标准件
              </button>
            )}
          </div>

          {/* AI Model Retraining Data Flywheel Panel */}
          {(isAiEngineer || isAdmin) && (
            <div className="bg-slate-900 text-white rounded-xl p-3 border border-slate-800 space-y-2.5 shadow-md">
              <div className="flex items-center gap-1.5 border-b border-slate-800 pb-1.5">
                <Sparkles className="h-3.5 w-3.5 text-amber-400" />
                <span className="text-xs font-bold uppercase tracking-wider text-slate-300">AI 增量学习训练飞轮</span>
              </div>
              <div className="space-y-2 text-[11px] leading-relaxed">
                <div className="flex justify-between">
                  <span className="text-slate-400">待重构样本数 (Bad Case):</span>
                  <span className="font-bold font-mono text-amber-400">{trainStatus?.correctedCount || 0} 个</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">当前训练状态:</span>
                  <span className={`font-bold uppercase font-mono ${
                    trainStatus?.status === 'training' ? 'text-blue-400 animate-pulse' :
                    trainStatus?.status === 'completed' ? 'text-emerald-400' :
                    trainStatus?.status === 'failed' ? 'text-rose-400' : 'text-slate-400'
                  }`}>
                    {trainStatus?.status === 'training' ? '进行中' :
                     trainStatus?.status === 'completed' ? '已完成' :
                     trainStatus?.status === 'failed' ? '失败' : '空闲'}
                  </span>
                </div>
                
                {trainStatus?.status === 'training' && (
                  <div className="space-y-1 mt-1">
                    <div className="flex justify-between text-[9px] text-slate-400">
                      <span>Epoch: {trainStatus.currentEpoch} / {trainStatus.totalEpochs}</span>
                      <span>进度: {trainStatus.progress}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div className="bg-gradient-to-r from-blue-500 to-cyan-400 h-full transition-all duration-300" style={{ width: `${trainStatus.progress}%` }}></div>
                    </div>
                  </div>
                )}
                
                {trainStatus?.logPreview && (
                  <div className="bg-slate-950 p-2 rounded text-[9px] font-mono text-slate-400 max-h-[100px] overflow-y-auto whitespace-pre-wrap border border-slate-800/80 mt-1 select-text font-semibold">
                    {trainStatus.logPreview}
                  </div>
                )}
                
                <button
                  onClick={handleStartTraining}
                  disabled={trainStatus?.status === 'training'}
                  className={`w-full py-1.5 rounded text-xs font-bold transition flex items-center justify-center gap-1.5 ${
                    trainStatus?.status === 'training'
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                      : 'bg-amber-500 hover:bg-amber-600 text-slate-950 shadow-sm cursor-pointer'
                  }`}
                >
                  <RefreshCw className={`h-3 w-3 ${trainStatus?.status === 'training' ? 'animate-spin' : ''}`} />
                  {trainStatus?.status === 'training' ? '增量训练执行中...' : '启动增量重训微调'}
                </button>
              </div>
            </div>
          )}

          {/* KPI & System Resources Panel */}
          {canViewKPI && (
            <div className="bg-slate-900 text-white rounded-xl p-3 border border-slate-800 space-y-2.5 shadow-md">
              <div className="flex items-center gap-1.5 border-b border-slate-800 pb-1.5">
                <Activity className="h-3.5 w-3.5 text-cyan-400 text-cyan-400" />
                <span className="text-xs font-bold uppercase tracking-wider text-slate-300">KPI 与运维监控看板</span>
              </div>
              {kpiData ? (
                <div className="space-y-2 text-[10px] leading-relaxed">
                  <div className="grid grid-cols-2 gap-1.5 text-center">
                    <div className="bg-slate-950/50 p-1 rounded border border-slate-800">
                      <div className="text-slate-400 text-[8px] uppercase tracking-wider font-bold">漏检率 (目标&lt;1%)</div>
                      <div className={`font-mono font-bold mt-0.5 text-xs ${kpiData.leakageRate < 0.01 ? 'text-emerald-400 text-emerald-400' : 'text-red-400'}`}>
                        {(kpiData.leakageRate * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="bg-slate-950/50 p-1 rounded border border-slate-800">
                      <div className="text-slate-400 text-[8px] uppercase tracking-wider font-bold">过杀率 (目标&lt;5%)</div>
                      <div className={`font-mono font-bold mt-0.5 text-xs ${kpiData.overkillRate < 0.05 ? 'text-emerald-400 text-emerald-400' : 'text-red-400'}`}>
                        {(kpiData.overkillRate * 100).toFixed(2)}%
                      </div>
                    </div>
                    <div className="bg-slate-950/50 p-1 rounded border border-slate-800">
                      <div className="text-slate-400 text-[8px] uppercase tracking-wider font-bold">平均检测延迟</div>
                      <div className="text-cyan-400 font-mono font-bold mt-0.5 text-xs">
                        {kpiData.avgDelayMs.toFixed(1)}ms
                      </div>
                    </div>
                    <div className="bg-slate-950/50 p-1 rounded border border-slate-800">
                      <div className="text-slate-400 text-[8px] uppercase tracking-wider font-bold">累计检测量</div>
                      <div className="text-slate-200 font-mono font-bold mt-0.5 text-xs">
                        {kpiData.totalInspections}
                      </div>
                    </div>
                  </div>
                  <div className="border-t border-slate-800/80 my-1"></div>
                  <div className="space-y-0.5 text-[9px] text-slate-400 font-mono">
                    <div className="flex justify-between"><span className="text-slate-500">YOLO推理设备:</span><span className="text-slate-300 font-bold">{kpiData.hardware.device}</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">YOLO模型权重:</span><span className="text-slate-300 truncate max-w-[130px]">{kpiData.hardware.yoloModel}</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">VLM服务商:</span><span className="text-slate-300">{kpiData.hardware.vlmProvider}</span></div>
                    <div className="flex justify-between"><span className="text-slate-500">数据库结构:</span><span className="text-slate-300">{kpiData.hardware.dbType}</span></div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-2 text-slate-500">加载中...</div>
              )}
            </div>
          )}
          
          <div className="border-t border-slate-150 mt-auto pt-3 space-y-3">
            {/* Visual Workspace calibrate control Sliders */}
            <div className="space-y-2.5">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-700">
                <Sliders className="h-3.5 w-3.5 text-blue-600" />
                <span>成像补偿与过滤校准</span>
                {!canAdjustSliders && <span className="text-red-500 text-[10px] flex items-center gap-0.5 ml-auto">🔒 锁定</span>}
              </div>
              
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-[11px] mb-0.5 font-mono">
                    <span className="text-slate-500">灵敏度对比度阈值</span>
                    <span className="text-blue-600 font-bold">{contrastThreshold}%</span>
                  </div>
                  <input
                    type="range"
                    min="15"
                    max="90"
                    value={contrastThreshold}
                    onChange={(e) => setContrastThreshold(Number(e.target.value))}
                    disabled={!canAdjustSliders}
                    className={`w-full h-1.5 bg-slate-100 rounded-lg appearance-none accent-blue-600 ${!canAdjustSliders ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                  />
                  <div className="flex justify-between text-[8px] text-slate-400 font-mono font-bold">
                    <span>LOW</span>
                    <span>STAND(50%)</span>
                    <span>HIGH_SENSITIVE</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] mb-0.5 font-mono">
                    <span className="text-slate-500">激光扫掠频段</span>
                    <span className="text-cyan-600 font-bold">{laserSpeed}00 Hz</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    value={laserSpeed}
                    onChange={(e) => setLaserSpeed(Number(e.target.value))}
                    disabled={!canAdjustSliders}
                    className={`w-full h-1.5 bg-slate-100 rounded-lg appearance-none accent-cyan-600 ${!canAdjustSliders ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                  />
                </div>
              </div>
            </div>

            {/* MAIN LAUNCH DIAGNOSIS BUTTON */}
            <button
              id="start_scan_btn_id"
              disabled={isScanning || !canScan}
              onClick={handlePerformAnalysis}
              className={`w-full py-3 px-4 rounded-xl font-bold flex items-center justify-center gap-2 shadow-md transition-all tracking-wider text-white ${
                isScanning
                  ? 'bg-slate-400 cursor-not-allowed text-slate-100'
                  : !canScan
                  ? 'bg-slate-350 dark:bg-slate-700 text-slate-500 cursor-not-allowed shadow-none'
                  : 'bg-gradient-to-r from-blue-600 via-cyan-500 to-emerald-500 hover:brightness-105 active:scale-[0.99]'
              }`}
            >
              {isScanning ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin animate-reverse" />
                  <span>高频扫描探测中...</span>
                </>
              ) : !canScan ? (
                <>
                  <Lock className="h-4 w-4 text-slate-500" />
                  <span>诊断未授权 (锁定)</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-current" />
                  <span>启动智能诊断扫描</span>
                </>
              )}
            </button>
          </div>
        </section>

        {/* PANEL 2 & 3: CENTER DISPLAY & METALLURGY WORKSPACE RETICLE (Middle Panels) */}
        <section className={`xl:col-span-2 flex flex-col bg-[#0F172A] overflow-hidden relative shadow-inner select-none transition-all duration-300 ease-in-out ${isFullscreen ? 'fixed inset-0 z-50 rounded-none border-0' : 'rounded-xl border border-slate-700'}`}>
          {/* Workstation Top Bar with Viewing Filters Mode Toggles - "五的抛光不错" Display calibration */}
          <div className="flex flex-wrap justify-between items-center gap-3 bg-[#0a101d] p-3 border-b border-slate-800">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
              <Camera className="h-4 w-4 text-cyan-400 text-cyan-400" />
              <span className="tracking-wide">智能工业 CCD 相机视域 FORWARD AREA</span>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              {/* Display filters selector - Represents high end aesthetic "抛光" */}
              <div className="flex rounded-lg overflow-hidden bg-slate-900 p-0.5 border border-slate-800 text-[11px]">
              <button
                id="filter_std_btn"
                onClick={() => setViewMode('standard')}
                className={`px-2.5 py-1 rounded font-bold transition flex items-center gap-1 ${
                  viewMode === 'standard' ? 'bg-slate-800 text-white shadow-sm' : 'text-slate-400 hover:text-white'
                }`}
                title="原片高精光泽层"
              >
                <Eye className="h-3 w-3" />
                真彩镜面
              </button>
              
              <button
                id="filter_ir_btn"
                onClick={() => setViewMode('infrared')}
                className={`px-2.5 py-1 rounded font-bold transition flex items-center gap-1 ${
                  viewMode === 'infrared' ? 'bg-rose-950/70 text-rose-400 shadow-sm' : 'text-slate-400 hover:text-rose-400'
                }`}
                title="仿真热成像应力突变区域"
              >
                <Flame className="h-3 w-3 text-rose-400" />
                红外应力
              </button>

              <button
                id="filter_edge_btn"
                onClick={() => setViewMode('edge')}
                className={`px-2.5 py-1 rounded font-bold transition flex items-center gap-1 ${
                  viewMode === 'edge' ? 'bg-cyan-950/70 text-cyan-400 shadow-sm' : 'text-slate-400 hover:text-cyan-400'
                }`}
                title="拉伸强度破坏边缘特征提取"
              >
                <Layers className="h-3 w-3 text-cyan-400" />
                边缘提取
              </button>

              <button
                id="filter_cnt_btn"
                onClick={() => setViewMode('contrast')}
                className={`px-2.5 py-1 rounded font-bold transition flex items-center gap-1 ${
                  viewMode === 'contrast' ? 'bg-amber-900/70 text-amber-400 shadow-sm' : 'text-slate-400 hover:text-amber-400'
                }`}
                title="对比度极化显示"
              >
                <ZoomIn className="h-3 w-3 text-amber-500" />
                偏光强化
              </button>
            </div>
            
            {/* Defect Heatmap Toggle */}
            {activeResult && activeResult.defects && activeResult.defects.length > 0 && (
              <button
                id="heatmap_toggle_btn"
                onClick={() => setShowHeatmap(!showHeatmap)}
                className={`px-2.5 py-1 rounded font-bold text-[11px] transition flex items-center gap-1 border ${
                  showHeatmap 
                    ? 'bg-rose-600 border-rose-500 text-white shadow-sm' 
                    : 'bg-slate-900 border-slate-800 border-slate-800 text-slate-400 hover:text-white'
                }`}
                title="切换缺陷分布密度热力图"
              >
                <Flame className="h-3 w-3" />
                <span>缺陷热力图</span>
              </button>
            )}
            
            <button
               onClick={() => setIsFullscreen(!isFullscreen)}
               className="p-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 border border-slate-700/50 text-slate-400 hover:text-white transition-colors"
               title={isFullscreen ? "退出全屏检视模式" : "开启全屏高分检视"}
            >
              {isFullscreen ? <Minimize className="h-4 w-4" /> : <Maximize className="h-4 w-4" />}
            </button>
            </div>
          </div>

          {/* THE MASTER VISUAL WORKSPACE STAGE - Canvas render and coordinate bbox representation */}
          <div className="flex-grow min-h-[300px] md:min-h-[420px] bg-[#07090c] relative flex items-center justify-center overflow-hidden group shadow-inner p-4">
            
            {/* Real steel procedural canvas viewport */}
            <div 
              className={`relative aspect-video max-w-full w-full flex items-center justify-center transition duration-300 ${isFullscreen ? 'max-h-[85vh]' : 'max-h-[500px]'}`}
              style={{ 
                transform: `scale(${zoomScale})`,
                transformOrigin: 'center center'
              }}
            >
              <canvas
                id="master_workspace_canvas"
                ref={workspaceCanvasRef}
                height={360}
                width={640}
                style={{ filter: getFilterCSS() }}
                className="shadow-2xl rounded-lg border border-white/5 max-w-full h-auto"
              />

              {showHeatmap && activeResult && activeResult.defects && activeResult.defects.length > 0 && (
                <canvas
                  id="heatmap_overlay_canvas"
                  ref={heatmapCanvasRef}
                  height={360}
                  width={640}
                  className="absolute inset-0 pointer-events-none z-20 max-w-full h-auto rounded-lg"
                />
              )}

              {/* LIVE LASER BEAM SWEEPER SWEEP - "五的抛光不错" Industrial laser animation overlay */}
              {isScanning && (
                <div 
                  className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 top-0 to-transparent shadow-[0_0_15px_#22d3ee] z-20 pointer-events-none"
                  style={{
                    animation: `laser-sweep ${8.5 / laserSpeed}s infinite linear`
                  }}
                />
              )}

              {/* CO-ORDINATE BOUNDING BOX OVERLAYS */}
              {!isScanning && activeResult && activeResult.defects && activeResult.defects.map((defect) => {
                const isSelected = selectedDefectId === defect.id;
                const isHovered = hoveredDefectId === defect.id;
                
                // Color mapping logic depending on severity levels
                const colorTheme = 
                  defect.severity === 'High' 
                    ? { border: 'border-red-500', bg: 'bg-red-500/15', text: 'text-red-400', labelBg: 'bg-red-950 text-red-300 border-red-500' }
                    : defect.severity === 'Medium'
                    ? { border: 'border-amber-500', bg: 'bg-amber-500/10', text: 'text-amber-400', labelBg: 'bg-amber-950 text-amber-300 border-amber-500' }
                    : { border: 'border-blue-500', bg: 'bg-blue-500/10', text: 'text-blue-400', labelBg: 'bg-blue-950 text-blue-300 border-blue-500' };

                return (
                  <div
                    key={defect.id}
                    onClick={() => {
                      setSelectedDefectId(isSelected ? null : defect.id);
                    }}
                    onMouseEnter={() => setHoveredDefectId(defect.id)}
                    onMouseLeave={() => setHoveredDefectId(null)}
                    className={`absolute border-2 pointer-events-auto cursor-pointer transition-all duration-200 z-10 ${colorTheme.border} ${colorTheme.bg} ${
                      isSelected ? 'ring-2 ring-white/40 scale-[1.01] shadow-2xl border-dashed' : ''
                    } ${isHovered ? 'brightness-125 border-opacity-100 scale-[1.005]' : 'border-opacity-70'}`}
                    style={getBBoxStyle(defect.bbox)}
                    title={`${defect.typeName} (${defect.severity})`}
                  >
                    {/* BBox Reticle Corner corners designs - Polish */}
                    <div className="absolute top-0 left-0 w-2.5 h-2.5 border-t-2 border-l-2 border-white -translate-x-[2px] -translate-y-[2px]" />
                    <div className="absolute top-0 right-0 w-2.5 h-2.5 border-t-2 border-r-2 border-white translate-x-[2px] -translate-y-[2px]" />
                    <div className="absolute bottom-0 left-0 w-2.5 h-2.5 border-b-2 border-l-2 border-white -translate-x-[2px] translate-y-[2px]" />
                    <div className="absolute bottom-0 right-0 w-2.5 h-2.5 border-b-2 border-r-2 border-white translate-x-[2px] translate-y-[2px]" />

                    {/* Interactive label tag inside spatial grid */}
                    <div className={`absolute top-1 left-2 rounded px-1.5 py-0.5 text-[10px] font-mono tracking-wide flex items-center gap-1 border ${colorTheme.labelBg} ${
                      isHovered || isSelected ? 'opacity-100' : 'opacity-85'
                    }`}>
                      <span className="font-bold">{defect.typeName}</span>
                      <span className="opacity-70">{(defect.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Static diagnostic instructions inside central viewer when idle */}
            {!activeResult && !isScanning && (
              <div className="absolute inset-x-8 bottom-8 text-center bg-slate-950/90 border border-slate-800 p-4 rounded-xl backdrop-blur-sm shadow-md z-10 pointer-events-none max-w-sm mx-auto">
                <p className="text-xs text-blue-400 font-bold mb-1 flex justify-center items-center gap-1.5">
                  <Info className="h-3.5 w-3.5" />
                  <span>图像光学检测点位就绪</span>
                </p>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  点击左侧的【启动智能诊断扫描】，系统将进行高速偏振投影、晶界显形并交付大模型分析。
                </p>
              </div>
            )}

            {/* Active Scan Laser sweeping indicator loading */}
            {isScanning && (
              <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-[1px] flex flex-col items-center justify-center z-15">
                <div className="relative bg-slate-900 p-6 rounded-2xl border border-slate-700 shadow-xl flex flex-col items-center">
                  <div className="relative">
                    {/* Glowing radar sweep circle */}
                    <div className="h-16 w-16 rounded-full border-2 border-cyan-400 animate-ping absolute -inset-2"></div>
                    <div className="relative h-12 w-12 rounded-full border-t border-b border-cyan-400 animate-spin bg-cyan-950/40 flex items-center justify-center text-cyan-400">
                      <Activity className="h-6 w-6" />
                    </div>
                  </div>
                  <p className="text-cyan-400 text-xs font-mono font-bold mt-5 tracking-[0.2em] uppercase">
                    RESOLVING GRID COORDINATES...
                  </p>
                  <p className="text-[10px] text-slate-500 font-mono mt-1 select-none">
                    RUNNING EDGE PIXEL METRICS...
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* VIEWPORT CONTROLLER BUTTON BAR - ZOOM / GRID CALIBRATE */}
          <div className="h-12 bg-slate-950 border-t border-slate-800 flex items-center justify-between px-4 shrink-0 text-xs text-slate-400">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setZoomScale(Math.min(1.8, zoomScale + 0.1))}
                className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 text-[11px] font-bold transition"
                title="放大探针视窗"
              >
                ＋
              </button>
              <button
                onClick={() => setZoomScale(Math.max(1.0, zoomScale - 0.1))}
                className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 text-[11px] font-bold transition"
                title="缩小探针视窗"
              >
                －
              </button>
              {zoomScale !== 1 && (
                <button
                  onClick={() => setZoomScale(1)}
                  className="text-slate-500 hover:text-white text-[10px] ml-1 font-bold"
                >
                  重置
                </button>
              )}
              <span className="h-4 w-px bg-slate-800"></span>
              <span className="text-slate-500 select-none">Reticle scale: <b className="text-slate-300">{zoomScale.toFixed(1)}x</b></span>
            </div>

            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-500"></span>
              <span className="text-[11px] uppercase font-mono tracking-wider text-slate-400">CCD OVERLAY FEED ACTIVE</span>
            </div>
          </div>
        </section>

        {/* PANEL 4: METALLOGRAPHY DIAGNOSTICS TERMINAL (Right Panel) */}
        <section className="xl:col-span-1 p-4 space-y-4 flex flex-col bg-white border border-slate-200 rounded-xl shadow-sm overflow-y-auto">
          
          {/* AI DECISION GRADE METALLURGY METRICS - "二的布局不错" Metric layouts */}
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-slate-800 border-b border-slate-100 pb-2">
              <Activity className="h-4 w-4 text-slate-500 text-slate-500" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">冶金理化检测与分级评估</h2>
            </div>

            {activeResult ? (
              <motion.div
                key={activeResult.id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
                className="space-y-3"
              >
                {/* Visual score KPI meter */}
                <div className="grid grid-cols-3 gap-2 text-center">
                  
                  {/* METRIC 1: Overall Coil batch status */}
                  <div className={`p-2.5 rounded-xl border flex flex-col justify-center items-center ${
                    activeResult.overallStatus === 'Pass' 
                      ? 'bg-emerald-50 border-emerald-200 text-emerald-800 text-emerald-800' 
                      : activeResult.overallStatus === 'Marginal'
                      ? 'bg-amber-50 border-amber-200 text-amber-800 text-amber-800 text-amber-800'
                      : 'bg-rose-50 border-rose-200 text-rose-800 text-rose-800'
                  }`}>
                    <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">板卷评级</span>
                    {activeResult.overallStatus === 'Pass' ? (
                      <ShieldCheck className="h-4.5 w-4.5 mb-0.5 text-emerald-600 animate-pulse" />
                    ) : activeResult.overallStatus === 'Marginal' ? (
                      <AlertTriangle className="h-4.5 w-4.5 mb-0.5 text-amber-600" />
                    ) : (
                      <ShieldAlert className="h-4.5 w-4.5 mb-0.5 text-rose-600" />
                    )}
                    <span className="text-sm font-bold font-sans">
                      {activeResult.overallStatus === 'Pass' ? '合格' : activeResult.overallStatus === 'Marginal' ? '降级' : '不合格'}
                    </span>
                  </div>

                  {/* METRIC 2: Severity Index */}
                  <div className="bg-slate-50 border border-slate-200 border-slate-200 p-2.5 rounded-xl flex flex-col justify-center items-center">
                    <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">损伤指数</span>
                    <span className="text-lg font-extrabold font-mono text-slate-800 leading-tight">
                      {activeResult.severityIndex}
                    </span>
                    <span className="text-[8px] font-bold text-slate-400 font-mono mt-0.5">Scale 0-100</span>
                  </div>

                  {/* METRIC 3: Defect Density AREA ratio */}
                  <div className="bg-slate-50 border border-slate-200 p-2.5 rounded-xl flex flex-col justify-center items-center">
                    <span className="text-[9px] text-slate-500 uppercase font-bold tracking-wider mb-1">面积占比</span>
                    <span className="text-lg font-extrabold font-mono text-slate-800 leading-tight">
                      {activeResult.defectDensity.toFixed(1)}%
                    </span>
                    <span className="text-[8px] font-bold text-slate-400 font-mono mt-0.5">Defect Area</span>
                  </div>
                </div>

                {/* Simulated database details if any */}
                {activeResult.isSimulated && (
                  <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-[11px] text-amber-800 text-amber-800 space-y-1">
                    <p className="font-bold flex items-center gap-1">
                      <AlertCircle className="h-3.5 w-3.5 text-amber-600" />
                      <span>专家诊断系统提示</span>
                    </p>
                    <p className="leading-relaxed opacity-90">{activeResult.simulatedReason}</p>
                  </div>
                )}
              </motion.div>
            ) : (
              <div className="py-6 text-center text-xs text-slate-400 border border-slate-150 border-dashed rounded-xl">
                <Database className="h-6 w-6 text-slate-300 mx-auto mb-1.5" />
                <span>等待分析报表产出...</span>
              </div>
            )}
          </div>

          {/* CHEMISTRY ROOT CAUSE & METALLURGY EXPLANATIONS */}
          <div className="pt-4 space-y-3">
            <div className="flex items-center gap-2 text-white border-b border-[#1f2833]/60 pb-1.5">
              <Info className="h-4 w-4 text-blue-600" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">晶包缺陷冶金成因机理 {!canViewRAG && '🔒'}</h2>
            </div>

            {!canViewRAG ? (
              <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl text-center">
                <Lock className="h-6 w-6 text-slate-400 mx-auto mb-1.5" />
                <p className="text-xs font-semibold text-slate-700">成因分析已锁定</p>
                <p className="text-[10px] text-slate-400 leading-relaxed mt-0.5">
                  仅限现场质检员、工艺工程师与管理员查看根因分析
                </p>
              </div>
            ) : activeResult ? (
              <motion.div
                key={activeResult.id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.1, ease: "easeOut" }}
                className="space-y-3 text-xs leading-relaxed text-slate-600"
              >
                <div className="p-3 bg-slate-50 border border-slate-200 rounded-xl relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-1 bg-slate-200 text-slate-600 rounded-bl text-[8px] uppercase font-mono tracking-widest font-bold">
                    Micro-Mechanics
                  </div>
                  <p className="font-semibold text-slate-700 text-[11px] mb-1.5 font-mono flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 bg-blue-500 rounded-full"></span>
                    <span>理化显微成因析置:</span>
                  </p>
                  <p className="indent-5 leading-relaxed text-slate-600 text-justify">
                    {activeResult.chemicalExplanation}
                  </p>
                </div>

                <div className="p-3 bg-blue-50/40 border border-blue-100 rounded-xl">
                  <p className="font-semibold text-blue-800 text-[11px] mb-1.5 font-mono flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 bg-blue-500 rounded-full"></span>
                    <span>再加工处置与排产指令建议:</span>
                  </p>
                  <p className="leading-relaxed font-sans text-slate-700">
                    {activeResult.recommendedAction}
                  </p>
                </div>
              </motion.div>
            ) : (
              <p className="text-xs text-slate-400 text-justify leading-relaxed">
                分析报告将输出微应力变化、冷却及结晶时相变、氧化物杂质沉降或热连轧过度压应力等理化学机理原因。
              </p>
            )}
          </div>

          {/* DETAILED DEFECT LIST / RETICLE POSITION REGIONS */}
          <div className="pt-3 space-y-3 flex-1 flex flex-col">
            <div className="flex justify-between items-center text-slate-800 border-b border-slate-100 pb-1.5">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className="h-4 w-4 text-blue-600" />
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">微细区域病灶扫描详单</h2>
              </div>
              {activeResult && activeResult.defects && activeResult.defects.length > 0 && (
                <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-705 border border-slate-200 font-bold">
                  {activeResult.defects.length} 个目标
                </span>
              )}
            </div>

            {activeResult && activeResult.defects && activeResult.defects.length > 0 ? (
              <motion.div
                key={activeResult.id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2, ease: "easeOut" }}
                className="space-y-2 max-h-[220px] overflow-y-auto pr-1"
              >
                {activeResult.defects.map((defect) => {
                  const isHovered = hoveredDefectId === defect.id;
                  const isSelected = selectedDefectId === defect.id;
                  
                  return (
                    <div
                      key={defect.id}
                      id={`defect_card_${defect.id}`}
                      onMouseEnter={() => setHoveredDefectId(defect.id)}
                      onMouseLeave={() => setHoveredDefectId(null)}
                      onClick={() => setSelectedDefectId(isSelected ? null : defect.id)}
                      className={`p-3 rounded-xl border text-xs cursor-pointer transition-all ${
                        isSelected || isHovered
                          ? 'bg-blue-50/70 border-blue-400 shadow-sm'
                          : 'bg-slate-50 border-slate-200 hover:bg-slate-100/80 text-slate-800'
                      }`}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <div className="flex items-center gap-1.5">
                          <span className={`h-2 w-2 rounded-full ${
                            defect.severity === 'High' ? 'bg-red-500' :
                            defect.severity === 'Medium' ? 'bg-amber-400' : 'bg-blue-400'
                          }`}></span>
                          <span className="font-bold text-slate-800">{defect.typeName}</span>
                        </div>
                        <span className={`px-1.5 py-0.5 rounded-[4px] text-[9px] font-mono font-bold ${
                          defect.severity === 'High' 
                            ? 'bg-red-100 text-red-700 border border-red-200' 
                            : defect.severity === 'Medium'
                            ? 'bg-amber-100 text-amber-700 border border-amber-200'
                            : 'bg-blue-100 text-blue-700 border border-blue-200'
                        }`}>
                          {defect.severity === 'High' ? '高危险' : defect.severity === 'Medium' ? '中等' : '低危害'}
                        </span>
                      </div>
                      
                      <p className="text-[11px] text-slate-500 leading-relaxed font-sans mt-0.5">
                        {defect.description}
                      </p>

                      <div className="flex justify-between items-center text-[10px] text-slate-400 font-mono mt-2 pt-2 border-t border-slate-200/60">
                        <span>CCD 坐标: [Y:{defect.bbox[0]}, X:{defect.bbox[1]}]</span>
                        <div className="flex items-center gap-2.5">
                          <span className="text-emerald-600 font-bold">置信值 {(defect.confidence * 100).toFixed(0)}%</span>
                          {/* Manual Audit Trigger button */}
                          {canAudit ? (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleOpenAuditModal(defect);
                              }}
                              className="text-blue-600 hover:text-blue-700 font-bold underline underline-offset-2"
                              title="修正缺陷属性分类与危险等级"
                            >
                              专家复审
                            </button>
                          ) : (
                            <span className="text-slate-400 flex items-center gap-0.5" title="复审未授权">
                              <Lock className="h-3 w-3" />
                              已锁定
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </motion.div>
            ) : activeResult ? (
              <motion.div
                key={activeResult.id}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: 0.2, ease: "easeOut" }}
                className="py-6 text-center text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-xl space-y-1"
              >
                <ShieldCheck className="h-6 w-6 text-emerald-600 mx-auto mb-1 animate-pulse" />
                <p className="font-bold">材质检验合格</p>
                <p className="text-[10px] text-slate-500">板面上表面精微扫描仪内未探查到可见缺陷。</p>
              </motion.div>
            ) : (
              <p className="text-xs text-slate-400 text-center py-6 border border-slate-150 border-dashed rounded-xl">
                请先在左下角启动诊断。
              </p>
            )}
          </div>
        </section>
      </main>

      {/* METALLURGY EXPERT AUDIT OVERRIDE MODAL - Human in the loop overrides */}
      {auditingDefect && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
          <div className="bg-white border border-slate-200 rounded-2xl w-full max-w-md p-6 relative shadow-2xl text-slate-80s text-slate-800">
            <button
              onClick={() => setAuditingDefect(null)}
              className="absolute top-4 right-4 text-slate-400 hover:text-slate-700 p-1 rounded-full hover:bg-slate-100"
            >
              <X className="h-5 w-5" />
            </button>

            <div className="flex items-center gap-2 mb-4 text-slate-800">
              <Settings className="h-5 w-5 text-blue-600" />
              <h3 className="text-sm font-bold uppercase tracking-wider">质检工程师复核审计窗</h3>
            </div>

            <div className="text-xs text-slate-600 mb-4 bg-slate-50 p-3 rounded-lg border border-slate-150 font-mono">
              <p>缺陷ID: <span className="font-bold text-slate-800">{auditingDefect.id}</span></p>
              <p className="mt-1">当前AI评语: <span className="text-slate-500">{auditingDefect.description}</span></p>
            </div>

            <div className="space-y-4">
              {/* Dropdown class taxonomy type selection */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-700 block">缺陷重归类分类 (Type)</label>
                <select
                  value={auditClass}
                  onChange={(e) => setAuditClass(e.target.value)}
                  className="w-full text-xs font-semibold bg-white border border-slate-200 rounded-lg px-3 py-2 text-slate-800 focus:outline-none focus:border-blue-500"
                >
                  <option value="Scratches">Scratches (划痕拉伤)</option>
                  <option value="Cracks">Cracks (边缘撕裂裂纹)</option>
                  <option value="Pitting">Pitting (超酸蚀集聚凹坑)</option>
                  <option value="Inclusions">Inclusions (保护渣钢内非金属杂质)</option>
                  <option value="Scale">Scale (局部粘结状氧化铁鳞皮)</option>
                  <option value="Patches">Patches (水油乳化清洗渍斑)</option>
                  <option value="None">None (无缺陷，重定为合格板面)</option>
                </select>
              </div>

              {/* Severity selection */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-700 block">危害严酷度 (Severity)</label>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  {(['Low', 'Medium', 'High'] as const).map((sev) => (
                    <button
                      key={sev}
                      type="button"
                      onClick={() => setAuditSeverity(sev)}
                      className={`py-1.5 px-2 rounded-lg font-bold border transition ${
                        auditSeverity === sev
                          ? sev === 'High' ? 'bg-red-50 border-red-300 text-red-700' : sev === 'Medium' ? 'bg-amber-50 border-amber-300 text-amber-700' : 'bg-blue-50 border-blue-300 text-blue-700'
                          : 'bg-white border-slate-200 text-slate-500 hover:text-slate-700'
                      }`}
                    >
                      {sev === 'High' ? 'HIGH危险' : sev === 'Medium' ? 'MEDIUM中等' : 'LOW微弱'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Expert physical commentary */}
              <div className="space-y-1.5">
                <label className="text-xs font-semibold text-slate-700 block">工程师理化审计备注 (必填)</label>
                <textarea
                  rows={3}
                  value={auditComment}
                  onChange={(e) => setAuditComment(e.target.value)}
                  placeholder="请输入人工分析备注。例如：辊片局部粘砂导致拉伸滑痕..."
                  className="w-full text-xs bg-white border border-slate-200 rounded-lg p-2.5 text-slate-800 focus:outline-none focus:border-blue-500 placeholder-slate-400"
                />
              </div>

              <div className="flex gap-2.5 pt-2">
                <button
                  type="button"
                  onClick={() => setAuditingDefect(null)}
                  className="flex-1 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs text-slate-700 font-semibold border border-slate-200 transition"
                >
                  取消
                </button>
                <button
                  type="button"
                  onClick={submitManualAudit}
                  className="flex-1 py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-xs text-white font-bold transition flex items-center justify-center gap-1.5 shadow-sm"
                >
                  <Check className="h-4 w-4" />
                  核查签发(Override)
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* HISTORICAL ARCHIVE PORTAL MODAL */}
      {showHistoryModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
          <div className="bg-white border border-slate-200 rounded-2xl w-full max-w-2xl p-6 relative shadow-2xl flex flex-col max-h-[85vh] text-slate-800">
            <button
              onClick={() => setShowHistoryModal(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-slate-700 p-1.5 rounded-full hover:bg-slate-100 bg-slate-50 border border-slate-100"
            >
              <X className="h-4.5 w-4.5" />
            </button>

            <div className="flex items-center gap-2.5 mb-3 border-b border-slate-100 pb-3">
              <History className="h-5 w-5 text-blue-600" />
              <div>
                <h3 className="text-sm font-bold uppercase tracking-wider text-slate-800">轧材缺陷光学历史探查日志</h3>
                <p className="text-[11px] text-slate-400 text-slate-500">检测履历已安全离线保存在本地快照数据库中</p>
              </div>
              <div className="ml-auto mr-10 flex items-center gap-2">
                <button
                  onClick={handleExportCSV}
                  className="flex items-center gap-1 px-2.5 py-1 bg-emerald-600 hover:bg-emerald-700 text-white rounded text-xs font-bold transition shadow-sm"
                  title="导出 CSV"
                >
                  <FileSpreadsheet className="h-3 w-3" />
                  CSV 导出
                </button>
                <button
                  onClick={() => window.open('/api/export/html', '_blank')}
                  className="flex items-center gap-1 px-2.5 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-bold transition shadow-sm"
                  title="生成 HTML 质量报告"
                >
                  <Eye className="h-3 w-3" />
                  HTML 报告
                </button>
                <button
                  onClick={() => window.open('/api/export/badcase', '_blank')}
                  className="flex items-center gap-1 px-2.5 py-1 bg-amber-600 hover:bg-amber-700 text-white rounded text-xs font-bold transition shadow-sm"
                  title="打包 Bad Case ZIP 数据集"
                >
                  <Upload className="h-3 w-3" />
                  Bad Case ZIP
                </button>
              </div>
            </div>

            {/* Tab Navigation in History Modal */}
            <div className="flex border-b border-slate-100 mb-4 text-xs font-semibold gap-4 shrink-0">
              <button
                onClick={() => setHistoryModalTab('list')}
                className={`pb-2 border-b-2 px-1 transition ${historyModalTab === 'list' ? 'border-blue-600 text-blue-600' : 'border-transparent text-slate-400 hover:text-slate-700'}`}
              >
                历史记录清单
              </button>
              <button
                onClick={() => setHistoryModalTab('stats')}
                className={`pb-2 border-b-2 px-1 transition ${historyModalTab === 'stats' ? 'border-blue-600 text-blue-600' : 'border-transparent text-slate-400 hover:text-slate-700'}`}
              >
                📊 KPI 与质量趋势分析
              </button>
            </div>

            {historyModalTab === 'stats' ? (() => {
              const passCount = history.filter(h => h.result.overallStatus === 'Pass').length;
              const passRateVal = history.length > 0 ? ((passCount / history.length) * 100).toFixed(1) : '100.0';
              const failCount = history.filter(h => h.result.overallStatus === 'Fail').length;
              const failRateVal = history.length > 0 ? ((failCount / history.length) * 100).toFixed(1) : '0.0';

              // 1. Calculate defect types count (for Donut Pie Chart)
              const defectCounts: Record<string, number> = {};
              history.forEach((h) => {
                if (h.result.defects) {
                  h.result.defects.forEach((d) => {
                    defectCounts[d.type] = (defectCounts[d.type] || 0) + 1;
                  });
                }
              });

              const totalDefectCount = Object.values(defectCounts).reduce((a, b) => a + b, 0);
              const pieData = Object.entries(defectCounts).map(([type, count]) => ({
                name: type,
                value: count,
                percentage: totalDefectCount > 0 ? (count / totalDefectCount) * 100 : 0
              })).sort((a, b) => b.value - a.value);

              // 2. Generate 30-day area trend data points
              const last30Days = [];
              for (let i = 29; i >= 0; i--) {
                const d = new Date();
                d.setDate(d.getDate() - i);
                const dateString = d.toISOString().slice(0, 10);
                last30Days.push(dateString);
              }

              const dateMap: Record<string, { totalArea: number; count: number }> = {};
              history.forEach((h) => {
                const dateStr = h.timestamp.slice(0, 10);
                if (!dateMap[dateStr]) {
                  dateMap[dateStr] = { totalArea: 0, count: 0 };
                }
                dateMap[dateStr].totalArea += h.result.defectDensity;
                dateMap[dateStr].count += 1;
              });

              const trendData = last30Days.map((date) => {
                const entry = dateMap[date];
                return {
                  date: date.slice(5), // MM-DD
                  value: entry ? (entry.totalArea / entry.count) : 0.0
                };
              });

              return (
                <div className="flex-1 overflow-y-auto space-y-5 py-2 pr-1">
                  {/* KPI Cards Grid */}
                  <div className="grid grid-cols-3 gap-3 text-center shrink-0">
                    <div className="bg-slate-50 border border-slate-200 p-3 rounded-xl shadow-sm">
                      <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider block mb-1">累计检测数</span>
                      <span className="text-xl font-bold font-mono text-slate-800">{history.length}</span>
                      <span className="text-[8px] block text-slate-400 font-mono mt-0.5">Total Scans</span>
                    </div>
                    <div className="bg-emerald-50 border border-emerald-200 p-3 rounded-xl shadow-sm">
                      <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider block mb-1">板卷合格率</span>
                      <span className="text-xl font-bold font-mono text-emerald-700">{passRateVal}%</span>
                      <span className="text-[8px] block text-slate-400 font-mono mt-0.5">Pass Rate</span>
                    </div>
                    <div className="bg-rose-50 border border-rose-200 p-3 rounded-xl shadow-sm">
                      <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider block mb-1">缺陷卷占比</span>
                      <span className="text-xl font-bold font-mono text-rose-700">{failRateVal}%</span>
                      <span className="text-[8px] block text-slate-400 font-mono mt-0.5">Fail Rate</span>
                    </div>
                  </div>

                  {/* Charts Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {/* Donut Chart Card */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
                      <h4 className="text-[11px] font-bold text-slate-400 self-start mb-3 uppercase tracking-wider">Top 缺陷分布比例</h4>
                      {pieData.length === 0 ? (
                        <div className="flex-1 flex items-center justify-center text-xs text-slate-500 py-16">
                          无缺陷统计数据
                        </div>
                      ) : (
                        <div className="w-full flex flex-col sm:flex-row items-center gap-6 justify-center">
                          <div className="relative shrink-0">
                            <svg width="130" height="130" viewBox="0 0 130 130" className="overflow-visible">
                              <circle cx="65" cy="65" r="40" fill="none" stroke="#1e293b" strokeWidth="10" />
                              {(() => {
                                let accumulatedPercent = 0;
                                return pieData.map((slice) => {
                                  const strokeColor = 
                                    slice.name === 'Scratches' ? '#f59e0b' :
                                    slice.name === 'Cracks' ? '#ef4444' :
                                    slice.name === 'Pitting' ? '#06b6d4' :
                                    slice.name === 'Inclusions' ? '#8b5cf6' :
                                    slice.name === 'Scale' ? '#ec4899' : '#10b981';
                                  
                                  // r=40 -> circumference = 2 * PI * 40 = 251.327
                                  const dashArray = `${slice.percentage * 2.51327} 251.327`;
                                  const dashOffset = `${-accumulatedPercent * 2.51327}`;
                                  accumulatedPercent += slice.percentage;
                                  
                                  return (
                                    <circle
                                      key={slice.name}
                                      cx="65"
                                      cy="65"
                                      r="40"
                                      fill="none"
                                      stroke={strokeColor}
                                      strokeWidth="10"
                                      strokeDasharray={dashArray}
                                      strokeDashoffset={dashOffset}
                                      transform="rotate(-90 65 65)"
                                      className="transition-all duration-300 hover:stroke-[13px] cursor-pointer"
                                    >
                                      <title>{`${slice.name}: ${slice.value} 处 (${slice.percentage.toFixed(1)}%)`}</title>
                                    </circle>
                                  );
                                });
                              })()}
                              <text x="65" y="62" fill="#94a3b8" textAnchor="middle" fontSize="9" fontWeight="bold">缺陷总量</text>
                              <text x="65" y="78" fill="#ffffff" textAnchor="middle" fontSize="13" fontFamily="monospace" fontWeight="bold">{totalDefectCount}</text>
                            </svg>
                          </div>
                          
                          <div className="space-y-1.5 flex-1 min-w-[120px]">
                            {pieData.slice(0, 5).map((slice) => {
                              const strokeColor = 
                                slice.name === 'Scratches' ? 'bg-[#f59e0b]' :
                                slice.name === 'Cracks' ? 'bg-[#ef4444]' :
                                slice.name === 'Pitting' ? 'bg-[#06b6d4]' :
                                slice.name === 'Inclusions' ? 'bg-[#8b5cf6]' :
                                slice.name === 'Scale' ? 'bg-[#ec4899]' : 'bg-[#10b981]';
                              
                              return (
                                <div key={slice.name} className="flex items-center justify-between text-[10px] text-slate-350">
                                  <div className="flex items-center gap-1.5 truncate">
                                    <span className={`h-2 w-2 rounded-full ${strokeColor}`}></span>
                                    <span className="truncate">{slice.name}</span>
                                  </div>
                                  <span className="font-mono text-slate-400 font-bold">{slice.percentage.toFixed(1)}%</span>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Trend Line Chart Card */}
                    <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col">
                      <h4 className="text-[11px] font-bold text-slate-400 mb-3 uppercase tracking-wider text-left">近 30 天缺陷面积占比趋势</h4>
                      <div className="flex-grow flex items-center justify-center min-h-[130px]">
                        <svg viewBox="0 0 500 180" className="w-full h-auto overflow-visible">
                          {/* Y Axis Grid Lines */}
                          {[0, 25, 50, 75, 100].map((val) => {
                            const y = 140 - (val * 110) / 100;
                            return (
                              <g key={val}>
                                <line 
                                  x1={35} 
                                  y1={y} 
                                  x2={475} 
                                  y2={y} 
                                  stroke="rgba(255,255,255,0.06)" 
                                  strokeWidth={1}
                                  strokeDasharray="4 4"
                                />
                                <text 
                                  x={30} 
                                  y={y + 3} 
                                  fill="#64748b" 
                                  fontSize={9} 
                                  textAnchor="end"
                                  fontFamily="monospace"
                                >
                                  {val}%
                                </text>
                              </g>
                            );
                          })}

                          {/* X Axis MM-DD Labels */}
                          {trendData.map((d, idx) => {
                            if (idx % 6 !== 0 && idx !== 29) return null;
                            const x = 35 + (idx * 440) / 29;
                            return (
                              <text
                                key={idx}
                                x={x}
                                y={158}
                                fill="#64748b"
                                fontSize={8}
                                textAnchor="middle"
                                fontFamily="monospace"
                              >
                                {d.date}
                              </text>
                            );
                          })}

                          {/* Line and Gradient Path */}
                          {trendData.length > 1 && (() => {
                            let pathD = "";
                            const maxVal = Math.max(...trendData.map(d => d.value), 2.0);
                            trendData.forEach((d, idx) => {
                              const x = 35 + (idx * 440) / 29;
                              const y = 140 - (d.value * 110) / maxVal;
                              pathD += `${idx === 0 ? 'M' : 'L'} ${x} ${y} `;
                            });
                            
                            const xStart = 35;
                            const xEnd = 475;
                            const areaPathD = `${pathD} L ${xEnd} 140 L ${xStart} 140 Z`;
                            
                            return (
                              <g>
                                <path
                                  d={areaPathD}
                                  fill="url(#trend-kpi-area-grad)"
                                  opacity={0.15}
                                />
                                <path
                                  d={pathD}
                                  fill="none"
                                  stroke="#22d3ee"
                                  strokeWidth={2}
                                />
                              </g>
                            );
                          })()}

                          <defs>
                            <linearGradient id="trend-kpi-area-grad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#22d3ee" />
                              <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
                            </linearGradient>
                          </defs>
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })() : (() => {
              const filteredHistory = history.filter((rec) => {
                // review status filter
                if (filterReviewStatus !== 'all') {
                  if (filterReviewStatus === 'pending') {
                    if (rec.result.defects && rec.result.defects.length > 0) {
                      if (rec.review_status !== 'pending') return false;
                    } else {
                      return false;
                    }
                  } else if (filterReviewStatus === 'reviewed') {
                    if (rec.review_status === 'pending') return false;
                  }
                }
                
                // defect type filter
                if (filterDefectType !== 'all') {
                  if (!rec.result.defects || !rec.result.defects.some(d => d.type === filterDefectType)) {
                    return false;
                  }
                }
                
                return true;
              });

              return (
                <div className="flex-grow flex flex-col min-h-0 overflow-hidden">
                  {/* Filter controls */}
                  <div className="flex flex-wrap gap-3 mb-3 bg-slate-50 p-2.5 rounded-lg border border-slate-200 text-xs items-center shrink-0">
                    <span className="font-bold text-slate-700">条件筛选:</span>
                    <div className="flex items-center gap-1.5">
                      <span className="text-slate-500">审核状态:</span>
                      <select
                        value={filterReviewStatus}
                        onChange={(e) => setFilterReviewStatus(e.target.value)}
                        className="bg-white border border-slate-300 rounded px-2 py-1 focus:outline-none focus:border-blue-500"
                      >
                        <option value="all">全部记录</option>
                        <option value="pending">待人工复核 (Bad Case)</option>
                        <option value="reviewed">已专家复核</option>
                      </select>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-slate-500">缺陷种类:</span>
                      <select
                        value={filterDefectType}
                        onChange={(e) => setFilterDefectType(e.target.value)}
                        className="bg-white border border-slate-300 rounded px-2 py-1 focus:outline-none focus:border-blue-500"
                      >
                        <option value="all">全部缺陷类型</option>
                        <option value="Scratches">划痕 (Scratches)</option>
                        <option value="Cracks">裂纹 (Cracks)</option>
                        <option value="Pitting">麻点 (Pitting)</option>
                        <option value="Inclusions">夹杂 (Inclusions)</option>
                        <option value="Scale">氧化皮 (Scale)</option>
                        <option value="Patches">斑块 (Patches)</option>
                      </select>
                    </div>
                    
                    <span className="ml-auto text-[11px] font-mono text-slate-500">
                      筛选出 <b className="text-blue-600 font-bold">{filteredHistory.length}</b> 条
                    </span>
                  </div>

                  <div className="flex-1 overflow-y-auto space-y-2.5 pr-1">
                    {filteredHistory.length > 0 ? (
                      filteredHistory.map((rec) => {
                        const hasDefects = rec.result.defects && rec.result.defects.length > 0;
                        const isPendingReview = rec.review_status === 'pending' && hasDefects;
                        return (
                          <div
                            key={rec.id}
                            onClick={() => handleRecallHistoryRecord(rec)}
                            className="p-3.5 bg-slate-50 border border-slate-200 hover:border-blue-400 hover:bg-white rounded-xl cursor-pointer transition flex flex-col md:flex-row justify-between items-start md:items-center gap-3 relative group"
                          >
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${hasDefects ? 'bg-amber-50 text-amber-700 border border-amber-200' : 'bg-emerald-50 text-emerald-705 border border-emerald-200'}`}>
                                  {rec.result.overallStatus === 'Pass' ? '合格' : rec.result.overallStatus === 'Marginal' ? '降级' : '不合格'}
                                </span>
                                {isPendingReview && (
                                  <span className="text-[9px] font-mono px-1 py-0.5 bg-rose-50 border border-rose-250 text-rose-700 rounded font-bold">
                                    待复核
                                  </span>
                                )}
                                <h4 className="text-xs font-bold text-slate-800 tracking-wide">{rec.imageName}</h4>
                              </div>
                              
                              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-500 font-mono">
                                <span>流水号: {rec.id}</span>
                                <span>时间: {rec.timestamp}</span>
                                {hasDefects && (
                                  <span className="text-amber-600 font-bold">检测缺陷 {rec.result.defects.length} 处</span>
                                )}
                              </div>
                            </div>

                            <div className="flex items-center gap-3 shrink-0 self-end md:self-center">
                              <span className="text-xs text-slate-600 font-mono font-bold bg-white px-2.5 py-1 rounded border border-slate-200">
                                指数: {rec.result.severityIndex}
                              </span>
                              
                              <button
                                onClick={(e) => handleDeleteHistoryItem(rec.id, e)}
                                className="p-1.5 bg-red-50 hover:bg-red-100 rounded border border-red-200 text-slate-500 hover:text-red-600 transition"
                                title="删除此存档"
                              >
                                <Trash2 className="h-4 w-4" />
                              </button>
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="py-12 text-center text-xs text-slate-400">
                        <Database className="h-8 w-8 text-slate-300 mx-auto mb-2" />
                        <span>无匹配的检测记录</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })()}

            <div className="pt-4 border-t border-slate-100 mt-4 flex justify-end">
              <button
                onClick={() => setShowHistoryModal(false)}
                className="py-1.5 px-4 bg-slate-800 hover:bg-slate-700 text-white rounded text-xs font-semibold transition"
              >
                关闭视窗
              </button>
            </div>
          </div>
        </div>
      )}

      {/* FLOATING AI ASSISTANT CHAT CONSOLE */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
        <AnimatePresence>
          {isAssistantOpen && (
            <motion.div
              initial={{ opacity: 0, y: 30, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 30, scale: 0.95 }}
              transition={{ duration: 0.2 }}
              className="w-80 h-[380px] bg-slate-900/95 border border-slate-700 rounded-2xl shadow-2xl backdrop-blur-md flex flex-col overflow-hidden mb-3 text-white"
            >
              {/* Header */}
              <div className="p-3 bg-slate-950/80 border-b border-slate-800 flex justify-between items-center shrink-0">
                <div className="flex items-center gap-1.5">
                  <Sparkles className="h-4 w-4 text-cyan-400" />
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-200">AI 冶金工艺助理</span>
                </div>
                <button
                  onClick={() => setIsAssistantOpen(false)}
                  className="text-slate-400 hover:text-white p-0.5 rounded-full hover:bg-slate-800"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-3 space-y-2.5 text-[11px] select-text">
                {chatMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-xl px-3 py-2 leading-relaxed whitespace-pre-wrap ${
                        msg.sender === 'user'
                          ? 'bg-blue-600 text-white rounded-br-none'
                          : 'bg-slate-800 text-slate-200 rounded-bl-none border border-slate-700/50'
                      }`}
                    >
                      {msg.text}
                    </div>
                  </div>
                ))}
                {isSendingChat && (
                  <div className="flex justify-start">
                    <div className="bg-slate-800 text-slate-400 rounded-xl rounded-bl-none border border-slate-700/50 px-3 py-2 flex items-center gap-1">
                      <RefreshCw className="h-3 w-3 animate-spin" />
                      <span>正在分析机理...</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Chat Input */}
              <form onSubmit={handleSendChat} className="p-2.5 bg-slate-950/60 border-t border-slate-800/80 flex gap-2 shrink-0">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="咨询问题..."
                  className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
                />
                <button
                  type="submit"
                  disabled={isSendingChat || !chatInput.trim()}
                  className="p-1.5 bg-blue-600 hover:bg-blue-550 text-white rounded-lg transition shrink-0 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Send className="h-3.5 w-3.5" />
                </button>
              </form>
            </motion.div>
          )}
        </AnimatePresence>

        <button
          onClick={() => setIsAssistantOpen(!isAssistantOpen)}
          className="w-12 h-12 rounded-full bg-blue-600 hover:bg-blue-550 text-white flex items-center justify-center cursor-pointer shadow-lg hover:scale-105 active:scale-95 transition-all"
          title="呼叫 AI 冶金助理"
        >
          <MessageSquare className="h-5 w-5" />
        </button>
      </div>

      {/* FOOTER METADATA telemetries */}
      <footer className="bg-white border-t border-slate-200 py-3 px-6 flex justify-between items-center text-[10px] text-slate-400 font-mono mt-auto select-none shrink-0 shadow-inner">
        <span>© SteelEye Surface Terminal • V4.2.0-Pro</span>
        <span>CCD: Active • Laser: {laserSpeed}00Hz • Contrast: {contrastThreshold}%</span>
      </footer>

      {/* CUSTOM STYLE KEYFRAMES DECLARATIONS for procedural laser sweep */}
      <style>{`
        @keyframes laser-sweep {
          0% {
            top: 0%;
            opacity: 0.8;
          }
          50% {
            opacity: 1;
            box-shadow: 0 0 20px #22d3ee, 0 0 10px #06b6d4;
          }
          100% {
            top: 100%;
            opacity: 0.8;
          }
        }
      `}</style>
    </div>
  );
}

<template>
  <div class="relative w-full bg-industrial-bg border border-industrial-border rounded-xl p-6 overflow-hidden select-none shadow-inner">
    <!-- Component Header -->
    <div class="flex justify-between items-center mb-4">
      <div class="flex items-center gap-2">
        <span class="relative flex h-3 w-3">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
          <span class="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
        </span>
        <h3 class="text-sm font-semibold tracking-wider text-slate-300 uppercase">
          数字孪生虚拟生产线 (Conveyor Belt Twin)
        </h3>
      </div>
      <div class="text-xs text-slate-400 flex items-center gap-3">
        <span>带钢运行速度: <strong class="text-blue-400">10.0 m/s</strong></span>
        <span>线速度: <strong class="text-green-400">30.0 FPS</strong></span>
      </div>
    </div>

    <!-- The Conveyor Belt Container -->
    <div class="relative h-48 w-full bg-slate-950/70 border-y-4 border-slate-700 rounded-lg flex items-center overflow-hidden">
      <!-- Background Grid lines sliding -->
      <div class="absolute inset-0 bg-[linear-gradient(90deg,rgba(255,255,255,0.03)_1px,transparent_1px)] bg-[size:60px_100%] animate-conveyor-slide"></div>

      <!-- Gear/Rollers on the left -->
      <div class="absolute left-4 z-10 w-16 h-16 bg-slate-800 border-4 border-slate-600 rounded-full flex items-center justify-center animate-spin-slow shadow-lg">
        <svg viewBox="0 0 100 100" class="w-10 h-10 fill-slate-500">
          <circle cx="50" cy="50" r="15" fill="#475569" />
          <path d="M50 0 L55 25 L45 25 Z" />
          <path d="M50 100 L55 75 L45 75 Z" />
          <path d="M0 50 L25 55 L25 45 Z" />
          <path d="M100 50 L75 55 L75 45 Z" />
          <path d="M15 15 L32 32 L25 39 Z" />
          <path d="M85 85 L68 68 L75 61 Z" />
          <path d="M15 85 L32 68 L25 61 Z" />
          <path d="M85 15 L68 32 L75 39 Z" />
        </svg>
      </div>

      <!-- Gear/Rollers on the right -->
      <div class="absolute right-4 z-10 w-16 h-16 bg-slate-800 border-4 border-slate-600 rounded-full flex items-center justify-center animate-spin-slow shadow-lg">
        <svg viewBox="0 0 100 100" class="w-10 h-10 fill-slate-500">
          <circle cx="50" cy="50" r="15" fill="#475569" />
          <path d="M50 0 L55 25 L45 25 Z" />
          <path d="M50 100 L55 75 L45 75 Z" />
          <path d="M0 50 L25 55 L25 45 Z" />
          <path d="M100 50 L75 55 L75 45 Z" />
          <path d="M15 15 L32 32 L25 39 Z" />
          <path d="M85 85 L68 68 L75 61 Z" />
          <path d="M15 85 L32 68 L25 61 Z" />
          <path d="M85 15 L68 32 L75 39 Z" />
        </svg>
      </div>

      <!-- Moving Steel Sheets -->
      <div 
        v-for="item in conveyorItems" 
        :key="item.id"
        class="absolute top-1/2 -translate-y-1/2 w-48 h-32 rounded-lg cursor-pointer transition-all duration-75 border-2 flex flex-col p-3 text-xs justify-between group"
        :class="{
          'bg-emerald-950/70 border-emerald-500/80 shadow-glow-green text-emerald-100 hover:bg-emerald-900/80': item.status === 'pass',
          'bg-rose-950/80 border-rose-500 shadow-glow-red text-rose-100 hover:bg-rose-900/90 animate-pulse-fast': item.status === 'defect',
          'bg-amber-950/70 border-amber-500/80 shadow-glow-blue text-amber-100 hover:bg-amber-900/80': item.status === 'processing'
        }"
        :style="{ left: `${item.position}%` }"
        @click="selectSheet(item)"
      >
        <div class="flex justify-between items-center font-mono">
          <span class="bg-slate-900/60 px-1.5 py-0.5 rounded border border-slate-700/50">
            🆔 {{ item.id }}
          </span>
          <span class="text-[10px] opacity-80">{{ item.time }}</span>
        </div>

        <div class="my-1.5 flex-1 flex flex-col justify-center">
          <div class="font-semibold text-center tracking-wide" :class="item.status === 'defect' ? 'text-rose-400' : 'text-slate-300'">
            {{ item.status === 'pass' ? '合格 PASS' : item.status === 'defect' ? '🚨 检测异常' : '⚙️ 专家会诊中' }}
          </div>
          <div class="text-[10px] text-slate-400 mt-1 truncate max-w-full text-center">
            {{ item.details }}
          </div>
        </div>

        <div class="flex justify-between items-center text-[9px] text-slate-500">
          <span>点击查看</span>
          <span class="opacity-0 group-hover:opacity-100 transition-opacity text-blue-400">INFO</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, computed } from 'vue';
import { useDefectStore } from '../store/defect';
import type { ConveyorItem } from '../store/defect';

const store = useDefectStore();
const conveyorItems = computed(() => store.conveyorItems);

// Smooth ticking frame update for moving sheets
let animationId = 0;
const updatePositions = () => {
  conveyorItems.value.forEach(item => {
    // 0.2% movement per frame (~12% per second)
    if (item.position < 82) {
      item.position += 0.15;
    } else {
      // Reached near the end, hold position or slow down to exit
      item.position += 0.05;
    }
  });

  animationId = requestAnimationFrame(updatePositions);
};

const selectSheet = (item: ConveyorItem) => {
  if (item.recordId) {
    const record = store.records.find(r => r.id === item.recordId);
    if (record) {
      store.selectedRecord = record;
    }
  }
};

onMounted(() => {
  animationId = requestAnimationFrame(updatePositions);
});

onUnmounted(() => {
  cancelAnimationFrame(animationId);
});
</script>

<style scoped>
@keyframes conveyor-slide {
  from { background-position-x: 0px; }
  to { background-position-x: 60px; }
}

.animate-conveyor-slide {
  animation: conveyor-slide 2s infinite linear;
}

@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.animate-spin-slow {
  animation: spin-slow 8s infinite linear;
}

@keyframes pulse-fast {
  0%, 100% { border-color: rgba(239, 68, 68, 1); box-shadow: 0 0 15px rgba(239, 68, 68, 0.5); }
  50% { border-color: rgba(239, 68, 68, 0.4); box-shadow: 0 0 5px rgba(239, 68, 68, 0.1); }
}

.animate-pulse-fast {
  animation: pulse-fast 1s infinite ease-in-out;
}
</style>

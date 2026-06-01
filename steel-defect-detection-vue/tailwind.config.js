/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        industrial: {
          bg: '#090d16',        // 极深工业背景色
          card: '#131924',      // 卡片与边栏背景
          border: '#1f293d',    // 工业风边界细线
          accent: '#2b6cb0',    // 精英蓝
          success: '#10b981',   // 运行正常绿
          warning: '#f59e0b',   // 缺陷预警黄
          danger: '#ef4444',    // 严重报警红
          process: '#ff7a00',   // 大模型会诊橙
        }
      },
      fontFamily: {
        sans: ['Outfit', 'Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'glow-blue': '0 0 15px rgba(59, 130, 246, 0.25)',
        'glow-red': '0 0 15px rgba(239, 68, 68, 0.35)',
        'glow-green': '0 0 15px rgba(16, 185, 129, 0.25)',
      }
    },
  },
  plugins: [],
}

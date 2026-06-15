/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { DefectSample } from '../types';

export const SAMPLE_PRESETS: DefectSample[] = [
  {
    id: 'clean_roll_01',
    name: 'Standard Cold-Rolled Plate',
    chineseName: '标准冷轧合格板材',
    type: 'None',
    description: '表面光洁平整，无可见宏观材质缺陷，粗糙度均匀符合一类等级标准。',
    renderType: 'clean',
  },
  {
    id: 'scratch_roll_02',
    name: 'Slight Scratch Strain',
    chineseName: '冷轧辊面压入撕裂划痕',
    type: 'Scratches',
    description: '在轧制过程中，由于辊面有硬质颗粒夹杂，摩擦导致表面形成数条宏观贯穿性纵向拉深划痕。',
    renderType: 'scratch',
  },
  {
    id: 'crack_roll_03',
    name: 'Thermal Stress Edge Fissure',
    chineseName: '板材边部热应力晶间裂纹',
    type: 'Cracks',
    description: '连铸降温过快或轧制温度不均，导致应力集中在板材边缘，产生树枝状不规则撕裂裂解缝隙。',
    renderType: 'crack',
  },
  {
    id: 'pitting_roll_04',
    name: 'Acid Wash Pitting Corrosion',
    chineseName: '酸洗过度孔洞性点蚀麻面',
    type: 'Pitting',
    description: '板坯酸洗过久导致局部产生点状晶间腐蚀酸坑，钢卷表面呈粗糙密集斑点孔穴。',
    renderType: 'pitting',
  },
  {
    id: 'inclusion_roll_05',
    name: 'Non-Metallic Slag Occlusion',
    chineseName: '连铸结晶器非金属夹杂物',
    type: 'Inclusions',
    description: '炼钢钢水中未脱氧彻底的Al2O3或保护渣颗粒混入铸坯表面，轧制后展开呈淡黄褐色的撕裂夹条。',
    renderType: 'scale', // we will handle inclusion under 'inclusion' render
  },
  {
    id: 'scale_roll_06',
    name: 'Primary Recrystallized Scale',
    chineseName: '高温热连轧铁素体残留氧化皮',
    type: 'Scale',
    description: '高温高压下，板材表面保护气不足或粗轧除鳞不净，冷却时产生暗黑色氧化铁（Fe3O4）片状鳞层。',
    renderType: 'scale',
  },
];

/**
 * Draw procedurally styled metal steel plate texture onto an HTML Canvas.
 */
export function drawSteelPlate(
  canvas: HTMLCanvasElement,
  type: 'clean' | 'scratch' | 'crack' | 'pitting' | 'inclusion' | 'scale'
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const width = canvas.width;
  const height = canvas.height;

  // 1. Draw Base Steel Metal Color (Brushed gray metal gradient)
  const grad = ctx.createLinearGradient(0, 0, width, height);
  if (type === 'scale') {
    grad.addColorStop(0, '#535b63'); // Darker oxide scale metal base
    grad.addColorStop(0.5, '#40464d');
    grad.addColorStop(1, '#32373c');
  } else {
    grad.addColorStop(0, '#dedffe'); // Classic light cold-rolled iron-metal
    grad.addColorStop(0.4, '#c1c3cf');
    grad.addColorStop(0.7, '#d3d5e2');
    grad.addColorStop(1, '#a8abbc');
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, width, height);

  // 2. Draw brushed metal linear micro-teeth grain textures
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  const count = type === 'scale' ? 250 : 150;
  for (let i = 0; i < count; i++) {
    const y = Math.random() * height;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y + (Math.random() * 6 - 3));
    ctx.stroke();
  }

  // Draw vertical roller ripple lines (subtle mill rolling stripes)
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.03)';
  for (let x = 30; x < width; x += 60 + Math.random() * 40) {
    ctx.beginPath();
    ctx.lineWidth = Math.random() * 4 + 1;
    ctx.moveTo(x, 0);
    ctx.lineTo(x + (Math.random() * 10 - 5), height);
    ctx.stroke();
  }

  // 3. Render specific procedural defect annotations
  switch (type) {
    case 'clean':
      // Very clean, just add subtle metallic specular sheen
      const sheen = ctx.createLinearGradient(100, 0, 200, height);
      sheen.addColorStop(0, 'rgba(255,255,255,0)');
      sheen.addColorStop(0.5, 'rgba(255,255,255,0.18)');
      sheen.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = sheen;
      ctx.fillRect(0, 0, width, height);
      break;

    case 'scratch':
      // Draw 2 major vertical or horizontal razor deep gouging metallic scratches
      // Scratch 1
      ctx.shadowColor = '#000000';
      ctx.shadowBlur = 1;
      
      ctx.strokeStyle = '#2d2e30'; // Dark groove core
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(120, 80);
      ctx.bezierCurveTo(220, 95, 320, 75, 480, 110);
      ctx.stroke();

      // Highlight bevel to make it look 3D engraved
      ctx.shadowBlur = 0;
      ctx.strokeStyle = '#ffffff'; // White scratch reflection edge
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(120, 81.5);
      ctx.bezierCurveTo(220, 96.5, 320, 76.5, 480, 111.5);
      ctx.stroke();

      // Scratch 2 (Smaller, diagonal)
      ctx.strokeStyle = '#383a3d';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(300, 220);
      ctx.lineTo(550, 290);
      ctx.stroke();

      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(300, 221);
      ctx.lineTo(550, 291);
      ctx.stroke();

      // Additional hairline tear scratches
      ctx.strokeStyle = '#4b4d52';
      ctx.lineWidth = 1;
      for (let s = 0; s < 3; s++) {
        const offset = s * 25;
        ctx.beginPath();
        ctx.moveTo(250, 150 + offset);
        ctx.lineTo(390, 170 + offset);
        ctx.stroke();
      }
      break;

    case 'crack':
      // Draw Tree-like fractal hot-rolling edge tears on the right and center
      ctx.strokeStyle = '#18120e'; // deep black-rust fissure
      ctx.lineWidth = 3;
      ctx.lineJoin = 'miter';

      // Grand fissure
      ctx.beginPath();
      ctx.moveTo(560, 150); // coming from right edge
      ctx.lineTo(440, 170);
      ctx.lineTo(380, 140);
      ctx.lineTo(290, 185);
      ctx.lineTo(210, 160);
      ctx.stroke();

      // Side branches
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(440, 170);
      ctx.lineTo(410, 220);
      ctx.lineTo(330, 240);
      ctx.moveTo(380, 140);
      ctx.lineTo(350, 100);
      ctx.lineTo(280, 85);
      ctx.stroke();

      // Highlight halo crack edges represent thermal distress discoloration
      ctx.strokeStyle = 'rgba(240, 120, 40, 0.25)'; // slight high-temp orange singe
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(560, 150);
      ctx.lineTo(440, 170);
      ctx.lineTo(380, 140);
      ctx.lineTo(290, 185);
      ctx.stroke();
      break;

    case 'pitting':
      // Rusted Acid Pits (crater dots)
      for (let j = 0; j < 45; j++) {
        const px = 100 + Math.random() * 400;
        const py = 60 + Math.random() * 280;
        const radius = 1.5 + Math.random() * 5;

        // Pit shadow/hollow
        ctx.fillStyle = '#1c1510'; // deep shadow
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();

        // Pit rusty oxidation halo glow
        ctx.strokeStyle = 'rgba(154, 74, 30, 0.45)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(px + 0.5, py + 0.5, radius + 1, 0, Math.PI * 2);
        ctx.stroke();

        // Metallic speck highlight
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(px - radius/3, py - radius/3, radius/4, 0, Math.PI * 2);
        ctx.fill();
      }
      break;

    case 'scale':
      // Scale / Inclusions
      // Scale manifests as broad flake-shaped scaly textures, charcoal colored patches.
      ctx.fillStyle = 'rgba(30, 32, 35, 0.72)'; // Dark fe3o4 scale sheet
      ctx.strokeStyle = 'rgba(10, 12, 15, 0.9)';
      ctx.lineWidth = 1.5;

      // Piece 1
      ctx.beginPath();
      ctx.moveTo(200, 100);
      ctx.lineTo(280, 80);
      ctx.lineTo(340, 140);
      ctx.lineTo(250, 180);
      ctx.lineTo(180, 130);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Inner scaling cracks of piece 1
      ctx.strokeStyle = '#1b1d20';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(220, 120);
      ctx.lineTo(270, 130);
      ctx.lineTo(310, 110);
      ctx.stroke();

      // Piece 2
      ctx.fillStyle = 'rgba(20, 22, 24, 0.65)';
      ctx.beginPath();
      ctx.moveTo(380, 180);
      ctx.lineTo(490, 175);
      ctx.lineTo(440, 260);
      ctx.lineTo(350, 240);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Slag Inclusion spots (brownish yellowish lines associated with inclusion)
      ctx.fillStyle = '#8f6534'; // Slag brownish yellow oxide
      ctx.strokeStyle = '#543b1a';
      ctx.lineWidth = 1;
      for (let inc = 0; inc < 4; inc++) {
        const ix = 80 + inc * 110;
        const iy = 260 + Math.random() * 50;
        ctx.beginPath();
        ctx.ellipse(ix, iy, 12, 3, Math.PI / 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      break;
  }
}

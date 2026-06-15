/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type DefectType = 'Scratches' | 'Cracks' | 'Pitting' | 'Inclusions' | 'Scale' | 'Patches' | 'None';

export interface DefectItem {
  id: string;
  type: DefectType;
  typeName: string; // Chinese technical name
  description: string; // Chinese explanation
  severity: 'Low' | 'Medium' | 'High';
  bbox: [number, number, number, number]; // [xmin, ymin, xmax, ymax] in 0-100 scale
  confidence: number;
}

export interface DetectionResult {
  id?: string;
  overallStatus: 'Pass' | 'Fail' | 'Marginal';
  severityIndex: number; // 0 to 100
  defectDensity: number; // Defective ratio %
  defects: DefectItem[];
  chemicalExplanation: string; // Metallurgical root cause
  recommendedAction: string; // Action recommendations
  isSimulated?: boolean;
  simulatedReason?: string;
  engine?: string;
}

export interface InspectionRecord {
  id: string;
  timestamp: string;
  imageName: string;
  imageUrl: string; 
  result: DetectionResult;
  review_status?: string;
  reviewer?: string;
  note?: string;
}

export interface DefectSample {
  id: string;
  name: string;
  chineseName: string;
  type: DefectType;
  description: string;
  renderType: 'clean' | 'scratch' | 'crack' | 'pitting' | 'inclusion' | 'scale';
}

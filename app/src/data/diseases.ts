import type { ClassName } from '../ml/contract';

export const LOW_CONFIDENCE_THRESHOLD = 0.6;

export interface DiseaseInfo {
  id: ClassName;
  name: string;
  causalAgent: string;
  hallmarks: string[];
  management: string[];
}

export const DISEASES: Record<ClassName, DiseaseInfo> = {
  common_rust: {
    id: 'common_rust',
    name: 'Common Rust',
    causalAgent: 'Puccinia sorghi (fungus)',
    hallmarks: [
      'Dusty reddish-brown pustules on both leaf surfaces',
      'Pustules rupture to release rust-colored spores',
      'Usually appears after tasseling in wet weather',
    ],
    management: [
      'Grow resistant hybrids',
      'Rotate away from maize for a season',
      'Fungicide only if severe before tasseling',
    ],
  },
  gray_leaf_spot: {
    id: 'gray_leaf_spot',
    name: 'Gray Leaf Spot',
    causalAgent: 'Cercospora zeina / Cercospora zeae-maydis (fungus)',
    hallmarks: [
      'Rectangular tan-to-gray lesions running parallel to leaf veins',
      'Lesions narrow and bounded by veins',
      'Leaves look blighted where lesions coalesce',
    ],
    management: [
      'Use resistant varieties',
      'Remove or bury maize residue after harvest',
      'Strobilurin or triazole fungicide if disease starts before tasseling',
    ],
  },
  northern_leaf_blight: {
    id: 'northern_leaf_blight',
    name: 'Northern Leaf Blight',
    causalAgent: 'Exserohilum turcicum (fungus)',
    hallmarks: [
      'Large cigar-shaped grey-green to tan lesions',
      'Lesions 2.5-30 cm long, tapering at both ends',
      'Starts on lower leaves and spreads upward',
    ],
    management: [
      'Plant resistant hybrids',
      'Crop rotation with non-cereal crops',
      'Fungicide at tasseling if upper leaves are infected',
    ],
  },
  healthy: {
    id: 'healthy',
    name: 'Healthy Maize Leaf',
    causalAgent: 'None detected',
    hallmarks: ['Uniform green color', 'No lesions, pustules or discoloration'],
    management: ['No action needed', 'Keep scouting; recheck after wet weather'],
  },
};

export function getDiseaseInfo(className: ClassName): DiseaseInfo {
  return DISEASES[className];
}

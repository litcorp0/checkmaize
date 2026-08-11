export const CONTRACT = {
  size: 224,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  inputName: 'input',
  outputName: 'output',
  inputShape: [1, 3, 224, 224],
} as const;

export type ClassName = 'common_rust' | 'gray_leaf_spot' | 'northern_leaf_blight' | 'healthy';

export const CLASS_ORDER: ClassName[] = ['common_rust', 'gray_leaf_spot', 'northern_leaf_blight', 'healthy'];

export interface Prediction {
  className: ClassName;
  confidence: number;
}

import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { MaizeClassifier } from './onnx';
import { preprocessImage } from './preprocess';
import type { Prediction } from './contract';

const modelModule = require('../../assets/model/model.onnx') as number;

export interface ModelMetrics {
  model: string;
  test_accuracy: number;
  macro_f1: number;
  onnx_bytes: number;
  int8_test_accuracy: number;
  shipped: string;
}

interface ModelContextValue {
  ready: boolean;
  classify: (uri: string) => Promise<Prediction[]>;
  metrics: ModelMetrics | null;
}

const ModelContext = createContext<ModelContextValue | null>(null);

export function ModelProvider({ children }: { children: React.ReactNode }) {
  const [classifier, setClassifier] = useState<MaizeClassifier | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [classifierInstance, metricsData] = await Promise.all([
          MaizeClassifier.create(modelModule),
          require('../../assets/model/metrics.json') as ModelMetrics,
        ]);
        if (!cancelled) {
          setClassifier(classifierInstance);
          setMetrics(metricsData);
          setReady(true);
        }
      } catch (error) {
        console.error('model load failed', error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo<ModelContextValue>(
    () => ({
      ready,
      classify: async (uri: string) => {
        if (!classifier) {
          throw new Error('classifier not ready');
        }
        const tensor = await preprocessImage(uri);
        return classifier.classify(tensor);
      },
      metrics,
    }),
    [ready, classifier, metrics]
  );

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export function useClassifier(): ModelContextValue {
  const value = useContext(ModelContext);
  if (!value) {
    throw new Error('useClassifier must be used within ModelProvider');
  }
  return value;
}

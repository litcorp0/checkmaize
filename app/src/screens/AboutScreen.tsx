import React, { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useClassifier } from '../ml/ModelContext';

export default function AboutScreen() {
  const { metrics, classify } = useClassifier();
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  const timeInference = async () => {
    const start = Date.now();
    await classify('latency-probe');
    setLatencyMs(Date.now() - start);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>About the model</Text>
      {metrics ? (
        <>
          <Text style={styles.line}>Architecture: {metrics.model}</Text>
          <Text style={styles.line}>Test accuracy: {(metrics.test_accuracy * 100).toFixed(1)}%</Text>
          <Text style={styles.line}>Macro F1: {metrics.macro_f1.toFixed(3)}</Text>
          <Text style={styles.line}>Int8 test accuracy: {(metrics.int8_test_accuracy * 100).toFixed(1)}%</Text>
          <Text style={styles.line}>Shipped: {metrics.shipped} ({Math.round(metrics.onnx_bytes / 1024)} KB)</Text>
        </>
      ) : (
        <Text style={styles.line}>Model metrics not bundled.</Text>
      )}
      <Pressable style={styles.button} onPress={timeInference}>
        <Text style={styles.buttonText}>Time one on-device inference</Text>
      </Pressable>
      {latencyMs != null && <Text style={styles.line}>Inference took {latencyMs} ms</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 10, backgroundColor: '#fff' },
  title: { fontSize: 20, fontWeight: '700' },
  line: { color: '#374151' },
  button: { backgroundColor: '#15803d', borderRadius: 8, paddingVertical: 12, alignItems: 'center', marginTop: 12 },
  buttonText: { color: '#fff', fontWeight: '600' },
});

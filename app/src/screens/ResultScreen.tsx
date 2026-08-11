import React from 'react';
import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { getDiseaseInfo } from '../data/diseases';
import type { RootStackParamList } from '../../App';

type Props = NativeStackScreenProps<RootStackParamList, 'Result'>;

export default function ResultScreen({ navigation, route }: Props) {
  const { imageUri, prediction, confidence } = route.params;
  const info = getDiseaseInfo(prediction);
  const percent = Math.round(confidence * 100);

  return (
    <View style={styles.container}>
      <Image source={{ uri: imageUri }} style={styles.image} />
      <Text style={styles.name}>{info.name}</Text>
      <View style={styles.bar}>
        <View style={[styles.barFill, { width: `${percent}%` }]} />
      </View>
      <Text style={styles.confidence}>{percent}% confidence</Text>
      <Text style={styles.hallmarks}>{info.hallmarks[0]}</Text>
      <Pressable style={styles.button} onPress={() => navigation.navigate('Info', { className: prediction })}>
        <Text style={styles.buttonText}>What to do about it</Text>
      </Pressable>
      <Pressable style={[styles.button, styles.secondary]} onPress={() => navigation.goBack()}>
        <Text style={styles.buttonText}>Scan another leaf</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', padding: 24, gap: 12 },
  image: { width: 220, height: 220, borderRadius: 12 },
  name: { fontSize: 24, fontWeight: '700' },
  bar: { width: '100%', height: 10, backgroundColor: '#e5e7eb', borderRadius: 5, overflow: 'hidden' },
  barFill: { height: '100%', backgroundColor: '#15803d' },
  confidence: { color: '#4b5563' },
  hallmarks: { color: '#374151', textAlign: 'center' },
  button: { backgroundColor: '#15803d', borderRadius: 8, paddingVertical: 12, paddingHorizontal: 32 },
  secondary: { backgroundColor: '#166534' },
  buttonText: { color: '#fff', fontWeight: '600' },
});

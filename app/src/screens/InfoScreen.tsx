import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { DISEASES } from '../data/diseases';
import type { RootStackParamList } from '../../App';

type Props = NativeStackScreenProps<RootStackParamList, 'Info'>;

export default function InfoScreen({ route }: Props) {
  const info = DISEASES[route.params.className];
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.name}>{info.name}</Text>
      <Text style={styles.agent}>Cause: {info.causalAgent}</Text>
      <Text style={styles.section}>How to recognise it</Text>
      {info.hallmarks.map((h, i) => (
        <Text key={i} style={styles.item}>{'\u2022'} {h}</Text>
      ))}
      <Text style={styles.section}>Recommended management</Text>
      {info.management.map((m, i) => (
        <Text key={i} style={styles.item}>{'\u2022'} {m}</Text>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 20, gap: 8 },
  name: { fontSize: 22, fontWeight: '700' },
  agent: { color: '#4b5563' },
  section: { fontSize: 16, fontWeight: '600', marginTop: 12 },
  item: { color: '#374151', lineHeight: 22 },
});

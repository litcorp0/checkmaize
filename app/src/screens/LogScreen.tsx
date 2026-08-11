import React, { useCallback, useState } from 'react';
import { FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { getScans, deleteScan } from '../db/scans';
import type { ScanRecord } from '../db/scans';
import { getDiseaseInfo } from '../data/diseases';

export default function LogScreen() {
  const [scans, setScans] = useState<ScanRecord[]>([]);

  useFocusEffect(
    useCallback(() => {
      getScans().then(setScans);
    }, [])
  );

  const remove = async (id: number) => {
    await deleteScan(id);
    setScans(await getScans());
  };

  return (
    <FlatList
      data={scans}
      keyExtractor={(s) => String(s.id)}
      contentContainerStyle={styles.container}
      renderItem={({ item }) => (
        <View style={styles.row}>
          <Image source={{ uri: item.imageUri }} style={styles.thumb} />
          <View style={styles.info}>
            <Text style={styles.name}>{getDiseaseInfo(item.prediction).name}</Text>
            <Text style={styles.meta}>
              {new Date(item.createdAt).toLocaleString()} - {Math.round(item.confidence * 100)}%
            </Text>
            <Text style={styles.meta}>
              {item.latitude != null ? `${item.latitude.toFixed(4)}, ${item.longitude!.toFixed(4)}` : 'no location'}
            </Text>
          </View>
          <Pressable onPress={() => remove(item.id!)}>
            <Text style={styles.delete}>Delete</Text>
          </Pressable>
        </View>
      )}
      ListEmptyComponent={<Text style={styles.empty}>No scans yet</Text>}
    />
  );
}

const styles = StyleSheet.create({
  container: { padding: 12, gap: 8 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 12, backgroundColor: '#fff', borderRadius: 8, padding: 8 },
  thumb: { width: 56, height: 56, borderRadius: 6 },
  info: { flex: 1 },
  name: { fontWeight: '600' },
  meta: { color: '#6b7280', fontSize: 12 },
  delete: { color: '#dc2626' },
  empty: { textAlign: 'center', marginTop: 40, color: '#6b7280' },
});

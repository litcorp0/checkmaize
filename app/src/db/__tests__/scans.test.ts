jest.mock('expo-sqlite', () => {
  let nextId = 0;
  const rows: any[] = [];
  const db = {
    execAsync: jest.fn(async () => {}),
    runAsync: jest.fn(async (_sql: string, ...args: any[]) => {
      nextId += 1;
      rows.push({ id: nextId, created_at: args[0], image_uri: args[1], prediction: args[2], confidence: args[3], latitude: args[4], longitude: args[5] });
      return { lastInsertRowId: nextId };
    }),
    getAllAsync: jest.fn(async () => rows),
  };
  return { openDatabaseAsync: jest.fn(async () => db) };
});

import { saveScan, getScans, deleteScan } from '../scans';

test('saved scans round-trip with geotag nulls', async () => {
  const saved = await saveScan({
    createdAt: '2026-08-04T10:00:00Z',
    imageUri: 'file:///a.jpg',
    prediction: 'northern_leaf_blight',
    confidence: 0.91,
    latitude: null,
    longitude: null,
  });
  expect(saved.id).toBeGreaterThan(0);
  const scans = await getScans();
  expect(scans).toHaveLength(1);
  expect(scans[0]).toMatchObject({
    createdAt: '2026-08-04T10:00:00Z',
    prediction: 'northern_leaf_blight',
    confidence: 0.91,
    latitude: null,
    longitude: null,
  });
});

test('newest scan comes first and delete removes it', async () => {
  await saveScan({ createdAt: '2026-08-04T09:00:00Z', imageUri: 'file:///a.jpg', prediction: 'common_rust', confidence: 0.8, latitude: 5.6, longitude: -0.2 });
  await saveScan({ createdAt: '2026-08-04T11:00:00Z', imageUri: 'file:///b.jpg', prediction: 'healthy', confidence: 0.99, latitude: null, longitude: null });
  const scans = await getScans();
  expect(scans[0].prediction).toBe('healthy');
  expect(scans[1].latitude).toBe(5.6);
  await deleteScan(scans[0].id!);
  const after = await getScans();
  expect(after).toHaveLength(1);
  expect(after[0].prediction).toBe('common_rust');
});

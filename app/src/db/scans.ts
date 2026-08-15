import * as SQLite from 'expo-sqlite';
import type { ClassName } from '../ml/contract';

export interface ScanRecord {
  id?: number;
  createdAt: string;
  imageUri: string;
  prediction: ClassName;
  confidence: number;
  latitude: number | null;
  longitude: number | null;
}

let dbPromise: Promise<SQLite.SQLiteDatabase> | null = null;

export function initDb(): Promise<SQLite.SQLiteDatabase> {
  if (!dbPromise) {
    dbPromise = SQLite.openDatabaseAsync('checkmaize.db').then(async (db) => {
      await db.execAsync(
        `CREATE TABLE IF NOT EXISTS scans (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL,
          image_uri TEXT NOT NULL,
          prediction TEXT NOT NULL,
          confidence REAL NOT NULL,
          latitude REAL,
          longitude REAL
        )`
      );
      return db;
    });
  }
  return dbPromise;
}

export async function saveScan(record: ScanRecord): Promise<ScanRecord> {
  const db = await initDb();
  const result = await db.runAsync(
    'INSERT INTO scans (created_at, image_uri, prediction, confidence, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)',
    record.createdAt,
    record.imageUri,
    record.prediction,
    record.confidence,
    record.latitude,
    record.longitude
  );
  return { ...record, id: result.lastInsertRowId };
}

export async function getScans(): Promise<ScanRecord[]> {
  const db = await initDb();
  const rows = await db.getAllAsync<{
    id: number;
    created_at: string;
    image_uri: string;
    prediction: string;
    confidence: number;
    latitude: number | null;
    longitude: number | null;
  }>('SELECT * FROM scans ORDER BY created_at DESC');
  return rows.map((r) => ({
    id: r.id,
    createdAt: r.created_at,
    imageUri: r.image_uri,
    prediction: r.prediction as ClassName,
    confidence: r.confidence,
    latitude: r.latitude,
    longitude: r.longitude,
  }));
}

export async function deleteScan(id: number): Promise<void> {
  const db = await initDb();
  await db.runAsync('DELETE FROM scans WHERE id = ?', id);
}

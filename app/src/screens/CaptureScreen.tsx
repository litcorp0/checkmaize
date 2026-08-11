import React, { useEffect, useRef, useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { CameraView } from 'expo-camera';
import * as Camera from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as Location from 'expo-location';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useClassifier } from '../ml/ModelContext';
import { saveScan } from '../db/scans';
import { LOW_CONFIDENCE_THRESHOLD } from '../data/diseases';
import type { RootStackParamList } from '../../App';

type Props = NativeStackScreenProps<RootStackParamList, 'Tabs'>;

export default function CaptureScreen({ navigation }: Props) {
  const cameraRef = useRef<CameraView>(null);
  const { ready, classify } = useClassifier();
  const [permission, setPermission] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [unclear, setUnclear] = useState(false);

  useEffect(() => {
    Camera.requestCameraPermissionsAsync().then((r) => setPermission(r.granted));
  }, []);

  const runScan = async (uri: string) => {
    if (!ready) {
      Alert.alert('Model still loading', 'Please wait a moment and try again.');
      return;
    }
    setBusy(true);
    setUnclear(false);
    try {
      const predictions = await classify(uri);
      const top = predictions[0];
      let latitude: number | null = null;
      let longitude: number | null = null;
      const perm = await Location.requestForegroundPermissionsAsync();
      if (perm.granted) {
        const pos = await Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced });
        latitude = pos.coords.latitude;
        longitude = pos.coords.longitude;
      }
      await saveScan({
        createdAt: new Date().toISOString(),
        imageUri: uri,
        prediction: top.className,
        confidence: top.confidence,
        latitude,
        longitude,
      });
      if (top.confidence >= LOW_CONFIDENCE_THRESHOLD) {
        navigation.navigate('Result', { imageUri: uri, prediction: top.className, confidence: top.confidence });
      } else {
        setUnclear(true);
      }
    } catch (error) {
      Alert.alert('Scan failed', String(error));
    } finally {
      setBusy(false);
    }
  };

  const onCapture = async () => {
    const photo = await cameraRef.current?.takePictureAsync({ quality: 1 });
    if (photo) {
      await runScan(photo.uri);
    }
  };

  const onPick = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ quality: 1 });
    if (!result.canceled) {
      await runScan(result.assets[0].uri);
    }
  };

  if (permission === null) {
    return <View style={styles.center}><Text>Requesting camera permission...</Text></View>;
  }
  if (permission === false) {
    return (
      <View style={styles.center}>
        <Text style={styles.title}>Camera access is required</Text>
        <Pressable style={styles.button} onPress={onPick}>
          <Text style={styles.buttonText}>Choose photo from gallery instead</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back" />
      <Text style={styles.tip}>Hold the camera close to a single symptomatic leaf, filling the frame.</Text>
      {unclear && (
        <View style={styles.unclearBox}>
          <Text style={styles.unclearText}>Unclear photo - re-capture with better light, or ask your extension agent.</Text>
        </View>
      )}
      <View style={styles.row}>
        <Pressable style={styles.button} onPress={onPick} disabled={busy}>
          <Text style={styles.buttonText}>Gallery</Text>
        </Pressable>
        <Pressable style={[styles.button, styles.shutter]} onPress={onCapture} disabled={busy}>
          <Text style={styles.buttonText}>{busy ? 'Scanning...' : 'Capture'}</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera: { flex: 1 },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12, padding: 24 },
  title: { fontSize: 18, fontWeight: '600' },
  tip: { color: '#fff', textAlign: 'center', padding: 8, backgroundColor: 'rgba(0,0,0,0.6)' },
  unclearBox: { backgroundColor: '#fde68a', padding: 10 },
  unclearText: { color: '#92400e', textAlign: 'center' },
  row: { flexDirection: 'row', gap: 12, padding: 16, justifyContent: 'center' },
  button: { backgroundColor: '#15803d', borderRadius: 8, paddingVertical: 12, paddingHorizontal: 24 },
  shutter: { backgroundColor: '#166534' },
  buttonText: { color: '#fff', fontWeight: '600' },
});

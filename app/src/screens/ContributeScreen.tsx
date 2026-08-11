import React, { useState } from 'react';
import { Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { File, Paths } from 'expo-file-system';
import * as Sharing from 'expo-sharing';

export default function ContributeScreen() {
  const [lastSaved, setLastSaved] = useState<string | null>(null);

  const onContribute = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ quality: 1 });
    if (result.canceled) {
      return;
    }
    const uri = result.assets[0].uri;
    const source = new File(uri);
    const bytes = await source.bytes();
    const dir = new File(Paths.document, 'contributions/');
    if (!(await dir.exists)) {
      await dir.create({ intermediates: true });
    }
    const name = `contribution_${Date.now()}.jpg`;
    const dest = new File(dir, name);
    await dest.write(bytes);
    setLastSaved(dest.uri);
    if (await Sharing.isAvailableAsync()) {
      await Sharing.shareAsync(dest.uri, {
        mimeType: 'image/jpeg',
        dialogTitle: 'Share this collection photo',
      });
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Help build the Ghana field dataset</Text>
      <Text style={styles.body}>
        Choose a leaf photo (ideally in its field background). It is saved to the app folder and you can
        share it to the researcher (WhatsApp, email, Drive). No labels are needed - the researcher labels later.
      </Text>
      {lastSaved && <Text style={styles.saved}>Saved: {lastSaved}</Text>}
      <Pressable style={styles.button} onPress={onContribute}>
        <Text style={styles.buttonText}>Pick a photo to contribute</Text>
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 24, gap: 12, backgroundColor: '#fff' },
  title: { fontSize: 20, fontWeight: '700' },
  body: { color: '#374151', lineHeight: 22 },
  saved: { color: '#15803d' },
  button: { backgroundColor: '#15803d', borderRadius: 8, paddingVertical: 12, alignItems: 'center' },
  buttonText: { color: '#fff', fontWeight: '600' },
});

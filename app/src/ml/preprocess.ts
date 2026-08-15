import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { decode as decodeJpeg } from 'jpeg-js';
import { CONTRACT } from './contract';

export function rgbaToCHWFloat32(
  rgba: Uint8Array,
  width: number,
  height: number,
  mean: readonly number[],
  std: readonly number[]
): Float32Array {
  const out = new Float32Array(3 * width * height);
  const pixels = width * height;
  for (let p = 0; p < pixels; p++) {
    for (let c = 0; c < 3; c++) {
      out[c * pixels + p] = (rgba[p * 4 + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}

export async function resizeToContract(uri: string): Promise<string> {
  const context = ImageManipulator.manipulate(uri);
  context.resize({ width: CONTRACT.size, height: CONTRACT.size });
  const rendered = await context.renderAsync();
  const result = await rendered.saveAsync({ format: SaveFormat.JPEG, compress: 1 });
  return result.uri;
}

export async function readFileBytes(uri: string): Promise<Uint8Array> {
  const { File } = await import('expo-file-system');
  return new File(uri).bytes();
}

export async function preprocessImage(uri: string): Promise<Float32Array> {
  const resized = await resizeToContract(uri);
  const bytes = await readFileBytes(resized);
  const jpeg = decodeJpeg(bytes, { useTArray: true });
  return rgbaToCHWFloat32(jpeg.data, jpeg.width, jpeg.height, CONTRACT.mean, CONTRACT.std);
}

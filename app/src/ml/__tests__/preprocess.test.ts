import fs from 'fs';
import path from 'path';
import { PNG } from 'pngjs';
import { rgbaToCHWFloat32 } from '../preprocess';
import { CONTRACT } from '../contract';

jest.mock('expo-image-manipulator', () => ({
  ImageManipulator: {
    manipulate: jest.fn(() => ({
      resize: jest.fn(),
      renderAsync: jest.fn(async () => ({
        saveAsync: jest.fn(async () => ({ uri: 'mock.jpg' })),
      })),
    })),
  },
  SaveFormat: { JPEG: 'jpeg' },
}));

const fixtures = path.join(__dirname, 'fixtures');

test('fixture reference tensor has 32x32x3 values', () => {
  const ref = JSON.parse(fs.readFileSync(path.join(fixtures, 'reference_tensor.json'), 'utf8'));
  expect(ref).toHaveLength(32 * 32 * 3);
});

test('JS normalization matches the Python reference exactly', () => {
  const png = PNG.sync.read(fs.readFileSync(path.join(fixtures, 'sample.png')));
  const rgba = new Uint8Array(png.data.buffer, png.data.byteOffset, png.data.length);
  const tensor = rgbaToCHWFloat32(rgba, png.width, png.height, CONTRACT.mean, CONTRACT.std);
  const ref = JSON.parse(fs.readFileSync(path.join(fixtures, 'reference_tensor.json'), 'utf8'));
  expect(tensor).toHaveLength(32 * 32 * 3);
  for (let i = 0; i < tensor.length; i++) {
    expect(Math.abs(tensor[i] - ref[i])).toBeLessThan(1e-5);
  }
});

test('CHW layout puts each channel in its own slab', () => {
  const rgba = new Uint8Array(4 * 2 * 2);
  rgba.set([255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255]);
  const t = rgbaToCHWFloat32(rgba, 2, 2, [0, 0, 0], [1, 1, 1]);
  expect(t[0]).toBeCloseTo(1.0, 5);
  expect(t[3]).toBeCloseTo(1.0, 5);
  expect(t[4]).toBeCloseTo(0.0, 5);
  expect(t[8]).toBeCloseTo(0.0, 5);
});

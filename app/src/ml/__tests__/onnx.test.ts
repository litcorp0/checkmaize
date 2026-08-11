import type { InferenceSession } from 'onnxruntime-react-native';
import { MaizeClassifier } from '../onnx';

jest.mock('expo-asset', () => ({
  Asset: {
    fromModule: jest.fn(() => ({
      downloadAsync: jest.fn(async () => {}),
      localUri: '/tmp/m.onnx',
      uri: '/tmp/m.onnx',
    })),
  },
}));

function fakeSession(logits: number[]) {
  return {
    run: jest.fn(async () => ({ output: { data: new Float32Array(logits) } })),
    release: jest.fn(async () => {}),
  } as unknown as InferenceSession;
}

test('classify returns softmax-sorted predictions', async () => {
  const session = fakeSession([0.1, 0.9, 0.2, 0.3]);
  const classifier = MaizeClassifier.fromSession(session);
  const tensor = new Float32Array(3 * 224 * 224);
  const preds = await classifier.classify(tensor);
  expect(session.run).toHaveBeenCalledWith({ input: expect.anything() });
  expect(preds).toHaveLength(4);
  expect(preds[0].className).toBe('gray_leaf_spot');
  expect(preds[0].confidence).toBeCloseTo(0.4008, 3);
  const confidences = preds.map((p) => p.confidence);
  expect([...confidences].sort((a, b) => b - a)).toEqual(confidences);
});

test('release delegates to the session', async () => {
  const session = fakeSession([0.25, 0.25, 0.25, 0.25]);
  const classifier = MaizeClassifier.fromSession(session);
  await classifier.release();
  expect(session.release).toHaveBeenCalled();
});

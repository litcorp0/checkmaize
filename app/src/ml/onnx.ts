import { Asset } from 'expo-asset';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import { CLASS_ORDER, CONTRACT } from './contract';
import type { ClassName, Prediction } from './contract';

export class MaizeClassifier {
  private constructor(private readonly session: InferenceSession) {}

  static async create(
    modelModule: number,
    sessionOptions?: InferenceSession.SessionOptions
  ): Promise<MaizeClassifier> {
    const asset = Asset.fromModule(modelModule);
    await asset.downloadAsync();
    const uri = (asset.localUri ?? asset.uri).replace(/^file:\/\//, '');
    const session = await InferenceSession.create(uri, {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'all',
      ...sessionOptions,
    });
    return new MaizeClassifier(session);
  }

  static fromSession(session: InferenceSession): MaizeClassifier {
    return new MaizeClassifier(session);
  }

  async classify(tensor: Float32Array): Promise<Prediction[]> {
    const input = new Tensor('float32', tensor, CONTRACT.inputShape);
    const result = await this.session.run({ [CONTRACT.inputName]: input });
    const logits = Array.from(result[CONTRACT.outputName].data as Float32Array);
    const max = Math.max(...logits);
    const exp = logits.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp
      .map((p, i) => ({ className: CLASS_ORDER[i] as ClassName, confidence: p / sum }))
      .sort((a, b) => b.confidence - a.confidence);
  }

  async release(): Promise<void> {
    await this.session.release();
  }
}

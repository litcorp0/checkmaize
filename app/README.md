# CheckMaize App

On-device maize leaf disease classifier for Ghana. Expo SDK 57 + onnxruntime-react-native.

## Model

The bundled int8 ONNX model (see `assets/model/metrics.json`) classifies
common_rust / gray_leaf_spot / northern_leaf_blight / healthy at 224x224 RGB.
The preprocessing contract is enforced by the parity test in `src/ml/__tests__/`.

## Development

- Node >= 22.13. `npm install` runs the onnxruntime autolink fix automatically.
- `npx expo run:android` (or `run:ios`) for a dev build; `npm test` for jest.

## Building installables

- Android APK: `npx eas-cli@latest build -p android --profile preview`
- iOS: `npx eas-cli@latest build -p ios --profile preview` (needs an Apple Developer account)

## Contributing field photos

Use the Contribute tab; photos are saved to the app folder and shared via the OS share sheet.

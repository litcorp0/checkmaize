const fs = require('fs');
const path = require('path');

const pkgRoot = path.join(__dirname, '..', 'node_modules', 'onnxruntime-react-native');
const unimodule = path.join(pkgRoot, 'unimodule.json');
if (fs.existsSync(unimodule)) {
  fs.rmSync(unimodule);
  console.log('fix-onnxruntime: removed unimodule.json (unblocks Expo autolinking)');
}

const gradle = path.join(pkgRoot, 'android', 'build.gradle');
if (fs.existsSync(gradle)) {
  const contents = fs.readFileSync(gradle, 'utf8');
  const block = /  if \(VersionNumber\.parse\(REACT_NATIVE_VERSION\)[\s\S]*?\n  \}\n/;
  if (block.test(contents)) {
    fs.writeFileSync(gradle, contents.replace(block, ''));
    console.log('fix-onnxruntime: removed dead VersionNumber gradle block');
  }
}

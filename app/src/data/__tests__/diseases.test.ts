import { CLASS_ORDER } from '../../ml/contract';
import { DISEASES, LOW_CONFIDENCE_THRESHOLD, getDiseaseInfo } from '../diseases';

test('every class in the contract has an info entry', () => {
  for (const cls of CLASS_ORDER) {
    expect(DISEASES[cls]).toBeDefined();
    expect(DISEASES[cls].id).toBe(cls);
  }
});

test('info entries are complete and non-empty', () => {
  for (const cls of CLASS_ORDER) {
    const info = DISEASES[cls];
    expect(info.name.length).toBeGreaterThan(0);
    expect(info.causalAgent.length).toBeGreaterThan(0);
    expect(info.hallmarks.length).toBeGreaterThan(0);
    expect(info.management.length).toBeGreaterThan(0);
  }
});

test('threshold is a probability', () => {
  expect(LOW_CONFIDENCE_THRESHOLD).toBeGreaterThan(0);
  expect(LOW_CONFIDENCE_THRESHOLD).toBeLessThan(1);
});

test('getDiseaseInfo returns the healthy entry', () => {
  expect(getDiseaseInfo('healthy').name).toBe('Healthy Maize Leaf');
});

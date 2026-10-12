"""检查混淆矩阵方向、指标计算、预测顺序与划分核对。"""
import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import calculate_metrics, predict_batches, verify_manifest


class EvaluateTests(unittest.TestCase):
    def test_metrics_and_matrix_direction_include_unpredicted_class(self):
        frame = pd.DataFrame({'true': ['a', 'a', 'b', 'b', 'c'],
                              'predicted': ['a', 'b', 'b', 'b', 'a']})
        metrics, matrix, report = calculate_metrics(frame, ['a', 'b', 'c'])
        self.assertEqual(matrix.to_numpy().tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 0]])
        self.assertAlmostEqual(metrics['accuracy'], 3 / 5)
        by_class = report.set_index('label')
        self.assertAlmostEqual(by_class.loc['a', 'precision'], 1 / 2)
        self.assertAlmostEqual(by_class.loc['a', 'recall'], 1 / 2)
        self.assertAlmostEqual(by_class.loc['b', 'f1'], 0.8)
        self.assertEqual(by_class.loc['c', 'precision'], 0)
        self.assertEqual(by_class.loc['c', 'f1'], 0)
        self.assertAlmostEqual(metrics['macro_f1'], (0.5 + 0.8) / 3)

    def test_prediction_preserves_order_and_includes_short_last_batch(self):
        logits = torch.tensor([[1., 5.], [3., 0.], [-1., 2.]])
        labels = torch.tensor([1, 0, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False)
        model = nn.Sequential(nn.Dropout(p=1), nn.Identity()).train()
        truth, predictions, scores = predict_batches(model, loader, torch.device('cpu'))
        self.assertFalse(model.training)
        self.assertEqual(truth, [1, 0, 0])
        self.assertEqual(predictions, [1, 0, 1])
        torch.testing.assert_close(torch.tensor(scores), logits.softmax(dim=1).max(dim=1).values)

    def test_manifest_hash_rejects_changed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'manifest.csv'
            path.write_bytes(b'path,label,split\na.jpg,a,test\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            verify_manifest(path, digest)
            path.write_bytes(b'path,label,split\na.jpg,a,train\n')
            with self.assertRaises(ValueError):
                verify_manifest(path, digest)

    def test_empty_records_and_nonfinite_outputs_are_rejected(self):
        with self.assertRaises(ValueError):
            calculate_metrics(pd.DataFrame(columns=['true', 'predicted']), ['a', 'b'])
        loader = DataLoader(TensorDataset(torch.tensor([[float('nan'), 0.]]), torch.tensor([0])))
        with self.assertRaises(ValueError):
            predict_batches(nn.Identity(), loader, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()

"""检查训练更新、验证不修改模型、指标加权和空数据处理。"""
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.train import run_epoch


class TrainTests(unittest.TestCase):
    def test_training_updates_parameters_and_validation_keeps_state(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2))
        images = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]])
        loader = DataLoader(TensorDataset(images, torch.tensor([0, 1, 1, 0])), batch_size=2)
        before = model[0].weight.detach().clone()
        result = run_epoch(model, loader, torch.device('cpu'), torch.optim.SGD(model.parameters(), lr=0.1))
        self.assertEqual(result['samples'], 4)
        self.assertFalse(torch.equal(before, model[0].weight))
        snapshot = {key: value.clone() for key, value in model.state_dict().items()}
        for parameter in model.parameters():
            parameter.grad = None
        run_epoch(model, loader, torch.device('cpu'))
        self.assertFalse(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(snapshot[key], value)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_metrics_weight_unequal_batches_by_sample_count(self):
        logits = torch.tensor([[5., 0.], [0., 5.], [4., 0.]])
        labels = torch.tensor([0, 1, 1])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = run_epoch(nn.Identity(), loader, torch.device('cpu'))
        expected = nn.CrossEntropyLoss()(logits, labels).item()
        self.assertAlmostEqual(result['loss'], expected, places=6)
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertEqual(result['samples'], 3)

    def test_empty_loader_is_rejected(self):
        loader = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)), batch_size=2)
        with self.assertRaises(ValueError):
            run_epoch(nn.Identity(), loader, torch.device('cpu'))

    def test_nonfinite_loss_is_rejected(self):
        loader = DataLoader(TensorDataset(torch.tensor([[float('nan'), 0.]]), torch.tensor([0])))
        with self.assertRaises(ValueError):
            run_epoch(nn.Identity(), loader, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()

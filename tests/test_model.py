"""检查残差连接、下采样维度和完整网络的参数更新。"""
import unittest

import torch
from torch import nn

from src.model import BasicBlock, ResNet18


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_identity_shortcut_preserves_input_when_main_branch_is_zero(self):
        block = BasicBlock(8, 8).eval()
        nn.init.zeros_(block.conv1.weight)
        nn.init.zeros_(block.conv2.weight)
        x = torch.randn(2, 8, 16, 16)
        # 如果漏写捷径相加，这里会得到全零，检查将失败。
        with torch.inference_mode():
            torch.testing.assert_close(block(x), torch.relu(x))

    def test_downsampling_matches_branches_and_has_finite_gradients(self):
        block = BasicBlock(8, 16, stride=2)
        x = torch.randn(2, 8, 17, 19, requires_grad=True)
        output = block(x)
        self.assertEqual(tuple(output.shape), (2, 16, 9, 10))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(block.shortcut[0].weight.grad).all())

    def test_resnet_outputs_for_different_input_sizes_and_class_counts(self):
        model = ResNet18(num_classes=7).eval()
        with torch.inference_mode():
            for height, width in [(32, 32), (96, 96), (128, 80)]:
                output = model(torch.randn(1, 3, height, width))
                self.assertEqual(tuple(output.shape), (1, 7))
                self.assertTrue(torch.isfinite(output).all())

    def test_loss_backpropagation_updates_network_parameters(self):
        model = ResNet18()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before_stem = model.stem[0].weight.detach().clone()
        before_classifier = model.classifier.weight.detach().clone()
        optimizer.zero_grad()
        output = model(torch.randn(2, 3, 64, 64))
        loss = nn.CrossEntropyLoss()(output, torch.tensor([0, 4]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.stem[0].weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before_stem, model.stem[0].weight))
        self.assertFalse(torch.equal(before_classifier, model.classifier.weight))

    def test_invalid_class_count_is_rejected(self):
        with self.assertRaises(ValueError):
            ResNet18(num_classes=0)


if __name__ == '__main__':
    unittest.main()

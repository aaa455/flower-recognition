"""检查单图预测的预处理、类别顺序、排序和异常输入。"""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from src.predict import predict_image


class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None
        self.gradient_enabled = None

    def forward(self, images):
        self.last_input = images.clone()
        self.gradient_enabled = torch.is_grad_enabled()
        return images.new_tensor([[1., 0., 3., -1., 2.]])


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.image = self.root / 'flower.png'
        Image.new('RGB', (20, 15), (255, 0, 0)).save(self.image)
        self.checkpoint = {'classes': ['tulips', 'sunflowers', 'roses', 'dandelion', 'daisy'],
                           'mean': [0.5, 0.4, 0.3], 'std': [0.2, 0.3, 0.4],
                           'config': {'image_size': 32}}
        self.device = torch.device('cpu')

    def tearDown(self):
        self.folder.cleanup()

    def test_candidates_use_checkpoint_order_and_descending_scores(self):
        model = FixedModel().train()
        results = predict_image(model, self.checkpoint, self.image, self.device)
        self.assertEqual([item['label'] for item in results], ['roses', 'daisy', 'tulips'])
        self.assertEqual([item['name'] for item in results], ['玫瑰', '雏菊', '郁金香'])
        self.assertEqual([item['rank'] for item in results], [1, 2, 3])
        expected = torch.tensor([1., 0., 3., -1., 2.]).softmax(0).topk(3).values
        torch.testing.assert_close(torch.tensor([item['score'] for item in results]), expected)
        self.assertFalse(model.training)
        self.assertFalse(model.gradient_enabled)

    def test_tensor_uses_saved_size_and_channel_normalization(self):
        model = FixedModel()
        predict_image(model, self.checkpoint, self.image, self.device)
        self.assertEqual(tuple(model.last_input.shape), (1, 3, 32, 32))
        expected = torch.tensor([(1 - 0.5) / 0.2, (0 - 0.4) / 0.3, (0 - 0.3) / 0.4])
        torch.testing.assert_close(model.last_input[0, :, 0, 0], expected)

    def test_grayscale_and_rgba_images_convert_to_rgb(self):
        for mode, color in [('L', 128), ('RGBA', (20, 40, 60, 100))]:
            path = self.root / f'{mode}.png'
            Image.new(mode, (10, 20), color).save(path)
            model = FixedModel()
            result = predict_image(model, self.checkpoint, path, self.device, top_k=5)
            self.assertEqual(tuple(model.last_input.shape), (1, 3, 32, 32))
            self.assertAlmostEqual(sum(item['score'] for item in result), 1, places=6)

    def test_invalid_top_k_missing_and_broken_images_are_rejected(self):
        for top_k in [0, 6]:
            with self.assertRaises(ValueError):
                predict_image(FixedModel(), self.checkpoint, self.image, self.device, top_k)
        with self.assertRaises(FileNotFoundError):
            predict_image(FixedModel(), self.checkpoint, self.root / 'missing.png', self.device)
        broken = self.root / 'broken.png'
        broken.write_bytes(b'not an image')
        with self.assertRaises(OSError):
            predict_image(FixedModel(), self.checkpoint, broken, self.device)


if __name__ == '__main__':
    unittest.main()

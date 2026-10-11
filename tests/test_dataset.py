"""检查图片张量形状、标签映射、归一化统计和 DataLoader。"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.dataset import FlowerDataset, calculate_statistics, create_loaders, make_transform
from src.download import CLASS_NAMES


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.root = Path(cls.folder.name) / 'images'
        rows = []
        for label_index, label in enumerate(CLASS_NAMES):
            for item_index in range(7):
                path = Path(label) / f'{item_index}.png'
                (cls.root / path).parent.mkdir(parents=True, exist_ok=True)
                color = (label_index * 40 + item_index, 30, 200 - label_index * 20)
                Image.new('RGB', (12 + item_index, 16 + item_index), color).save(cls.root / path)
                rows.append({'path': path.as_posix(), 'label': label,
                             'width': 12 + item_index, 'height': 16 + item_index,
                             'sha256': f'{label}-{item_index}',
                             'split': 'train' if item_index < 5 else ('val' if item_index == 5 else 'test')})
        cls.frame = pd.DataFrame(rows)
        cls.manifest = Path(cls.folder.name) / 'manifest.csv'
        cls.frame.to_csv(cls.manifest, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_statistics_use_only_train_images(self):
        mean, std = calculate_statistics(self.root, self.frame, image_size=8)
        self.assertEqual(len(mean), 3)
        self.assertEqual(len(std), 3)
        self.assertTrue(all(value > 0 for value in std))
        changed = self.frame.copy()
        changed.loc[changed['split'] != 'train', 'path'] = 'does/not/exist.png'
        other_mean, other_std = calculate_statistics(self.root, changed, image_size=8)
        self.assertEqual(mean, other_mean)
        self.assertEqual(std, other_std)

    def test_dataset_returns_chw_tensor_and_zero_based_label(self):
        mean, std = calculate_statistics(self.root, self.frame, image_size=8)
        dataset = FlowerDataset(self.root, self.frame, 'train', image_size=32, mean=mean, std=std)
        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertIsInstance(label, int)
        self.assertEqual(label, 0)
        self.assertTrue(torch.isfinite(image).all())

    def test_validation_transform_is_deterministic_and_training_can_augment(self):
        mean, std = calculate_statistics(self.root, self.frame, image_size=8)
        validation = FlowerDataset(self.root, self.frame, 'val', 32, mean, std)
        first, _ = validation[0]
        second, _ = validation[0]
        torch.testing.assert_close(first, second)
        transform = make_transform(32, mean, std, training=True)
        # 非对称图片能区分翻转前后；固定两次随机种子避免概率性的检查。
        image = Image.new('RGB', (32, 32), 'black')
        image.paste('white', (0, 0, 16, 32))
        with torch.random.fork_rng():
            torch.manual_seed(0)
            flipped = transform(image)
            torch.manual_seed(1)
            original = transform(image)
        self.assertFalse(torch.equal(flipped, original))
        torch.testing.assert_close(flipped, torch.flip(original, dims=[2]))

    def test_create_loaders_has_three_sets_and_expected_batch(self):
        loaders, datasets, stats = create_loaders(self.root, self.manifest, image_size=32, batch_size=4)
        self.assertEqual(set(loaders), {'train', 'val', 'test'})
        self.assertEqual({key: len(value) for key, value in datasets.items()}, {'train': 25, 'val': 5, 'test': 5})
        images, labels = next(iter(loaders['train']))
        self.assertEqual(tuple(images.shape), (4, 3, 32, 32))
        self.assertEqual(tuple(labels.shape), (4,))
        self.assertTrue(set(labels.tolist()).issubset(set(range(5))))
        self.assertEqual(stats['classes'], list(CLASS_NAMES))

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            make_transform(16, [0, 0, 0], [1, 1, 1])
        with self.assertRaises(ValueError):
            FlowerDataset(self.root, self.frame, 'unknown', 32, [0, 0, 0], [1, 1, 1])
        with self.assertRaises(ValueError):
            create_loaders(self.root, self.manifest, batch_size=0)


if __name__ == '__main__':
    unittest.main()

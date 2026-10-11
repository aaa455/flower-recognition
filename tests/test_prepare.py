"""检查清洗规则、划分可复现性和集合间完全重复文件隔离。"""
import unittest

import pandas as pd

from src.download import CLASS_NAMES
from src.prepare import clean_images, split_images


def example_images():
    return pd.DataFrame([
        {'path': f'{label}/{i:02}.jpg', 'label': label, 'width': 32, 'height': 32,
         'sha256': f'{label}-{i}'}
        for label in CLASS_NAMES for i in range(20)
    ])


class PrepareTests(unittest.TestCase):
    def test_clean_excludes_all_conflicts_before_deduplication(self):
        frame = example_images()
        duplicate = dict(frame.iloc[0], path='daisy/copy.jpg')
        conflict = dict(frame.iloc[1], path='roses/conflict.jpg', label='roses')
        frame = pd.concat([frame, pd.DataFrame([duplicate, conflict])], ignore_index=True)
        clean, conflicts, duplicates = clean_images(frame)
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(clean), 99)
        self.assertNotIn('daisy-1', set(clean['sha256']))
        self.assertEqual(clean['sha256'].nunique(), len(clean))
        self.assertEqual(duplicates['path'].tolist(), ['daisy/copy.jpg'])
        # 函数不会改写调用者的原始记录。
        self.assertEqual(len(frame), 102)

    def test_same_seed_is_independent_of_input_row_order(self):
        frame = example_images()
        first = split_images(frame, 42)
        shuffled = split_images(frame.sample(frac=1, random_state=9), 42)
        pd.testing.assert_frame_equal(first, shuffled)
        self.assertFalse(first['split'].eq('').any())

    def test_different_seed_changes_assignment(self):
        frame = example_images()
        self.assertFalse(split_images(frame, 42)['split'].equals(split_images(frame, 43)['split']))

    def test_every_class_has_expected_split_counts_and_no_overlap(self):
        manifest = split_images(example_images())
        for _, group in manifest.groupby('label'):
            self.assertEqual(group['split'].value_counts().to_dict(), {'train': 14, 'val': 3, 'test': 3})
        hashes = {split: set(manifest.loc[manifest['split'] == split, 'sha256'])
                  for split in ['train', 'val', 'test']}
        self.assertFalse(hashes['train'] & hashes['val'])
        self.assertFalse(hashes['train'] & hashes['test'])
        self.assertFalse(hashes['val'] & hashes['test'])
        self.assertEqual(len(set().union(*hashes.values())), 100)

    def test_split_rejects_duplicates(self):
        frame = example_images()
        with self.assertRaises(ValueError):
            split_images(pd.concat([frame, frame.iloc[:1]], ignore_index=True))

    def test_split_rejects_too_few_samples_and_negative_seed(self):
        with self.assertRaises(ValueError):
            split_images(example_images().groupby('label').head(6))
        with self.assertRaises(ValueError):
            split_images(example_images(), -1)


if __name__ == '__main__':
    unittest.main()

"""下载并解压五类花卉图片。运行：python -m src.download。"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
CLASS_NAMES = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')


def download_data(root: Path) -> Path:
    """将数据保存到 root，返回解压后的图片目录。"""
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / 'flower_photos'
    marker = image_dir / '.download_complete'

    # 完成标记和类别目录都存在时，不重复下载、解压。
    complete = marker.is_file() and (image_dir / 'LICENSE.txt').is_file()
    complete = complete and all((image_dir / name).is_dir() for name in CLASS_NAMES)
    if complete:
        print(f'数据已准备好：{image_dir}')
        return image_dir

    archive = root / 'flower_photos.tgz'
    if not archive.is_file():
        print('下载花卉数据，约 218 MiB……', flush=True)
        # 先写临时文件；下载成功后再改名，避免把中断的文件当作完整数据。
        temporary = root / 'flower_photos.tgz.part'
        urllib.request.urlretrieve(DATA_URL, temporary)
        temporary.replace(archive)
    else:
        print(f'使用已下载的压缩包：{archive}', flush=True)

    print('解压数据……', flush=True)
    with tarfile.open(archive, 'r:gz') as bundle:
        # 解压前检查路径与链接，确保文件只写入指定数据目录。
        for member in bundle.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root.resolve()) or member.issym() or member.islnk():
                raise ValueError(f'压缩包中包含不安全路径：{member.name}')
        bundle.extractall(root, filter='data')

    if not (image_dir / 'LICENSE.txt').is_file():
        raise ValueError('数据包缺少 LICENSE.txt，请检查下载内容。')
    if not all((image_dir / name).is_dir() for name in CLASS_NAMES):
        raise ValueError('数据包缺少花卉类别目录，请检查下载内容。')
    # 只有解压和目录检查都成功，才写完成标记。
    marker.write_text(DATA_URL + '\n', encoding='utf-8')
    print(f'数据准备完成：{image_dir}')
    return image_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data'), help='数据保存目录，默认 data')
    download_data(parser.parse_args(argv).root)


if __name__ == '__main__':
    main()

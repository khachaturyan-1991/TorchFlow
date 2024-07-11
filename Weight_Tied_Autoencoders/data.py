import tensorflow as tf
import cv2
import glob
from utils import smart_resize
import re


class SmartDataset(tf.data.Dataset):

    def _generator(img_paths, target_size):
        pattern = r'(/images/)(img_)'
        replacement = r'/masks/seg_'
        for img_path in img_paths:
            if isinstance(img_path, bytes):
                img_path = img_path.decode('utf-8')
            mask_path = re.sub(pattern, replacement, img_path)
            img = cv2.imread(img_path)
            Y = cv2.imread(mask_path)
            img = smart_resize(img, target_size=target_size, channel=3)
            Y = smart_resize(Y, target_size=target_size, channel=3)
            img = tf.cast(img, tf.float32) / 255.0
            Y = tf.cast(Y, tf.float32)
            mask = tf.where(Y > 1, tf.ones_like(Y), Y)
            yield img[:, :, 0:1], mask[:, :, 0:1]

    def __new__(cls,
                img_dir: str,
                first_img: int = 0,
                last_img: int = 32,
                target_size: int = 128
                ):
        img_paths = glob.glob(f"{img_dir}/*.png")[first_img:last_img]
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)),
            args=(img_paths, target_size)
        )


def create_dataloader(img_dir,
                      target_size=128,
                      batch_size=32,
                      shuffle=False,
                      first_img: int = 0,
                      last_img: int = 32):
    """
    Description
    -------
    Creates a dataloader from the custom dataset

    Parameters
    -------
    - image_dir: Path to the directory containing images.
    - masks_dir: Path to the directory containing masks.
    - target_size: Tuple (width, height) for resizing the images and masks.
    - batch_size: The size of the batches.
    - shuffle: Whether to shuffle the dataset.

    Returns
    -------
    - A tf.data.Dataset object.
    """
    dataset = SmartDataset(img_dir=img_dir,
                           target_size=target_size,
                           first_img=first_img, last_img=last_img)

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(glob.glob(f"{img_dir}/*.png")))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

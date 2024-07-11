import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import argparse


def smart_resize(img: np.array,
                 target_size: int = 128,
                 channel: int = 3):
    """
    Description
    -------
    Resize an image to squeeze into a template; \
    It should be used to resize images to a shape suitable \
    for neaural a neural network, \
    but preserving all the proportions of an object on an image

    Parameters
    -------
    img (ndarra): an image
    target_size (int): a size of an expected template

    Return
    -------
    tamplate (ndarray): resized image
    """
    height, width = img.shape[:2]
    scale = min(target_size / width, target_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(img,
                               (new_width, new_height),
                               interpolation=cv2.INTER_AREA)

    tamplate = np.zeros((target_size, target_size, channel),
                        dtype=np.uint8)
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    tamplate[y_offset:y_offset + new_height,
             x_offset:x_offset + new_width] = resized_image

    return tamplate


def test_prediction(model,
                    dataloader,
                    loss_fn,
                    accuracy_fn,
                    device: str = "cpu"):
    """
    Description
    -------
        Takes a trained model and test data to astimate loss
        and plot examples of Predictins and Masks

    Parameteres
    -------
        model: a trained model
        dataloader: a test dataloader
        loss_fn: loss function
        device: device on which to run calculations

    Return
    -------
        None
    """
    acumalative_loss = 0
    accuracy_loss = 0
    n = 0
    with tf.device(device):
        for X, mask in dataloader:
            n += 1
            Y_pred = model.predict(X)
            Y_pred = tf.squeeze(Y_pred, axis=-1)
            acumalative_loss += loss_fn(Y_pred, mask)
            accuracy_loss += accuracy_fn(Y_pred, mask)
    print("Average error on test data: ", acumalative_loss / n)
    print("Average accuracy on test data: ", accuracy_loss / n)

    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    axes[0].imshow(Y_pred[0])
    axes[1].imshow(mask[0])
    axes[2].imshow(Y_pred[1])
    axes[3].imshow(mask[1])
    plt.tight_layout()
    plt.show()


def dice_loss(y_true, y_pred, epsilon=1e-6):
    """Dice loss"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(
        y_true, axis=[1, 2]) + tf.reduce_sum(
            y_pred, axis=[1, 2])

    dice = (2. * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice

    return tf.reduce_mean(dice_loss)


def plot_hystory(h_train: dict,
                 h_test: dict,
                 file_name: str = "history"):
    _ = plt.figure(figsize=(6, 6))

    plt.plot(h_train.keys(),
             h_train.values(),
             "-o", label="Train")

    plt.plot(h_test.keys(),
             h_test.values(),
             "-*", label="Validation")

    plt.legend(title="Dataset")

    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.savefig(f"{file_name}.png")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Autoencoder", help="Choose between Autoencoder and ZeroDecoder")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, gpu, mpi")
    parser.add_argument("--data_folder", type=str, default="../data/clothing/images")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--output_freq", type=int, default=2, help="output frequency")
    parser.add_argument("--save_history", type=str, default=None, help="name of the file to save history")
    args = parser.parse_args()
    return args

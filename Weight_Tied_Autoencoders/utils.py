import cv2
import numpy as np
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
    parser.add_argument("--framework", type=str, default="tensorflow", help="Choose between tensorflow and torch")
    parser.add_argument("--model", type=str, default="autoencoder", help="Choose between autoencoder, autoencoder_wt, unet, unet_wt")
    parser.add_argument("--device", type=str, default="cpu", help="cpu, gpu, mps")
    parser.add_argument("--data_folder", type=str, default="../data/clothing/images")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--img_size", type=int, default=128, help="image size")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--output_freq", type=int, default=2, help="output frequency")
    parser.add_argument("--save_history", type=str, default=None, help="name of the file to save history")
    args = parser.parse_args()
    return args


def count_torch_parameters(model):
    trainable_params = 0
    non_trainable_params = 0
    trainable_weights = 0
    non_trainable_weights = 0

    for param in model.parameters():
        param_count = param.numel()
        if param.requires_grad:
            trainable_params += param_count
            trainable_weights += param_count * param.element_size()
        else:
            non_trainable_params += param_count
            non_trainable_weights += param_count * param.element_size()

    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
    print(f"Trainable weights (Mb): {trainable_weights / 1e6}")
    print(f"Non-trainable weights (Mb): {non_trainable_weights / 1e6}")

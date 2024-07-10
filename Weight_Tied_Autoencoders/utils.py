import cv2
import numpy as np
import tensorflow as tf
import tqdm
import matplotlib.pylab as plt


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

    tamplate[y_offset:y_offset+new_height,
             x_offset:x_offset+new_width] = resized_image

    return tamplate


def train_fn(model,
             loss_fn,
             train_dataloader,
             val_dataloader,
             optimizer,
             first_step: int = 0,
             last_step: int = 100,
             output_freq: int = 2,
             device: str = "cpu"):
    """
    Description
    -------
        Trained a model on a train data;

    Parameteres
    -------
        model: a trained model
        dataloader: a test dataloader
        loss_fn: loss function
        optimizer: optimization function
        first_step: epoch from which training begins
        epochs: number of epochs
        output_freq: frequency of printing out training results
        device: device on which to run calculations

    Return
    -------
        average_loss - dictionary with average loss per epoch
    """
    avg_train_loss = {}
    avg_val_loss = {}
    with tf.device(device):
        for epoch in tqdm.tqdm(range(first_step, last_step)):
            train_loss = 0
            val_loss = 0
            n = 0
            for X, Y in train_dataloader:
                mask = tf.where(Y > 1, tf.ones_like(Y), Y)
                with tf.GradientTape() as tape:
                    Y_pred = model(X, training=True)
                    loss = loss_fn(Y_pred, mask)
                    train_loss += loss.numpy()
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients,
                                              model.trainable_variables))
                n += 1
            avg_train_loss[epoch] = train_loss / n
            n = 0
            for X, Y in val_dataloader:
                mask = tf.where(Y > 1, tf.ones_like(Y), Y)
                Y_pred = model(X, training=False)
                loss = loss_fn(Y_pred, mask)
                val_loss += loss.numpy()
                n += 1
            avg_val_loss[epoch] = val_loss / n
            if epoch % output_freq == 0:
                print(f"Epoch {epoch + 1 + first_step}/{first_step + last_step}, \
                      Train loss: {avg_train_loss[epoch]:.4f} Validation loss: {avg_val_loss[epoch]:.4f}")
        return avg_train_loss, avg_val_loss


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
        for X, Y in dataloader:
            n += 1
            mask = tf.where(Y > 1, tf.ones_like(Y), Y)
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

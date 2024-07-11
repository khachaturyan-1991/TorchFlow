import tensorflow as tf


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

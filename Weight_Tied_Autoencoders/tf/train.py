import tensorflow as tf
from tensorflow import keras
import tqdm
import matplotlib.pylab as plt


class Trainer():

    def __init__(self,
                 model: keras.models.Model,
                 loss_fn: keras.losses.Loss,
                 optimizer: keras.optimizers.Optimizer,
                 device: str = "cpu:0"
                 ) -> None:
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self,
                   dataloader: tf.data.Dataset):
        step_loss = 0
        n = 0
        for X, mask in dataloader:
            with tf.GradientTape() as tape:
                pred = self.model(X, training=True)
                loss = self.loss_fn(pred, mask)
                step_loss += loss.numpy()
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients,
                                               self.model.trainable_variables))
            n += 1
        return step_loss / n

    def test_step(self,
                  dataloader: tf.data.Dataset):
        step_loss = 0
        n = 0
        for X, mask in dataloader:
            pred = self.model(X, training=False)
            loss = self.loss_fn(pred, mask)
            step_loss += loss.numpy()
            n += 1
        return step_loss / n

    def fit(self,
            train_dataloder: tf.data.Dataset,
            test_dataloder: tf.data.Dataset,
            output_freq: int = 2,
            first_step: int = 0,
            epochs: int = 10
            ):
        last_step = first_step + epochs
        avg_train_loss = {}
        avg_val_loss = {}
        with tf.device(self.device):
            for epoch in tqdm.tqdm(range(epochs)):
                train_loss = self.train_step(train_dataloder)
                test_loss = self.test_step(test_dataloder)
                avg_train_loss[first_step + epoch] = train_loss
                avg_val_loss[first_step + epoch] = test_loss
                if epoch % output_freq == 0:
                    print(f"Epoch {epoch + 1 + first_step}/{first_step + last_step}, \
                        Train loss: {train_loss:.4f} Validation loss: {test_loss:.4f}")
        return avg_train_loss, avg_val_loss


def test_prediction(model,
                    dataloader,
                    loss_fn,
                    image_name: str = "test_prediction"):
    loss = 0
    n = 0
    with tf.device("cpu"):
        for X, masks in dataloader:
            Y_pred = model.predict(X)
            loss += loss_fn(Y_pred, masks).numpy()
            n += 1

    print("Loss per test: ", loss / n)

    Y_pred = tf.squeeze(Y_pred, axis=-1)
    masks = masks.numpy()
    _, axes = plt.subplots(2, 4, figsize=(8, 5))
    axes = axes.ravel()
    for i in range(4):
        axes[i].imshow(Y_pred[i][:, :, 0])
        axes[i].axis("off")
        y = masks[i][:, :, 0]
        y[y > 1] = 1
        axes[i + 4].imshow(y)
        axes[i + 4].axis("off")

    plt.tight_layout()
    plt.savefig(f"{image_name}.png")

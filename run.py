import tensorflow as tf
import matplotlib.pylab as plt
import time

from autoencoder import Autoencoder, AutoencoderZeroDecoder
from utils import train_fn, dice_loss
from data import create_dataloader

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/cpu:0"
print("Device: ", device)

BATCH_SIZE = 32
IMG_SIZE = 128

train_dataloader = create_dataloader(img_dir="../data/clothing/images",
                                     target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                     first_img=0, last_img=32)
test_dataloader = create_dataloader(img_dir="../data/clothing/images",
                                    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                    first_img=900, last_img=950)
val_dataloader = create_dataloader(img_dir="../data/clothing/images",
                                   target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                   first_img=0, last_img=32)

model = Autoencoder()
input_tensor = tf.random.normal([1, 128, 128, 1])
output_tensor = model(input_tensor)
print("Input tensor shape: ", input_tensor.shape)
print("Output tensor shape: ", output_tensor.shape)
print(model.summary())

optim = tf.keras.optimizers.Adam(learning_rate=1e-4)

t1 = time.time()

FIRST_STEP, EPOCHS = 0, 30
h_train, h_test = train_fn(model=model,
                           train_dataloader=train_dataloader,
                           val_dataloader=val_dataloader,
                           loss_fn=dice_loss,
                           optimizer=optim,
                           first_step=FIRST_STEP,
                           last_step=FIRST_STEP + EPOCHS,
                           device="\CPU",
                           output_freq=5)

model.save('autoencoder_zero_decoder_cloth.keras')

t2 = time.time()
print("Time Spent: ", (t2 - t1) // 60)


fig = plt.figure(figsize=(6, 6))

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
plt.savefig("loss_history_auto.png")

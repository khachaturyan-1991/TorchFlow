import tensorflow as tf
import matplotlib.pylab as plt
import time

from args_parse import parse_arguments
from autoencoder import Autoencoder, AutoencoderZeroDecoder
from utils import train_fn, dice_loss
from data import create_dataloader


if __name__ == "__main__":

    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    DEVICE = f"/{args.device.lower()}:0"
    EPOCHS = args.epochs
    OUTPUT_FREQUENCY = args.output_freq
    SAVE_HISTORY = args.save_history
    MODEL_TYPE = args.model.lower()
    DATA_FOLDER = args.data_folder

    models_list = ["autoencoder", "zero_decoder"]

    train_dataloader = create_dataloader(img_dir=f"{DATA_FOLDER}",
                                         target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                         first_img=0, last_img=32)
    test_dataloader = create_dataloader(img_dir=f"{DATA_FOLDER}",
                                        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                        first_img=900, last_img=950)
    val_dataloader = create_dataloader(img_dir=f"{DATA_FOLDER}",
                                       target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                       first_img=0, last_img=32)

    assert MODEL_TYPE in models_list, "Model type mast be either autoencoder or zero_decoder"
    if MODEL_TYPE == "zero_decoder":
        model = AutoencoderZeroDecoder()
    else:
        model = Autoencoder()

    input_tensor = tf.random.normal([1, IMG_SIZE, IMG_SIZE, 1])
    output_tensor = model(input_tensor)
    print("Input tensor shape: ", input_tensor.shape)
    print("Output tensor shape: ", output_tensor.shape)
    print(model.summary())

    optim = tf.keras.optimizers.Adam(learning_rate=1e-4)

    t1 = time.time()

    FIRST_STEP = 0
    h_train, h_test = train_fn(model=model,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               loss_fn=dice_loss,
                               optimizer=optim,
                               first_step=FIRST_STEP,
                               last_step=FIRST_STEP + EPOCHS,
                               device=DEVICE,
                               output_freq=OUTPUT_FREQUENCY)

    model.save(f'{MODEL_TYPE}.keras')

    t2 = time.time()
    print("Time Spent: ", (t2 - t1) // 60)

    if SAVE_HISTORY:

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
        plt.savefig(f"{SAVE_HISTORY}.png")

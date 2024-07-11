import tensorflow as tf
from args_parse import parse_arguments
from tf.autoencoder import Autoencoder, AutoencoderZeroDecoder
from utils import dice_loss, plot_hystory
from tf.data import create_dataloader
from tf.train import Trainer


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
    print(model.summary())

    optim = tf.keras.optimizers.Adam(learning_rate=1e-4)

    FIRST_STEP = 0
    model_train = Trainer(model=model,
                          loss_fn=dice_loss,
                          optimizer=optim,
                          device=DEVICE)

    h_train, h_test = model_train.fit(train_dataloder=train_dataloader,
                                      test_dataloder=test_dataloader,
                                      output_freq=OUTPUT_FREQUENCY,
                                      epochs=EPOCHS,
                                      first_step=FIRST_STEP)

    model.save(f'{MODEL_TYPE}.keras')

    if SAVE_HISTORY:
        plot_hystory(h_train,
                     h_test,
                     SAVE_HISTORY)

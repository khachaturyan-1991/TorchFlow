import tensorflow
import torch
from utils import plot_hystory, parse_arguments


if __name__ == "__main__":

    # Get parameters from a comand line
    args = parse_arguments()
    FRAME = args.framework.lower()
    BATCH_SIZE = args.batch_size
    IMG_SIZE = args.img_size
    DEVICE = args.device.lower()
    EPOCHS = args.epochs
    OUTPUT_FREQUENCY = args.output_freq
    SAVE_HISTORY = args.save_history
    MODEL_TYPE = args.model.lower()
    DATA_FOLDER = args.data_folder

    frams_list = ["tensorflow", "torch"]
    assert FRAME in frams_list, "Framework must be either tensorflow or torch"
    if FRAME == "tensorflow":
        from tf.autoencoder import Autoencoder, AutoencoderZeroDecoder
        from tf.data import create_dataloader
        from tf.train import Trainer
        from tf.losses import dice_loss
    else:
        from tr.autoencoder import Autoencoder, AutoencoderZeroDecoder
        from tr.data import create_dataloader
        from tr.train import Trainer
        from tr.losses import dice_loss

    # Get data from the DATA_FOLDER
    train_dataloader = create_dataloader(img_dir=f"{DATA_FOLDER}",
                                         target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                         first_img=0, last_img=900)
    test_dataloader = create_dataloader(img_dir=f"{DATA_FOLDER}",
                                        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                        first_img=900, last_img=950)
    val_dataloader = create_dataloader(img_dir=f"{DATA_FOLDER}",
                                       target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                       first_img=0, last_img=32)

    # MODEL_TYPE is limited by a models_list
    models_list = ["autoencoder", "zero_decoder"]
    assert MODEL_TYPE in models_list, "Model type mast be either autoencoder or zero_decoder"
    if MODEL_TYPE == "zero_decoder":
        model = AutoencoderZeroDecoder()
    else:
        model = Autoencoder()

    # Instantiate model and optimizer
    if FRAME == "tensorflow":
        input_tensor = tensorflow.random.normal([1, IMG_SIZE, IMG_SIZE, 1])
        _ = model(input_tensor)
        print(model.summary())
        optim = tensorflow.keras.optimizers.Adam(learning_rate=1e-4)
    else:
        model.to(device=DEVICE)
        for name, param in model.named_parameters():
            print(f"{name}: {param.size()}")
        optim = torch.optim.Adam(params=model.parameters(),
                                 lr=1e-4)

    # Instantiate and traning
    FIRST_STEP = 0
    model_train = Trainer(model=model,
                          loss_fn=dice_loss,
                          optimizer=optim,
                          device=DEVICE)

    # Run traning
    h_train, h_test = model_train.fit(train_dataloder=train_dataloader,
                                      test_dataloder=test_dataloader,
                                      output_freq=OUTPUT_FREQUENCY,
                                      epochs=EPOCHS,
                                      first_step=FIRST_STEP)

    if FRAME == "tensorflow":
        model.save(f'{MODEL_TYPE}.keras')
    else:
        torch.save(model.state_dict(), f'{MODEL_TYPE}.pth')

    # Plot loss history
    if SAVE_HISTORY:
        plot_hystory(h_train,
                     h_test,
                     SAVE_HISTORY)

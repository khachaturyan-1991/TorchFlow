# TorchFlow
This repository contans PyTorhc (./tr) and TensorFlow (./tf) implementations of autoencoder architecture with trainable decoder and non-trainable decoduer using tying weights approch for convolution transposed layers.

to use the repository clone it and place your data above the the Weight_Tied_Autoencoders directory:
root
    |
    |
    |--- Weight_Tied_Autoencoders
    |                            |---tr
    |                            |---tf
    |
    |
    |--- data
            |---images
            |---masks

then go to the Weight_Tied_Autoencoders directory and type:
python -m run.py
you may use one of the follwoing optional flags
    --framework help="Choose between tensorflow and torch"
    --model help="Choose between Autoencoder and ZeroDecoder"
    --device help="cpu, gpu, mpi"
    --data_folder help="path/to/images"
    --batch_size help="batch size"
    --img_size help="image size"
    --epochs help="number of epochs"
    --output_freq help="output frequency"
    --save_history help="name of the file to save history"

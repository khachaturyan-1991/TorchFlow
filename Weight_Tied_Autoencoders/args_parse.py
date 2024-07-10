import argparse


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

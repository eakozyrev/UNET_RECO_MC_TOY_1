import gen_fig_1
import numpy as np

from unet_model import UNet
from dataset import Data_Loader
from torch import optim
import torch.nn as nn
import torch
from train import *
from predict import *
from gen_fig_1 import *

if __name__ == "__main__":
    Net = UNet(n_channels=1, n_classes=1)
    Net.to()
    data_path = "data/"
    #generate_all(700)
    #train_nn(Net, data_path, epochs=50, batch_size=10, lr=0.0001)
    predict()
    analyze()
    draw_loss_from_file("history_loss.dat")


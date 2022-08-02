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
from validate import *
from shutil import copyfile
import os

# takes in a module and applies the specified weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    Net = UNet(n_channels=1, n_classes=1, track_running=True)
    Net.to(device=device)
    Net.apply(weights_init_uniform_rule)


    train_nn(Net, device, 'data/', epochs=1, batch_size=1)
    predict('data/','results/best_model_L1_emb.pth')
    draw_loss_from_file("results/history_loss_L1.dat")
    
''''
    for el in [2,10,50]:
        data_path = f'data_{el}/'
        #train_nn(Net, device, data_path, epochs=3, batch_size=10)
        #copyfile('best_model.pth', f'results/best_model_{el}.pth')
        #copyfile('history_loss.dat',f'results/best_model_{el}.dat')
        draw_loss_from_file("results/best_model_" + str(el) + ".dat")
        predict(f'data_{el}/', f"results/best_model_{el}.pth")
'''
    # generate_all(700)
    #analyze('D:\PENN\MC_1_2022_1\data_10/')
    #analyze()
    #list = [2,10,50]
    #for el in list:
    #    draw_loss_from_file(f'results/best_model_{el}.dat')
    #    predict(f'data_{el}/', f"results/best_model_{el}.pth")
    #for el in list:
        #draw_loss_from_file("results/" + el + ".dat")
        #predict(f'../MC_1_2022_2/data_{el}/', f"../MC_1_2022_2/results/best_model_{el}.pth")
     #   copyfile('data_10/validation/unet/6.png', "data_10/validation/unet/6_" + el + ".png")
    #test = Validate()
    #print(test.calculate_loss('D:\PENN\MC_1_2021_12\data_10/validation/image_mixed/0.png',
    #                          'D:\PENN\MC_1_2021_12\data_10/validation/unet/0.png'))
    #print(test.calculate_loss('D:\PENN\MC_1_2021_12\data_10/validation/image_mixed/0.png',
    #                          'D:\PENN\MC_1_2021_12\data_10/validation/image/0.png'))
    #image_1 = validate_sample('D:\PENN\MC_1_2021_12\data_10/validation/image_mixed/')
    #image_2 = validate_sample('D:\PENN\MC_1_2021_12\data_10/validation/unet/')
    #crit = nn.L1Loss()
    #print('loss', crit(image_1, image_2))
    #draw_fig(['D:\PENN\MC_1_2021_12\data_10/validation\label/6.png',
     #                'D:\PENN\MC_1_2021_12\data_10/validation\image/6.png'])
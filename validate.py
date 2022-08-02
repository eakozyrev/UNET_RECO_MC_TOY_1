import glob
import os
import random
import cv2
import torch
import math
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
import torch
from numpy import loadtxt
import codecs
import math
from torch.autograd import Variable
import time
from unet_model import UNet
import matplotlib.pyplot as plt
#from torchviz import make_dot

class Validate:

    def __init__(self):
        self.criterion = nn.L1Loss()
        self.loss = []

    def calculate_loss(self,image,label):
        image = cv2.imread(image)
        label = cv2.imread(label)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label= label.reshape(1, label.shape[0], label.shape[1])
        image = torch.tensor(image)/255.
        label = torch.tensor(label)/255.
        image = image.to(device='cpu', dtype=torch.float32)
        label = label.to(device='cpu', dtype=torch.float32)
        res = self.criterion(image, label)
        return res



    def calculate_loss1(self, image, label):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1,1, image.shape[0], image.shape[1])
        image = torch.tensor(image)/255.
        image = image.to(device='cpu', dtype=torch.float32)
        res = self.criterion(image, label)
        return res



    def loop_over_valid(self):
        tests_path = glob.glob('D:\PENN\MC_1_2022_1\data_10/validation\label/*.png')
        for test_path in tests_path:
            num = test_path.split('.')[0].split('label\\')[1]
            label_path = "D:\PENN\MC_1_2022_1\data_10/validation/image/" + str(num) + ".png"
            self.loss.append(self.calculate_loss(test_path,label_path))

    def draw(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('#')
        ax.set_title('L1 loss for validation (blurred image vs label)')
        plt.plot(self.loss)
        plt.show()

    def plot_slice(self,files):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for file in files:
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = torch.tensor(image) / 255.
            plt.plot(image[49][33:117],label=file.split('validation')[1])
        plt.legend()
        plt.show()

    def plot_slice1(self,files):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for file in files:
            ar = []
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = torch.tensor(image) / 255.
            r = 0.2
            for angle in range(-5,355):
                x = (r * math.cos(angle/360*2.*math.pi) + 0.5)*144.
                y = (r * math.sin(angle/360*2.*math.pi) + 0.5)*144.
                ar.append(image[int(x)][int(y)])
            name = file
            if '/valid' in name: name = name.split('/valid')[1]
            plt.plot(ar,label=name)
        plt.legend()
        plt.show()

def vis():
    '''
    net = UNet(n_channels=1, n_classes=1, track_running=True)
    device = 'cpu'
    net.to(device=device)
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    net.eval()
    img = cv2.imread('D:\PENN\MC_3_2022_1\data\image/54.png')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.reshape(1, 1, img.shape[0], img.shape[1])
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    pred = net(img_tensor)
    make_dot(pred, params=dict(list(net.named_parameters()))).render("rnn_torchviz", format="png")
'''
    model = nn.Sequential()
    model.add_module('W0', nn.Linear(8, 16))
    model.add_module('tanh', nn.Tanh())
    model.add_module('W1', nn.Linear(16, 1))

    x = torch.randn(1, 8)
    y = model(x)
    print(y)
    g = make_dot(y, params=dict(list(model.named_parameters())))
    g.view()
    input("Press Enter to continue...")


if __name__ == '__main__':
    #vis()

    test = Validate()
    #test.loop_over_valid()
    #test.draw()

    #path = glob.glob(os.path.join('D:\PENN\MC_1_2021_12\data_10/validation/unet/','6_*.png'))
    test.plot_slice1(['2.png','3.png','data/validation/unet/2_2.png'])
    #print(path)
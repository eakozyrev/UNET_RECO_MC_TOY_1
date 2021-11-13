import gen_fig_1
import numpy as np
from unet_model import UNet
from dataset import Data_Loader
from torch import optim
import torch.nn as nn
import torch

def train_nn(Net, data_path, epochs=10, batch_size=4, lr=0.0001):

    isbi_dataset = Data_Loader(data_path)
    file = open("history_loss.dat","w")
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    optimizer = optim.RMSprop(Net.parameters(), lr=lr, weight_decay=0, momentum=0.95)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        Net.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(dtype=torch.float32)
            label = label.to(dtype=torch.float32)
            pred = Net(image)
            loss = criterion(pred, label)
            file.write(str(loss.item()) + '\n')
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
                torch.save(Net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()
    file.close()

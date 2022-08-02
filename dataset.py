import glob
import os
import random
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class Data_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip



    def get_small_matrix(self, matrix, lines):

        res = []
        if len(lines) == 0:
            while len(res) < 625:
                res.append(0)
            return res
        if len(lines.shape) == 1:
            ind1 = int(lines[0])
            ind2 = int(lines[1])
            res.append(matrix[0][ind2][ind1])
            while len(res) < 625:
                res.append(0)
            return res
        for el in lines:
            if len(str(el).strip())==0: continue
            ind1 = int(el[0])
            ind2 = int(el[1])
            what_append = 0;
            if ind1 >= 0 and ind2 >= 0: what_append = matrix[0][ind2][ind1]
            res.append(what_append)
        while len(res) < 625:
            res.append(0)
        return res

    def get_small_matrix_banch(self, matrix, lines):
        res = []
        i = 0
        for el in matrix:
            res.append(self.get_small_matrix(el,lines[i]))
            i+=1
        res = torch.tensor(res)
        return res

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')
        label_emb_path = image_path.replace('image', 'label_emb')
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        label_emb = cv2.imread(label_emb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label_emb = cv2.cvtColor(label_emb, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        label_emb = label_emb.reshape(1, label_emb.shape[0], label_emb.shape[1])
        if label.max() > 1:
            label = label / 255
            label_emb = label_emb/255
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            label_emb = self.augment(label_emb, flipCode)
        return image, label, label_emb

    def __len__(self):
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = Data_Loader("D:\PENN\MC_1_2021_12\data/")
    print("number：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=3,
                                               shuffle=True)
    for image, label,image_emb, label_emb,lines in train_loader:
        print(image_emb)

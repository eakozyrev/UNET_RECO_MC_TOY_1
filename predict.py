import glob
import numpy as np
import torch
import os
import cv2
from unet_model import UNet
import matplotlib.pyplot as plt

def analyze():
    tests_path = glob.glob('data/test/*.png')
    for test_path in tests_path:
        num = test_path.split('.')[0].split('test/')[1]
        figure, axis = plt.subplots(2, 2)
        label_path = "data/test/label/"+str(num)+".png"
        t_path = "data/test/"+str(num)+".png"
        u_path = "data/test/unet/"+str(num)+".png"
        save_res_path = "data/test/unet/"+str(num)+"_res.png"
        image0 = cv2.cvtColor(cv2.imread(label_path),cv2.COLOR_BGR2GRAY)
        image1 = cv2.cvtColor(cv2.imread(t_path),cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(cv2.imread(u_path),cv2.COLOR_BGR2GRAY)
        im = axis[0][0].imshow(image1)
        axis[0][0].text(1, 1, 'image', bbox={'facecolor': 'white', 'pad': 2})
        figure.colorbar(im, ax = axis[0][0])
        im = axis[0][1].imshow(image2)
        axis[0][1].text(1, 1, 'unet', bbox={'facecolor': 'white', 'pad': 2})
        figure.colorbar(im, ax=axis[0][1])
        init_r = np.ndarray(shape = image2.shape)
        for i in range(init_r.shape[0]):
            for j in range(init_r.shape[1]):
                init_r[i][j] = int(image1[i][j]) - int(image0[i][j])
        im = axis[1][0].imshow(init_r)
        axis[1][0].text(1, 1, 'image/label', bbox={'facecolor': 'white', 'pad': 2})
        figure.colorbar(im, ax=axis[1][0])
        fin_r = np.ndarray(shape = image2.shape)
        for i in range(fin_r.shape[0]):
            for j in range(fin_r.shape[1]):
                fin_r[i][j] = int(image2[i][j]) - int(image0[i][j])
        im = axis[1][1].imshow(fin_r)
        axis[1][1].text(1, 1, 'unet/label', bbox={'facecolor': 'white', 'pad': 2})
        figure.colorbar(im, ax=axis[1][1])
        figure.savefig(save_res_path)
        plt.show()


def draw_loss_from_file(file):
    with open(file) as fp:
        arr = fp.readlines()
        y = []
        for el in arr:
            try: y.append(float(el))
            except: continue
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Batches processed')
    ax.set_title('dataset=700,RMSProp,ep=50,b_size=10,lr=0.0001,mom=0.95,w_d=0')
    plt.plot(y)
    plt.savefig("loss_hist.png", dpi=fig.dpi)
    plt.show()

def predict():
    net = UNet(n_channels=1, n_classes=1)
    net.to()
    net.load_state_dict(torch.load('best_model_this.pth'))
    net.eval()
    tests_path = glob.glob('data/test/*.png')
    for test_path in tests_path:
        num = test_path.split('.')[0].split('test/')[1]
        save_res_path = "data/test/unet/" + num + '.png'
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(dtype=torch.float32)
        pred = net.forward(img_tensor)
        pred = np.array(pred.data.cpu()[0])[0]*255
        cv2.imwrite(save_res_path, pred)

if __name__ == "__main__":
    #predict()
    analyze()
    #draw_loss_from_file("history_loss.dat")
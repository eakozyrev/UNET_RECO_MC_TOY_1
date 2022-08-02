import glob
import numpy as np
import torch
import os
import cv2
import torch.nn as nn
from unet_model import UNet
import matplotlib.pyplot as plt
from validate import *

def analyze(path):
    tests_path = glob.glob(path+'validation/image/*.png')
    for test_path in tests_path:
        num = test_path.split('.')[0].split('image\\')[1]
        figure, axis = plt.subplots(2, 2)
        label_path = path + "validation/label/"+str(num)+".png"
        t_path = path + "validation/image/"+str(num)+".png"
        u_path = path + "validation/unet/"+str(num)+".png"
        save_res_path = path + "validation/unet/"+str(num)+"_res.png"
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
        im = axis[1][0].imshow(init_r,vmin = -150,vmax = 150)
        axis[1][0].text(1, 1, 'image-label', bbox={'facecolor': 'white', 'pad': 2})
        figure.colorbar(im, ax=axis[1][0])
        fin_r = np.ndarray(shape = image2.shape)
        for i in range(fin_r.shape[0]):
            for j in range(fin_r.shape[1]):
                fin_r[i][j] = int(image2[i][j]) - int(image0[i][j])
        im = axis[1][1].imshow(fin_r, vmin = -150,vmax = 150)
        axis[1][1].text(1, 1, 'unet-label', bbox={'facecolor': 'white', 'pad': 2})
        figure.colorbar(im, ax=axis[1][1])
        figure.savefig(save_res_path)
        plt.show()


def draw_loss_from_file(file):
    with open(file) as fp:
        arr = fp.readlines()
        x = []
        x1 = []
        ylossL1 = []
        ylossSSIM = []
        factor = []
        ylossL1_2, ylossSSIM_2, ylossL1_10, ylossSSIM_10, ylossL1_20, ylossSSIM_20, ylossL1_50, ylossSSIM_50  = [],[],[],[],[],[],[],[]
        lr = []
        i = 0
        for el in arr:
            try:
                i += 1
                lr.append(float(el.split(' ')[0].replace('[','').replace(']','')))
                factor.append(float(el.split(' ')[1]))
                ylossL1.append(float(el.split(' ')[2]))
                ylossSSIM.append(float(el.split(' ')[3]))
                x.append(i)
                ylossL1_2.append(float(el.split(' ')[4]))
                ylossSSIM_2.append(float(el.split(' ')[5]))
                ylossL1_10.append(float(el.split(' ')[6]))
                ylossSSIM_10.append(float(el.split(' ')[7]))
                ylossL1_20.append(float(el.split(' ')[8]))
                ylossSSIM_20.append(float(el.split(' ')[9]))
                ylossL1_50.append(float(el.split(' ')[10]))
                ylossSSIM_50.append(float(el.split(' ')[11]))
                x1.append(i)
            except: continue
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Batches processed')
    ax.set_title(file.split('/')[-1])
    plt.plot(x,ylossL1,label='L1 loss')
    plt.plot(x, ylossSSIM,label = "EMB loss")
   # plt.plot(x1, ylossL1_2,'p',label ='L1_2')
   # plt.plot(x1, ylossL1_10, 'p', label='L1_10')
   # plt.plot(x1, ylossL1_20, 'p', label='L1_20')
   # plt.plot(x1, ylossL1_50, 'p', label='L1_50')
    plt.plot(x,factor)
  #  plt.plot(x1, ylossSSIM_2, '--', label='EMB_2')
  #  plt.plot(x1, ylossSSIM_10, '--', label='EMB_10')
    #plt.plot(x1, ylossSSIM_20, '--', label='EMB_20')
  #  plt.plot(x1, ylossSSIM_50, '--', label='EMB_50')
    plt.plot(x, lr)
    plt.legend()

    intercept = np.polyfit(x[-50:-1], ylossL1[-50:-1], 0)
    abline_values = [intercept for i in x[-50:-1]]
    plt.plot(x[-50:-1], abline_values, 'b')
    plt.text(x[-1] + 10, intercept, str("%.4f" % intercept), fontsize=12)

    intercept = np.polyfit(x[-50:-1], ylossSSIM[-50:-1], 0)
    abline_values = [intercept for i in x[-50:-1]]
    plt.plot(x[-50:-1], abline_values, 'b')
    plt.text(x[-1] + 10, intercept, str("%.4f" % intercept), fontsize=12)

    intercept = np.polyfit(x[-50:-1], factor[-50:-1], 0)
    abline_values = [intercept for i in x[-50:-1]]
    plt.plot(x[-50:-1], abline_values, 'b')
    plt.text(x[-70] , intercept-0.0003, str("factor = %.4f " % (intercept)), fontsize=12)

    plt.savefig(file + '.png', dpi=fig.dpi)
    plt.show()

def predict(path,bestmodel):
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = UNet(n_channels=1, n_classes=1, track_running=True)
    net.to(device=device)
    net.load_state_dict(torch.load(bestmodel, map_location=device))
    net.eval()
    tests_path = glob.glob(path+'validation/image/*.png')
    print(tests_path)
    try: os.mkdir(path+'validation/unet/')
    except OSError: print('ciao')
    for test_path in tests_path:
        num = test_path.split('image\\')[1].split('.png')[0]
        save_res_path = path + "validation/unet/" + num + '.png'
        img = cv2.imread(test_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        pred = net(img_tensor)
        #test = Validate()
        #imaget = cv2.imread(str('D:\PENN\MC_1_2021_12\data_10/validation/label/'+num+'.png'))
        #imaget = cv2.cvtColor(imaget, cv2.COLOR_BGR2GRAY)
        #imaget = imaget.reshape(1, 1, imaget.shape[0], imaget.shape[1])
        #imaget = imaget / 255
        #imaget = torch.tensor(imaget)
        #crit = nn.L1Loss()
        #print(num,'loss',crit(imaget,pred))
        #print(num,'loss',test.calculate_loss1(str('D:\PENN\MC_1_2021_12\data_10/validation/label/'+num+'.png'),pred))
        pred = np.array(pred.data[0])[0];
        #factor = 255/np.max(pred)
        pred = pred*255
        cv2.imwrite(save_res_path, pred)


def draw_fig(path):
    figure, axis = plt.subplots(1, 2)
    img = cv2.imread(path[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im = axis[0].imshow(img) #,vmin = -150,vmax = 150)
    figure.colorbar(im, ax=axis[0])

    img = cv2.imread(path[1])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im = axis[1].imshow(img) #,vmin = -150,vmax = 150)
    figure.colorbar(im, ax=axis[1])
    plt.show()

if __name__ == "__main__":
    draw_fig(("data\label_emb/6.png","data\label_emb/6.png"))
    #analyze()
    #draw_loss_from_file("history_loss.dat")
#  Evgeny Kozyrev


import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
import math
import cv2
from scipy import integrate
import shutil


class Toy_fig_1:
    """This is a base class to generate toy figure type #1"""

    def __init__(self,width, height):
        self.TOF_res = 45    # TOF resolution = 45 mm
        self.LOR_res = 5     # LOR resolution = 5 mm
        self.w = width
        self.h = height
        self.size = 1
        self.diameter = 87
        self.SCALE = 1
        self.voxelsize=4    # mm
        self.bckg = 133
        self.kernel0 = np.ndarray(shape=(25, 25))
        self.kernel = np.ndarray(shape=(25,25))
        self.kernel_high_stat = np.ndarray(shape=(25, 25))
        self.data = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.data1 = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.data_emb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.embedded_pos = [] # a position of embedded objects

    def dr_rect(self, par):
        par[0] = int(par[0]*self.w)
        par[1] = int(par[1]*self.h)
        par[2] = int(par[2]*self.size)
        par[3] = int(par[3]*self.size)
        level = (par[4]-0.5)*200 + self.bckg + 40*(par[4]-0.5)/abs((par[4]-0.5))

        for i in range(int(par[0]) - int(par[2]/2), int(par[0]) + int(par[2]/2)):
            for j in range(max(0,int(par[1]) - int(par[3]/2)), min(int(par[1]) + int(par[3]/2),self.h)):
                try:
                    self.data[round(i),round(j)] = np.array([level,level, level])
                except: continue

    def dr_quadr(self,par):
        par.sort()
        #print(par)

    def dr_circle(self,par):
        xc = par[0]*self.w
        yc = par[1]*self.h
        ax_a = par[2]*0.95/2*self.size
        ax_b = par[3]*0.95/2*self.size
        level = (par[4] - 0.5) * 200 + self.bckg + 40 * (par[4] - 0.5) / abs((par[4] - 0.5))

        for i in range(self.w):
            for j in range(self.h):
                try:
                    if ( (i+0.5-xc)**2/ax_a**2 + (j+0.5-yc)**2/ax_b**2 ) < 1:
                        self.data[i, j] = np.array([level, level, level])
                except:
                    continue

    def dr_embedded(self,par):
        xc = int(par[0]*self.w)
        yc = int(par[1]*self.h)
        ax_a = par[2]*4 + 0.01
        ax_b = par[2]*4 + 0.01
        level = (par[3]) * 200 + self.bckg + 40 * (par[3] - 0.5) / abs((par[3] - 0.5))
        if level > 255: level = 255
        for i in range(self.w):
            for j in range(self.h):
                try:
                    if math.sqrt((i - self.w / 2.) ** 2 + (j - self.h / 2.) ** 2) < self.diameter / 2.:
                        if ((i-xc)**2/ax_a**2 + (j-yc)**2/ax_b**2 ) <= 1:
                            self.data[i, j] = np.array([level, level, level])
                            self.embedded_pos.append([i,j])
                            self.data_emb[i, j] = np.array([level, level, level])
                        if ((i-1-xc)**2/ax_a**2 + (j-yc)**2/ax_b**2 ) <= 1 or \
                            ((i-1-xc)**2/ax_a**2 + (j-1-yc)**2/ax_b**2 ) <= 1 or \
                            ((i-xc)**2/ax_a**2 + (j-1-yc)**2/ax_b**2 ) <= 1 or \
                            ((i + 1 - xc) ** 2 / ax_a ** 2 + (j - yc) ** 2 / ax_b ** 2) <= 1 or \
                            ((i + 1 - xc) ** 2 / ax_a ** 2 + (j + 1 - yc) ** 2 / ax_b ** 2) <= 1 or \
                            ((i - xc) ** 2 / ax_a ** 2 + (j + 1 - yc) ** 2 / ax_b ** 2) <= 1:
                            self.data_emb[i, j] = self.data[i, j]
                except:
                    continue

    def clear(self):
        xc, yc = self.w / 2., self.h / 2
        for i in range(int(self.w)):
            for j in range(int(self.h)):
                if math.sqrt((i-xc)**2+(j-yc)**2) <= self.diameter/2.: self.data[i][j] = [self.bckg,self.bckg,self.bckg]
        self.embedded_pos.clear()
        self.data_emb = np.zeros((self.h, self.w, 3), dtype=np.uint8)
#        self.data_emb[:] = 0

    def contour(self,num):
        xc, yc = self.w / 2., self.h / 2
        for i in range(int(self.w)):
            for j in range(int(self.h)):
                if math.sqrt((i-xc)**2+(j-yc)**2) > self.diameter/2.:
                    if num == 0: self.data[i][j][:] = 0
                    if num == 1: self.data1[i][j][:] = 0

    def convolve2D(self, kernel, axis = 0, padding=0, strides=1):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))
        image = self.data[:,:,axis]
        # Gather Shapes of Kernel + Image + Padding
        xKernShape = kernel.shape[0]
        yKernShape = kernel.shape[1]
        xImgShape = image.shape[0]
        yImgShape = image.shape[1]

        # Shape of Output Convolution
        xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
        yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
        output = np.zeros((xOutput, yOutput))

        # Apply Equal Padding to All Sides
        if padding != 0:
            imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
            imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
            #print(imagePadded)
        else:
            imagePadded = image

        # Iterate through image
        for y in range(image.shape[1]):
            # Exit Convolution
      #      if y > image.shape[1] - yKernShape:
      #          break
            # Only Convolve if y has gone down by the specified Strides
            if y % strides == 0:
                for x in range(image.shape[0]):
                    # Go to next row once kernel is out of bounds
             #       if x > image.shape[0] - xKernShape:
             #           break
                    try:
                        # Only Convolve if x has moved by the specified Strides
                        if x % strides == 0:
                            output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                    except:
                        break

        return output

    def gaussian(self,x1,y1,sigmax,sigmay,angle):
        x = x1*math.cos(angle) + y1*math.sin(angle)
        y = -x1*math.sin(angle) + y1*math.cos(angle)
        sigmax = sigmax/self.voxelsize/2.36;
        sigmay = sigmay/self.voxelsize/2.36;
        return math.exp(-(x**2/sigmax**2 + y**2/sigmay**2)/2);

    def kernel_function(self,x,y):
        d = 0;
        for i in range (10):
            d+= self.gaussian(x,y,self.TOF_res,self.LOR_res,2.*math.pi/20*i)
        return d

    def generate_kernel(self):
        ker = np.ndarray(shape=(25, 25))
        for i in range(ker.shape[0]):
            for j in range(ker.shape[1]):
                limx = np.array([i,i+1])-ker.shape[0]/2.
                limy = np.array([j, j + 1]) - ker.shape[1] / 2.
                ker[i][j] = integrate.nquad(self.kernel_function, [[limx[0], limx[1]],[limy[0], limy[1]]])[0]
        ker = ker/(sum(sum(ker))-0.0001)
        return ker

    def draw_kernel(self):
        figure, axis = plt.subplots(1, 1)
        im = axis.imshow(self.kernel0)
        axis.text(1, 1, 'kernel', bbox={'facecolor': 'white', 'pad': 6})
        figure.colorbar(im, ax=axis)
        plt.savefig("kernel.png")
        #plt.plot(self.kernel[int(self.kernel.shape[0]/2) + 1,:],'s')
        plt.show()

    def blurring(self):
        kernel = self.kernel0
        aa = self.convolve2D(kernel, axis=0, padding= int((kernel.shape[0]-1)/2))
        bb = self.convolve2D(kernel, axis=1, padding= int((kernel.shape[0]-1)/2))
        cc = self.convolve2D(kernel, axis=2, padding= int((kernel.shape[0]-1)/2))

        self.data1 = np.stack((aa,bb,cc),axis=2)
        self.data1 = self.data1.astype(np.uint8)

    def randomize(self):
        for i in range(self.w):
            for j in range(self.h):
                con = self.data1[i, j]
                fl = min(np.random.poisson(con[0] * self.SCALE) / self.SCALE, 255)
                fl = max(fl, 0)
                self.data1[i, j] = np.array([fl, fl, fl])

    def plot(self):
        figure, axis = plt.subplots(1, 2)
        image = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        im = axis[0].imshow(image)
        axis[0].text(1, 1, 'label', bbox={'facecolor': 'white', 'pad': 2})
        figure.colorbar(im, ax=axis[0])

        image = cv2.cvtColor(self.data1, cv2.COLOR_BGR2GRAY)
        im = axis[1].imshow(image)
        axis[1].text(1, 1, 'test', bbox={'facecolor': 'white', 'pad': 2})
        figure.colorbar(im, ax=axis[1])

        plt.show()

    def update_embedded(self):
        for el in self.embedded_pos:
            posx = int(el[0])
            posy = int(el[1])
            self.data1[posx][posy]=self.data[posx][posy]


    def savedata(self,number,image_path):
        im = Image.fromarray(self.data)
        im.save(image_path + str(number) + ".png")
        np.savetxt(image_path + str(number) + ".dat",self.embedded_pos,fmt='%i')

    def save(self,number,image_path):
        im = Image.fromarray(self.data1)
        im.save(image_path + str(number) + ".png")

    def save_emb(self,number,image_path):
        im = Image.fromarray(self.data_emb)
        im.save(image_path + str(number) + ".png")


def plot_pattern():
    plot = Toy_fig_1(144, 144)
    plot.bckg = 40
    plot.clear()
    for num in [0,40,80,120,160,200,240,280,325]:
        angle = num/180.*math.pi
        r = 0.2
        x = r*math.cos(angle)+0.5
        y = r*math.sin(angle)+0.5
        plot.size = num/20.+2.
        plot.dr_circle([x,y,0,0,220])
    plot.kernel0 = plot.generate_kernel()
    plot.SCALE = 0.8
    plot.blurring()
    plot.randomize()
    plot.contour(1)
    plot.save(2, "")
    plot.savedata(3, "")
    plot.plot()

def generate_all(N,plotsize,path):
    plot = Toy_fig_1(144, 144)
   # try:
    #    shutil.rmtree(path)
   # except OSError:
    #    print('ciao0')
    try:
        os.mkdir(path)
    except OSError:
        print('ciao1')
    try: os.mkdir(path + 'label_emb/')
    except: pass
    try:
        os.mkdir(path + 'label/')
    except OSError:
        print('ciao2')
    try:
        os.mkdir(path + 'image_high_stat/')
    except OSError:
        print('ciao3')
    try:
        os.mkdir(path + 'image_mixed/')
    except OSError:
        print('ciao4')
    try:
        os.mkdir(path + 'image/')
    except OSError:
        print('ciao5')
    plot.kernel = plot.generate_kernel()
    plot.TOF_res = 4
    plot.kernel_high_stat = plot.generate_kernel()

    for i in range(N):
        print(i)
        plot.clear();
        plot.bckg = 133 + (np.random.rand(1).squeeze() - 0.5) * 70.
        plot.size = plotsize*(np.random.rand(1).squeeze()+0.5)/1.3
        for j in range(int(np.random.rand(1)[0]*400/plotsize)):
            plot.dr_rect(np.random.rand(5))
            plot.dr_quadr(np.random.rand(4, 2))
            plot.dr_circle(np.random.rand(5))
        for j in range(10):
            plot.dr_embedded(np.random.rand(4))

        plot.SCALE = 1000
        plot.kernel0 = plot.kernel_high_stat
        plot.blurring()
        plot.randomize()
        plot.contour(1)
        plot.save(i,path + "image_high_stat/")
        #plot.plot()

        plot.update_embedded()
        plot.contour(1)
        plot.save(i,path+ "image_mixed/")
        #plot.plot()

        plot.SCALE = 10
        plot.kernel0 = plot.kernel
        plot.blurring()
        plot.randomize()
        plot.contour(1)
        plot.save(i,path + "image/")

        plot.contour(0)
        plot.savedata(i, str(path + "label/"))

        plot.save_emb(i, path + "label_emb/")
        #plot.plot()

        #plot.plot()
        # gc.collect()




if __name__ == '__main__':
    #plot_pattern()
    generate_all(20,50,'data/')
    generate_all(20,50,'data/validation/')
    #generate_all(10, 50, 'data_50/validation/')
    #generate_all(2000,20,'data_20/')
    #generate_all(10, 20, 'data_20/validation/')
    #generate_all(2000,10,'data_10/')
    #generate_all(10, 10, 'data_10/validation/')
    #generate_all(2000,2,'data_2/')
    #generate_all(10, 2, 'data_2/validation/')


    #gen.generate_kernel()
    #gen.draw_kernel()

    #plot_pattern()

# import matplotlib.pyplot as plt
# from pylab import imread,subplot,imshow,show
# image = imread('data/label/21.png')
# plt.imshow(image)

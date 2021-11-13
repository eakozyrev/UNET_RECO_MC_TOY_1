import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
import math
import cv2
from scipy import integrate

class Toy_fig_1:
    """This is a base class to generate toy figure type #1"""

    def __init__(self,width, height):
        self.w = width
        self.h = height
        self.diameter = 87
        self.voxelsize=4    # mm
        self.bckg = 133
        self.kernel = np.ndarray(shape=(25,25))
        self.data = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.data1 = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def dr_rect(self, par):
        par[0] = int(par[0]*self.w)
        par[1] = int(par[1]*self.h)
        par[2] = int(par[2]*self.w/3.)
        par[3] = int(par[3]*self.h/3.)
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
        ax_a = par[2]*self.w/7 + 6
        ax_b = par[3]*self.h/7 + 6
        level = (par[4] - 0.5) * 200 + self.bckg + 40 * (par[4] - 0.5) / abs((par[4] - 0.5))

        for i in range(self.w):
            for j in range(self.h):
                try:
                    if ( (i-xc)**2/ax_a**2 + (j-yc)**2/ax_b**2 ) < 1:
                        self.data[round(i), round(j)] = np.array([level, level, level])
                except:
                    continue

    def clear(self):
        xc, yc = self.w / 2., self.h / 2
        for i in range(int(self.w)):
            for j in range(int(self.h)):
                if math.sqrt((i-xc)**2+(j-yc)**2) <= self.diameter/2.: self.data[i][j] = [self.bckg,self.bckg,self.bckg]

    def contour(self):
        xc, yc = self.w / 2., self.h / 2
        for i in range(int(self.w)):
            for j in range(int(self.h)):
                if math.sqrt((i-xc)**2+(j-yc)**2) > self.diameter/2.: self.data[i][j][:] = 0

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

    def gaussian(self,x,y):
        sigma = 45./self.voxelsize/2.36;
        return 1/2./math.pi/sigma**2*math.exp(-(x**2 + y**2)/2/(sigma**2));

    def generate_kernel(self):
        for i in range(self.kernel.shape[0]):
            for j in range(self.kernel.shape[1]):
                limx = np.array([i,i+1])-self.kernel.shape[0]/2.
                limy = np.array([j, j + 1]) - self.kernel.shape[1] / 2.
                self.kernel[i][j] = integrate.nquad(self.gaussian, [[limx[0], limx[1]],[limy[0], limy[1]]])[0]
        self.kernel = self.kernel/sum(sum(self.kernel))

    def draw_kernel(self):
        figure, axis = plt.subplots(1, 1)
        im = axis.imshow(self.kernel)
        axis.text(1, 1, 'kernel', bbox={'facecolor': 'white', 'pad': 6})
        figure.colorbar(im, ax=axis)
        plt.savefig("kernel.png")
        plt.show()

    def blurring(self):
        kernel = self.kernel
        aa = self.convolve2D(kernel, axis=0, padding= int((kernel.shape[0]-1)/2))
        bb = self.convolve2D(kernel, axis=1, padding= int((kernel.shape[0]-1)/2))
        cc = self.convolve2D(kernel, axis=2, padding= int((kernel.shape[0]-1)/2))

        self.data1 = np.stack((aa,bb,cc),axis=2)
        self.data1 = self.data1.astype(np.uint8)

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


    def save(self,number):
        im = Image.fromarray(self.data)
        im.save("data/label/" + str(number) + ".png")
        im = Image.fromarray(self.data1)
        im.save("data/image/" + str(number) + ".png")

def generate_all(N):
    plot = Toy_fig_1(144, 144)
    plot.generate_kernel()
    for i in range(N):
        print(i)
        plot.bckg = 133 + (np.random.rand(1).squeeze() - 0.5) * 70.
        plot.clear();
        for j in range(8):
            plot.dr_rect(np.random.rand(5))
            plot.dr_quadr(np.random.rand(4, 2))
            plot.dr_circle(np.random.rand(5))
        plot.contour()
        plot.blurring()
        plot.save(i)
        # plot.plot()
        # gc.collect()




if __name__ == '__main__':
    generate_all(1)
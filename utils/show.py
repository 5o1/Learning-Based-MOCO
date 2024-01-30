# Plt Functions

from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from matplotlib.colors import CSS4_COLORS,to_rgba
from random import shuffle
import re
import os
from matplotlib.ticker import FormatStrFormatter

COLORS = list(CSS4_COLORS.items())
COLORS = [(name, color) for name, color in COLORS if to_rgba(color)[:-1][2] < 0.7]
shuffle(COLORS)


def _get_shape(n):
    d = np.array([ [j , n%j if n%j != 0 else j ] for j in range(4,0,-1)])
    col = np.min(d[d[:,1] == np.max(d[:,1])][:,0])
    row = np.ceil(n//col).astype(np.int64)
    return (row , col)

def _shape2index(t):
    if t[0] == 1:
        return range(t[1])
    l = [ (i,j) for i in range(t[0]) for j in range(t[1]) ]
    return l

def imshow(I: np.ndarray, titles = None, view = np.abs):

    I = np.array(I)

    if len(I.shape) == 3:
        n = len(I)
        shape = _get_shape(n)
        fig, axes = plt.subplots(*shape,figsize = (16,16) if n == 1 else (12,12))
        index = _shape2index(shape)
        for i, image in enumerate(I):
            if view is not None:
                image = view(image)
            axes[index[i]].imshow(image, cmap = 'gray')
            if titles is not None:
                axes[index[i]].set_title(titles[i])
            axes[index[i]].axis('off')
        plt.subplots_adjust(hspace = -0.65)
    else:
        image = I
        if view is not None:
            image = view(image)
        plt.figure()
        plt.imshow(image, cmap = 'gray')
        if titles is not None:
            plt.title(titles)
        plt.axis('off')


def plot(Y: np.ndarray | list, titles = None, view = None):
    if isinstance(Y, list):
        Y = np.array(Y) 

    if len(Y.shape) == 2:
        lower = min( min(row) for row in Y)
        upper = max( max(row) for row in Y)
        n = len(Y)
        shape = _get_shape(n)
        fig, axes = plt.subplots(*shape,figsize = (16,16) if n == 1 else (12,12//n))
        index = _shape2index(shape)
        for i, y in enumerate(Y):
            if view is not None:
                y = view(y)
            axes[index[i]].plot(np.arange(len(y)),y)
            if titles is not None:
                axes[index[i]].set_title(titles[i])
            axes[i].set_ylim(lower, upper)
        plt.subplots_adjust(hspace = -0.65)  
    else:
        y = Y
        if view is not None:
            y = view(y)
        plt.figure()
        plt.plot(np.arange(len(y)),y)
        if titles is not None:
            plt.title(titles)


def plot_inone(Y: np.ndarray | list, titles = None, view = None):
    lower = min( min(row) for row in Y)
    upper = max( max(row) for row in Y)
    plt.figure(figsize=(16,8))
    for i, y in enumerate(Y):
        if view is not None:
            y = view(y)
        plt.plot(np.arange(len(y)), y, color=COLORS[i][0], 
                    label=titles[i] if titles is not None else f'Line {i}')
        plt.ylim(lower, upper)
    plt.legend()


class Display:
    def __init__(self, image0, title,view= lambda x: Display.norm(np.abs(x))) -> None:
        self.fig, self.ax = plt.subplots()
        self.view = view
        image0 = self.view(image0)
        self.array = self.ax.imshow(image0, cmap='gray')
        self.title_format = title
        self.ax.set_title(title)
        self.ax.set_axis_off()  # Set axis off through the axis object


    def __call__(self, image, title):
        image = self.view(image)
        self.array.set_array(image)
        self.ax.set_title(title)
        display(self.fig)
        clear_output(wait=True)

    @staticmethod
    def norm(x):
        x = np.abs(x)
        lower = np.min(x)
        upper = np.max(x)
        return (x- lower) / (upper - lower)
    

class LossFigure:
    def __init__(self):
        self.pos = 0
        self.fig = plt.figure(figsize=(12,8))

    def __call__(self, x, y, title):
        self.pos = self.pos + 1
        plt.subplot(2,3,self.pos)
        plt.plot(x, y)
        plt.title(title)


    def save(self, path):
        self.fig.savefig(path)

def plot_loss(logpath : str):
    pattern = r".?Epoch\[(\d+)/(\d+)\]\s?([a-zA-Z\s]+):\s?([-+]?\d*\.\d+|\d+).?([a-zA-Z\s]+):\s*([-+]?\d*\.\d+|\d+).?([a-zA-Z\s]+):\s?([-+]?\d*\.\d+|\d+)"
    train = []
    val = []
    with open(logpath, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match is None:
                continue
            x, _, _, ssim, _, psnr, _, loss = match.groups()
            x = int(x)
            ssim = float(ssim)
            psnr = float(psnr)
            loss = float(loss)
            if "Training Results" in line:
                train.append([x,ssim,psnr,loss])
            elif "Validation Results" in line:
                val.append([x,ssim,psnr,loss])
    pngpath = os.path.join(os.path.split(logpath)[0],'loss.png')
    p = LossFigure()
    xs = [train[i][0] for i in range(len(train))]
    ssim = [train[i][1] for i in range(len(train))]
    psnr = [train[i][2] for i in range(len(train))]
    loss = [train[i][3] for i in range(len(train))]
    p(xs,ssim,'train SSIM')
    p(xs,psnr,'train PSNR')
    p(xs,loss,'train LOSS')
    xs = [val[i][0] for i in range(len(val))]
    ssim = [val[i][1] for i in range(len(val))]
    psnr = [val[i][2] for i in range(len(val))]
    loss = [val[i][3] for i in range(len(val))]
    p(xs,ssim,'val SSIM')
    p(xs,psnr,'val PSNR')
    p(xs,loss,'val LOSS')
    p.save(pngpath)

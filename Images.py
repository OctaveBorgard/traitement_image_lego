#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from Couleurs import *


# In[1]:


def Fourrier(image) :
    tf = np.fft.fftshift(np.fft.fft2(image))
    return tf


# In[6]:


def passe_bas(tf2, size_init, precision) :
    dimx, dimy = size_init[0], size_init[1]
    tf_c = np.copy(tf2)
    mx = dimx//2
    my = dimy//2
    for l in range(dimx) :
        if l<mx-mx/precision or l> mx + mx/precision:
            tf_c[l, :] = 0
    for c in range(dimy) :
        if c<my-my/precision or c> my + my/precision:
            tf_c[:, c] = 0
    return tf_c

def img_resize(image, precision) :
    dimx, dimy = image.shape
    new_tf = passe_bas(Fourrier(image), [dimx, dimy], precision)
    new_image = np.real(np.fft.ifft2(np.fft.ifftshift(new_tf)))
    dx, dy = image.shape
    n_img = np.zeros((dx//precision, dy//precision))
    for i in range(dx//precision) :
        for k in range(dy//precision) :
            n_img[i, k] = new_image[i*precision, k*precision]
    return n_img

def color_img_resize(image, precision) : 
    new_r = img_resize(image[:,:,0], precision)
    new_b = img_resize(image[:,:,2], precision)
    new_g = img_resize(image[:,:,1], precision)
    return rgb_to_image(new_r, new_g, new_b)


# In[5]:


def passe_bas2(tf2, size_init, size) :
    dimx, dimy = size_init[0], size_init[1]
    mx = dimx//2
    my = dimy//2
    n_tf = tf2[mx-size[0]//2:mx + size[0]//2, my-size[1]//2:my + size[1]//2]
    return n_tf

def compress_resize(image, size) :
    img = np.real(np.fft.ifft2(np.fft.ifftshift(passe_bas2(Fourrier(image), image.shape, size))))
    img = img/(image.shape[0]*image.shape[1])
    img = img*(img.shape[0]*img.shape[1])
    return img

def compressed_image(image, size) :
    new_r2 = compress_resize(image[:,:,0], size)
    new_b2 = compress_resize(image[:,:,2], size)
    new_g2 = compress_resize(image[:,:,1], size)
    return rgb_to_image(new_r2, new_g2, new_b2)


# In[ ]:


def image_to_pixel(image) :
    return image.reshape(1,image.shape[0]*image.shape[1], 3)[0]

def pixel_to_image(pix, size) :
    pix.resize(size)


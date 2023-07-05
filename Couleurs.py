#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def int_tab(mat) :
    return np.ndarray.astype(np.around(mat), int)
def chroma_point(r, g, b) :
    metrique = np.array([[0.2126, 0.7152, 0.0722],
                        [-0.1146, -0.3854, 0.5],
                        [0.5, -0.4541, -0.0458]])
    vect = metrique.dot([r, g, b])
    return vect[0], vect[1], vect[2]

def chroma(pixel) :
    c = np.zeros(pixel.shape)
    i = 0
    for p in pixel :
        y, r, b = chroma_point(p[0], p[1], p[2])
        c[i] = [y, r, b]
        i+=1
    return c

def rgb_point(y, cb, cr) :
    metrique = np.array([[0.999843, -0.000122007, 1.57484],
                       [1.00005, -0.187289, -0.468147],
                       [1, 1.85561, 0.000105745]])
    vect = metrique.dot([y, cb, cr])
    return vect[0], vect[1], vect[2]

def rgb(pixel) :
    vect = np.zeros(pixel.shape)
    i=0
    for p in pixel :
        r, g, b = rgb_point(p[0], p[1], p[2])
        vect[i] = [r, g, b]
        i+=1
    return int_tab(vect)


# In[9]:


def rgb_to_image(r, g, b) :
    img = np.zeros((r.shape[0], r.shape[1], 3))
    if r.max()-3 > 1 :
        for i in range (img.shape[0]) :
            for j in range(img.shape[1]) :
                img[i, j] = [max(0, min(255, r[i, j])), max(0, min(255, g[i, j])), max(0, min(255, b[i, j]))]
    else :
        for i in range (img.shape[0]) :
            for j in range(img.shape[1]) :
                img[i, j] = [max(0, min(255, r[i, j]*255)), max(0, min(255, g[i, j]*255)), max(0, min(255, b[i, j]*255))]
    #print(img_finale.max())
    return int_tab(img)

def image_to_rgb(image) :
    return image[:, :, 0], image[:, :, 1], image[:, :, 2]


# In[14]:


def img_int(image) : 
    img = np.zeros((image.shape[0], image.shape[1], 3))
    if image.max()-3 > 1 :
        for i in range (img.shape[0]) :
            for j in range(img.shape[1]) :
                img[i, j] = [max(0, min(255, image[i, j, 0])), max(0, min(255, image[i, j, 1])), max(0, min(255, image[i, j,2]))]
    else :
        for i in range (img.shape[0]) :
            for j in range(img.shape[1]) :
                img[i, j] = [max(0, min(255, image[i, j,0]*255)), max(0, min(255, image[i, j,1]*255)), max(0, min(255, image[i, j,2]*255))]
    return int_tab(img)

def img_int_BW(image) : 
    img = np.zeros((image.shape[0], image.shape[1]))
    if image.max()-3 > 1 :
        for i in range (img.shape[0]) :
            for j in range(img.shape[1]) :
                img[i, j] = max(0, min(255, image[i, j]))
    else :
        for i in range (img.shape[0]) :
            for j in range(img.shape[1]) :
                img[i, j] = max(0, min(255, image[i, j]*255))
    return int_tab(img)


# In[1]:


def couleur_majoritaire(p) :
    d = {}
    num_p = {}
    for i in p : 
        isin = -1
        for k in num_p.keys() :
            if (num_p[k] == i) :
                isin = k
        if isin != -1 :
            d[isin] += 1
        else :
            x = int(len(num_p))
            num_p[x] = i
            d[x] = 1
    maxk = 0
    for k in d.keys() :
        if d[maxk] < d[k] :
            maxk = k
    return num_p[maxk]


# In[ ]:


def sat(r, g, b) :
    y, cb, cr = chroma_point(r, g, b)
    if y >85 :
        y+=20
    if cb>0:
        cb += 10
    if y<85 :
        y-=20
    r2, g2, b2 = rgb_point(y, cb, cr)
    return r2, g2, b2


def lissage_point(p, contour) :
    nb_similaire = 0
    contour = [np.ndarray.tolist(contour[i]) for i in range(len(contour))]
    nbc=0
    for c in contour :
        #print(c, p, all(c == p))
        if all(c == p) :
            nb_similaire += 1
    if nb_similaire <= 1 :
        x = np.zeros(len(contour))
        it = 0
        for i in contour :
            x[it] = contour.count(i)
            it+=1
        #print(x)
        p = contour[x.argmax()]
    return p

def lissage(image) :
    """Cette fonction de lissage n'est pas linéaire !"""
    image2 = np.copy(image)
    for i in range(0,image.shape[0]) :
        for j in range(0,image.shape[1]) :
            if i == 0 and j==0 :
                image2[i, j] = lissage_point(image[i, j], [image[i+1, j], image[i+1, j+1], image[i, j+1]])
            elif i == 0 and j == image.shape[1]-1 :
                image2[i, j] = lissage_point(image[i, j], [image[i, j-1],image[i+1, j-1], image[i+1, j]])
            elif i==image.shape[0]-1 and j==0 :
                image2[i, j] = lissage_point(image[i, j], [image[i, j+1], image[i-1, j+1], image[i-1, j]])
            elif i==image.shape[0]-1 and j == image.shape[1]-1 :
                image2[i, j] = lissage_point(image[i, j], [image[i, j-1], image[i-1, j], image[i-1, j-1]])
            elif i==0 :
                image2[i, j] = lissage_point(image[i, j], [image[i, j-1],image[i+1, j-1], image[i+1, j], image[i+1, j+1], image[i, j+1]])
            elif i==image.shape[0]-1 :
                image2[i, j] = lissage_point(image[i, j], [image[i, j-1], image[i, j+1], image[i-1, j+1], image[i-1, j], image[i-1, j-1]])
            elif j==0:
                image2[i, j] = lissage_point(image[i, j], [image[i+1, j], image[i+1, j+1], image[i, j+1], image[i-1, j+1], image[i-1, j]])
            elif j == image.shape[1]-1 :
                image2[i, j] = lissage_point(image[i, j], [image[i, j-1],image[i+1, j-1], image[i+1, j], image[i-1, j], image[i-1, j-1]])
            else :
                image2[i, j] = lissage_point(image[i, j], [image[i, j-1],image[i+1, j-1], image[i+1, j], image[i+1, j+1], image[i, j+1], image[i-1, j+1], image[i-1, j], image[i-1, j-1]])
    return image2


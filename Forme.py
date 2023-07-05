#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import cv2

from Images import *
from Couleurs import *

def scalaire(D, T) :
    return np.trace(np.transpose(D).dot(T))

def n2(A) :
    #return np.linalg.norm(A, 2)
    s = 0
    for i in range(A.shape[0]) :
        for j in range(A.shape[1]) :
            s+= A[i, j]**2
    return s

def rond(r, g, b, size, saturation, r2, g2, b2) :
    rayon = size//2
    img_rond = np.ones((size, size, 3))
    if saturation :
        r, g, b = sat(r, g, b)
    for i in range(size) :
        for j in range(size) :
            if np.linalg.norm(np.array([rayon, rayon])-np.array([i, j]))<=rayon :
                img_rond[i,j] = [r, g, b]
            else :
                img_rond[i,j] = [r2, g2, b2]
    return int_tab(img_rond)


def triangle_isocele(r, g, b, size, saturation, r2, g2, b2) :
    img_triangle = np.zeros((size, size,3))
    for i in range(size) :
        for j in range(size) :
            if i+j < size :
                img_triangle[i,j] = [r, g, b]
            else :
                img_triangle[i,j] = [r2, g2, b2]
    return int_tab(img_triangle)

def petit_triangle(r, g, b, size, saturation, r2, g2, b2) :
    img_triangle = np.zeros((size, size,3))
    for i in range(size) :
        for j in range(size) :
            if i+j < size-size//2.5 :
                img_triangle[i,j] = [r, g, b]
            else :
                img_triangle[i,j] = [r2, g2, b2]
    return int_tab(img_triangle)


def rectangle(r, g, b, size, saturation, r2, g2, b2) :
    img_rect = np.zeros((size, size,3))
    for i in range(size) :
        for j in range(size) :
            if size//2-size//3  < i < size//2+size//3 :
                img_rect[i,j] = [r, g, b]
            else :
                img_rect[i,j] = [r2, g2, b2]
    return int_tab(img_rect)

def demi_rectangle(r, g, b, size, saturation, r2, g2, b2) :
    img_rect = np.zeros((size, size,3))
    for i in range(size) :
        for j in range(size) :
            if i < size//2 :
                img_rect[i,j] = [r, g, b]
            else :
                img_rect[i,j] = [r2, g2, b2]
    return int_tab(img_rect)


def rotation(image) :
    return np.rot90(image)


# In[2]:


def rond_contour(bw1, size, bw2) :
    rayon = size//2
    img_rond = np.ones((size, size))
    for i in range(size) :
        for j in range(size) :
            if rayon-1<np.linalg.norm(np.array([rayon, rayon])-np.array([i, j]))<=rayon+1 :
                img_rond[i,j] = bw2
            else :
                img_rond[i,j] = bw1
    return int_tab(img_rond)

def triangle_isocele_contour(bw1, size, bw2) :
    img_triangle = np.zeros((size, size))
    for i in range(size) :
        for j in range(size) :
            if i+j == size :#or i== 0 or j ==0:
                img_triangle[i,j] = bw2
            else :
                img_triangle[i,j] = bw1
    return int_tab(img_triangle)


def petit_triangle_contour(bw1, size, bw2) :
    img_triangle = np.zeros((size, size))
    for i in range(size) :
        for j in range(size) :
            if i+j == size-size//2.5 :#or (i== 0 and j<size-size//2.5) or (j ==0 and i<size-size//2.5) :
                img_triangle[i,j] = bw2
            else :
                img_triangle[i,j] = bw1
    return int_tab(img_triangle)


def rectangle_contour(bw1, size, bw2) :
    img_rect = np.zeros((size, size))
    for i in range(size) :
        for j in range(size) :
            if size//2-size//3  == i or i == size//2+size//3 or (j==0 and size//2-size//3<i<size//2+size//3) or (j==size-1 and size//2-size//3<i<size//2+size//3):
                img_rect[i,j] = bw2
            else :
                img_rect[i,j] = bw1
    return int_tab(img_rect)

def demi_rectangle_contour(bw1, size, bw2) :
    img_rect = np.zeros((size, size))
    for i in range(size) :
        for j in range(size) :
            if i == size//2 or (j==0 and i<size//2) or (j==size-1 and i<size//2):
                img_rect[i,j] = bw2
            else :
                img_rect[i,j] = bw1
    return int_tab(img_rect)

def triangleplus_contour(num) :
    a = np.zeros((16, 16))
    size = 16
    for i in range(15):
        for j in range(15) :
            if i+j == size+num :
                a[i, j] = 255
                a[i, j-1] = 100
                a[i-1, j] = 100
                a[i, j+1] = 100
                a[i+1, j] = 100
    return a

def rotation(image) :
    return np.rot90(image)


def petite_piece(size) :
    dl = []
    #ronds
    dl.append(rond_contour(0,size, 255))
    #print(len(dl))
    norme = []
    for i in range(len(dl)) :
        #plt.imshow(dl[i], cmap="Greys")
        #plt.show()
        norme.append(n2(dl[i]))
        dl[i] = dl[i]/n2(dl[i])
    return dl, norme

def grande_piece(size) :
    dl = []
    #triangles
    dl.append(triangle_isocele_contour(0, size, 255))
    for j in range(3) :
        dl.append(rotation(dl[-1]))
    dl.append(petit_triangle_contour(0, size, 255))
    for j in range(3) :
        dl.append(rotation(dl[-1]))
    #rectangle
    dl.append(rectangle_contour(0, size, 255))
    dl.append(rotation(dl[-1]))
    dl.append(demi_rectangle_contour(0, size, 255))
    for j in range(3) :
        dl.append(rotation(dl[-1]))
    for i in range(1, 8) :
        dl.append(triangleplus_contour(i))
    for i in range(1, 8) :
        dl.append(rotation(triangleplus_contour(i)))
    norme = []
    #rectangle plus
    for i2 in range(5, 12) :
        a = np.zeros((16, 16))
        size = 16
        for i in range(16):
            for j in range(16) :
                if i == size-i2 :
                    a[i, j] = 255
                    a[i+1, j] = 100
                    a[i-1, j] = 100
        dl.append(a)
    for i2 in range(5, 12) :
        a = np.zeros((16, 16))
        size = 16
        for i in range(16):
            for j in range(16) :
                if i == size-i2 :
                    a[i, j] = 255
                    a[i+1, j] = 100
                    a[i-1, j] = 100
        dl.append(rotation(a))
    

    for z in range(4) :
        a = np.zeros((16, 16))
        size = 16
        for i in range(16):
            for j in range(16) :
                if i == size-10-z :
                    a[i, j] = 255
                    a[i+1, j] = 100
                    a[i-1, j] = 100
                if i == size - 3-z :
                    a[i, j] = 255
                    a[i+1, j] = 100
                    a[i-1, j] = 100
        dl.append(a)
    for z in range(4) :
        a = np.zeros((16, 16))
        size = 16
        for i in range(16):
            for j in range(16) :
                if i == size-10-z :
                    a[i, j] = 255
                    a[i+1, j] = 100
                    a[i-1, j] = 100
                if i == size - 3-z :
                    a[i, j] = 255
                    a[i+1, j] = 100
                    a[i-1, j] = 100
        dl.append(rotation(a))
        
    for i in range(1, 8) :
        dl.append(triangleplus_contour(-i))
    for i in range(1, 8) :
        dl.append(rotation(triangleplus_contour(-i)))
    for i in range(len(dl)) :
        norme.append(n2(dl[i]))
        dl[i] = dl[i]/n2(dl[i])
    return dl, norme


def deco_lego(size) :
    """Un dictionnaire de piece lego de même taille"""
    dl = []
    #ronds
    dl.append(rond_contour(0,size, 255))
    print(len(dl))
    #triangles
    dl.append(triangle_isocele_contour(0, size, 255))
    for j in range(3) :
        dl.append(rotation(dl[-1]))
    print(len(dl))
    dl.append(petit_triangle_contour(0, size, 255))
    for j in range(3) :
        dl.append(rotation(dl[-1]))
    print(len(dl))
    #rectangle
    dl.append(rectangle_contour(0, size, 255))
    dl.append(rotation(dl[-1]))
    print(len(dl))
    dl.append(demi_rectangle_contour(0, size, 255))
    for j in range(3) :
        dl.append(rotation(dl[-1]))
    print(len(dl))
    dl.append(np.zeros((size, size)))
    norme = []
    for i in range(len(dl)) :
        norme.append(n2(dl[i]))
        if i != len(dl)-1 :
            dl[i] = dl[i]/n2(dl[i])
    return dl, norme


# In[3]:


def canny(path) :
    imgcv2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)#img_clust2_256x384.jpg
    assert imgcv2 is not None, "file could not be read, check with os.path.exists()"
    edges = cv2.Canny(imgcv2,100,300)
    plt.subplot(121), plt.imshow(imgcv2,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges,cmap = 'gray')
    plt.title('Canny Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    return edges

def debruitage(edges) :
    img_copi = np.zeros((edges.shape[0], edges.shape[1]))
    for i in range(edges.shape[0]) :
        for j in range(edges.shape[1]) :
            if edges[i, j].max() > 40 :
                img_copi[i, j] = np.mean(edges[i, j])#255
            else :
                img_copi[i, j] = 0
    plt.imshow(img_copi, cmap = "Greys")
    return img_copi

def grande_forme(img, dict_contour, sensibilite = 80):
    img_copi2 = np.copy(img)
    for k in dict_contour.keys() :
        if len(dict_contour[k]) < sensibilite:
            for p in dict_contour[k] :
                img_copi2[p[0], p[1]] = 0
    return img_copi2

def contour(path, sensibilite = 80) :
    debruit = debruitage(canny(path))
    edges = grande_forme(debruit, contour_grand(debruit), sensibilite) 
    return edges


# In[4]:


def contour_grand(image2) :
    image = np.zeros((image2.shape[0]+2, image2.shape[1]+2))
    image[1:image.shape[0]-1, 1:image.shape[1]-1] = image2
    d = {}
    num = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]) :
            if image[i, j] != 0 :
                isnot = True
                for k in d.keys() :
                    if [i, j] in d[k] :
                        isnot = False
                if isnot :
                    d[num] = [[i, j]] + rec_contour(image, [i, j], [])
                    num += 1
    for k in d.keys() :
        for cp in d[k] :
            cp[0] -=1
            cp[1] -=1
    return d
        
def rec_contour(image, cp, l) :
    i, j = cp[0], cp[1]
    prec = 1000
    #print(image[i, j-1] != 0)
    if image[i][j-1] != 0 and [i, j-1] not in l and len(l) < prec:
        l.append([i, j-1])
        l = rec_contour(image, [i, j-1], l)
    if image[i+1][j-1] != 0 and [i+1, j-1] not in l and len(l) < prec:
        l.append([i+1, j-1])
        l = rec_contour(image, [i+1, j-1], l)
    if image[i+1][j] != 0 and [i+1,j] not in l and len(l) < prec:
        l.append([i+1, j])
        l = rec_contour(image, [i+1, j], l)
    if image[i+1][j+1] != 0 and [i+1, j+1] not in l and len(l) < prec:
        l.append([i+1, j+1])
        l = rec_contour(image, [i+1, j+1], l)   
    if image[i][j+1] != 0 and [i,j+1] not in l and len(l) < prec:
        l.append([i, j+1])
        l = rec_contour(image, [i, j+1], l)
    if image[i-1][j+1] != 0 and [i-1, j+1] not in l and len(l) < prec:
        l.append([i-1, j+1])
        l = rec_contour(image, [i-1, j+1], l)
    if image[i-1][j] != 0 and [i-1,j] not in l and len(l) < prec:
        l.append([i-1, j])
        l = rec_contour(image, [i-1, j], l)
    if image[i-1][j-1] != 0 and [i-1,j-1] not in l and len(l) < prec:
        l.append([i-1, j-1])
        l = rec_contour(image, [i-1, j-1], l)
    #print(len(l))
    return l


# In[ ]:





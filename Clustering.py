#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
from Images import *
from Couleurs import *


# In[ ]:


def groupe(centre, pixel) :
    """centre correspond aux coordonées de mes clusters et pixel aux coordonées de mes pixels dans le plan dans lequel est défini pixel.
    On va calculer quel pixel va avec quel cluster"""
    classe = {}
    for c in range(len(centre)) :
        classe[c] = []
    for p in pixel :
        dist_c = (p-centre)
        dist=np.ones(len(centre))
        for k in range(len(centre)) :
            dist[k] = np.linalg.norm(dist_c[k, :])
        #print(dist)
        classe[np.argmin(dist)].append(p)
    return classe

def ajust(centre, clust, pixel) :
    new_centre = np.zeros((len(centre), 3))
    for k in clust.keys() :
        if clust[k] != [] :
            for color in range(3) :
                new_centre[k, color] = np.sum(np.array(clust[k])[:,color])/len(clust[k])
        else :
            print("Un cluster vide")
            new_centre[k] = pixel[random.randint(0,len(pixel)-1)]
    return new_centre

def ajustement(pixel, k) :
    l_centre = []
    centre = np.zeros((k, 3))
    for i in range(k) :
        random_index = random.randint(0,len(pixel)-1)
        centre[i] = pixel[random_index]
    iteration = 0
    while iteration == 0 or (np.linalg.norm(l_centre[-1] - centre ) > 1e-1 and iteration < 500) :
        l_centre.append(centre)
        print("ajust num ", iteration)
        g=groupe(centre, pixel)
        #print(g.keys(), "g")
        centre = ajust(centre, g, pixel)
        iteration += 1
    l_centre.append(centre)
    return l_centre, iteration, centre

def cluster(pixel, k) :
    return ajustement(pixel, k)[2]


# In[ ]:


def Visualisation_cluster(x, pixel):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    
    ax = plt.figure().add_subplot(projection='3d')
    group = groupe(x, pixel)
    
    for k in group.keys() :
        color = (x[k][0]/255, x[k][1]/255, x[k][2]/255)
        print("couleur du cluster : ", color)
        for p in group[k] :
            ax.scatter(p[0], p[1], p[2], c = [color])
    ax.set_xlabel('r')
    ax.set_ylabel('g')
    ax.set_zlabel('b')


# In[1]:


def cluster_to_image(img, centre, group, pixel):
    pix = [np.ndarray.tolist(pixel[i]) for i in range(len(pixel))]
    new_img2 = np.zeros(np.shape(img))
    nb_l, nb_c, nb_couleur = np.shape(img)
    g2={}
    for k in group.keys() :
        g2[k] = [np.ndarray.tolist(group[k][i]) for i in range(len(group[k]))]
    color= []
    for k in group.keys() :
        color.append((centre[k][0]/255, centre[k][1]/255, centre[k][2]/255))
    print("fin de set up")
    
    for l in range(nb_l) :
        for c in range(nb_c) :
            for k in group.keys():
                if pix[nb_c*l+c] in g2[k] :                
                    new_img2[l,c,:] = [color[k][0], color[k][1], color[k][2]]
    return new_img2

def ClustImage(image, k) :
    pix = image_to_pixel(image)
    centre = cluster(pix, k)
    return cluster_to_image(image, centre, groupe(centre, pix), pix)


# In[3]:


def opt_cluster(pixel, k) :
    l=[]
    for i in range(7) :
        l.append(cluster(pixel, k))
    liste=[]
    for i in range(len(l)) :
        #print(l[i][0])
        for j in range(len(l[i])) :
            liste.append(l[i][j])
    occurence = {"init" : 0}
    num_clust = {}
    num=0
    for c in liste :
        if "init" in occurence.keys() :
            num_clust[num] = c
            occurence[num] = 1
            num += 1
            del occurence["init"]
        else :
            deja = False
            for k in occurence.keys() :
                if np.max(np.abs(num_clust[k] - c)) < 20 :
                    #print("déjà")
                    occurence[k] += 1
                    deja = True
            if not deja :
                num_clust[num] = c
                occurence[num] = 1
                num += 1
    l_occ = []
    for i in occurence.keys() :
        l_occ.append(occurence[i])
    topk = []
    for i in range(k) :
        topk.append(np.argmax(l_occ))
        #print(i, topk)
        l_occ[topk[i]] =0
    centre = []
    for i in range(k) :
        centre.append(num_clust[topk[i]])
    return centre


# In[4]:


def ClustImage2(image, k) :
    pix = image_to_pixel(image)
    centre = opt_cluster(pix, k)
    return cluster_to_image(image, centre, groupe(centre, pix), pix)


# In[ ]:





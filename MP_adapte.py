#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Images import *
from Couleurs import *
from Forme import *


# In[2]:


def similarity_complet(image16, dico8, dico16) :
    max_v4 = 0
    w_tot =[]
    index_tot = []
    
    grad = 0
    dico = dico16
    for i in range(image16.shape[0]) :
        for j in range(image16.shape[1]) :
            grad += image16[i, j]
            
    for i in range(2):
        for j in range(2) :
            dico = dico8
            img8=image16[i*8:(i+1)*8, j*8:(j+1)*8]
            reste = np.copy(img8)
            reste.astype(float)
            index = []
    
            w=np.zeros(len(dico))
            vraissemblance = np.zeros(len(dico))
            it=0
            for d in dico :
                vraissemblance[it] = scalaire(d, reste)
                it += 1
            ind = np.argmax(np.abs(vraissemblance))
            index.append(ind)
            max_v4 += vraissemblance[ind]
            w[ind] += scalaire(dico[ind], reste)
            reste = reste - scalaire(dico[ind], reste)*dico[ind]
            w_tot.append(w)
            index_tot.append(index)
    max_v4 = max_v4/4
    
    dico = dico16
    reste = np.copy(image16)
    reste.astype(float)
    index2 = []
    w2=np.zeros(len(dico))
    vraissemblance = np.zeros(len(dico))
    it=0
    for d in dico :
        vraissemblance[it] = scalaire(d, reste)
        it += 1
    ind = np.argmax(np.abs(vraissemblance))
    index2.append(ind)
    w2[ind] += scalaire(dico[ind], reste)
    reste = reste - scalaire(dico[ind], reste)*dico[ind]
    if vraissemblance[ind] >= max_v4 and grad>3000:#1000 :
        return w2, index2[0], 1
    else :
        return w_tot, index_tot, 0
        
def mp_adapt(image) :
    taille = 16
    di8, norme8 = petite_piece(8)
    di16, norme16 = grande_piece(16)
    nimage = np.zeros(image.shape)
    t = 0
    ind = []
    for i in range(image.shape[0]//taille) :
        for j in range(image.shape[1]//taille) :
            t+=1
            w, index, num = similarity_complet(image[i*taille:(i+1)*taille, j*taille:(j+1)*taille], di8, di16)
            nimg = np.zeros((16,16))
            if num == 0 :
                x, y =0, 0
                for i2 in range(len(w)) :
                    for z in range(len(di8)):
                        if z in index[i2] :
                            nimg[x*8:x*8+8, y*8:y*8+8] += w[i2][z]*di8[z]*norme8[z]*1000
                            y+=1
                            if y ==2 :
                                y = 0
                                x +=1
                ind.append(index)
            elif num == -1 :
                nimg += w[-1]*di16[-1]*norme16[-1]
                ind.append([index+len(di8)])
            else :
                for z in range(len(di16)):
                    nimg += w[z]*di16[z]*norme16[z]*1000
                ind.append([index+len(di8)])
            nimg = img_int_BW(nimg)
            
            nimage[i*taille:(i+1)*taille, j*taille:(j+1)*taille] = nimg
    return nimage, ind


# In[ ]:





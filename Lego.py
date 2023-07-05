#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

from Images import *
from Couleurs import *
from Clustering import *
from Forme import *
from MP_adapte import *


# In[4]:


def lego(img10, img2, size_piece, size_lego, saturation) :
    dx, dy, c = img10.shape
    plaque = np.zeros((size_lego[0]*size_piece, size_lego[1]*size_piece, 3))    
    for i in range(dx) :
        for j in range(dy):
            r2, g2, b2 = img2[i,j,:]
            plaque[i*size_piece:(1+i)*size_piece, j*size_piece:(j+1)*size_piece] = rond(img10[i, j, 0], img10[i, j, 1], img10[i, j, 2], size_piece, saturation, r2, g2, b2)
    return img_int(plaque)


# In[3]:


def lego2(c, image_clust, img_clust2) :
    size = 16
    nimg, d = mp_adapt(c)
    plt.imshow(nimg, cmap="Greys")
    plt.title("Formes lego")
    plt.show()
    
    image_lego = np.ones(image_clust.shape)
    couleur2 = np.copy(img_clust2)
    saturation = False
    for i in range(image_clust.shape[0]//size) :
        for j in range(image_clust.shape[1]//size) :
            if len(d[i*image_clust.shape[1]//size+j]) != 1 :
                imgtemp = np.zeros((size, size, 3))
                x, y = 0, 0
                for k in d[i*image_clust.shape[1]//size+j] :
                    l=[]
                    l2 = []
                    rayon = 4
                    for line in range(8) :
                        for col in range(8) :
                            if np.linalg.norm(np.array([rayon, rayon])-np.array([line, col]))<=rayon :
                                l.append(np.ndarray.tolist(image_clust[i*size+x+line, j*size+y+col]))
                            else :
                                l2.append(np.ndarray.tolist(img_clust2[i*size+x+line, j*size+y+col]))
                    cmaj =couleur_majoritaire(l)
                    r, g, b = cmaj[0], cmaj[1], cmaj[2]
                    cmaj =couleur_majoritaire(l2)
                    r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                    imgtemp[x:x+size//2, y:y+size//2, :] = img_int(rond(r, g, b, size//2, saturation, r2, g2, b2))
                    y += size//2
                    if y == size :
                        y = 0
                        x+= size//2
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = imgtemp
            elif d[i*image_clust.shape[1]//size+j][0] == 1 or 15<=d[i*image_clust.shape[1]//size+j][0]<=19 or 51<=d[i*image_clust.shape[1]//size+j][0]<=55 :
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if line+col < size :
                            for zf in range(32-(line+col)*2) :
                                l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            for zf in range((line+col)*2) :
                                l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = triangle_isocele(r, g, b, size, saturation, r2, g2, b2)
            elif d[i*image_clust.shape[1]//size+j][0] == 2 or 22<=d[i*image_clust.shape[1]//size+j][0]<=25 or 58<=d[i*image_clust.shape[1]//size+j][0]<=62 :
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if not (line-col < 0) :
                            for zf in range(abs(line-col)*2) :
                                l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            for zf in range(32-abs(line-col)*2) :
                                l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rotation(triangle_isocele(r, g, b, size, saturation, r2, g2, b2))
            elif d[i*image_clust.shape[1]//size+j][0] == 3 :
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if line+col > size :
                            for zf in range((line+col)*2) :
                                l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            for zf in range(32-(line+col)*2) :
                                l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rotation(rotation(triangle_isocele(r, g, b, size, saturation, r2, g2, b2)))
            elif d[i*image_clust.shape[1]//size+j][0] == 4 :
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if line-col < 0 :
                            for zf in range(32-abs(line-col)*2) :
                                l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            for zf in range(abs(line-col)*2) :
                                l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rotation(rotation(rotation(triangle_isocele(r, g, b, size, saturation, r2, g2, b2))))
            #"petit triangle"
            elif d[i*image_clust.shape[1]//size+j][0] == 5 or 56<=d[i*image_clust.shape[1]//size+j][0]<=57 :
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if line+col < size-size//2.5 :
                            for zf in range(50-(line+col)*2) :
                                l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            for zf in range((line+col)*2) :
                                l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = petit_triangle(r, g, b, size, saturation, r2, g2, b2)
            elif d[i*image_clust.shape[1]//size+j][0] == 6 or 63<=d[i*image_clust.shape[1]//size+j][0]<=64 :
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if -line+col < -16+(16//2.5)*2 :
                            for zf in range(abs(line-col)*2) :
                                l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            for zf in range(32-abs(line-col)*2) :
                                l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rotation(petit_triangle(r, g, b, size, saturation, r2, g2, b2))
            elif d[i*image_clust.shape[1]//size+j][0] == 7 or 20<=d[i*image_clust.shape[1]//size+j][0]<=21:
                size = 16
                
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if line+col > size+size//2.5 :
                            for zf in range(64-(line+col)*2) :
                                l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            for zf in range((line+col)*2) :
                                l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rotation(rotation(petit_triangle(r, g, b, size, saturation, r2, g2, b2)))
            elif d[i*image_clust.shape[1]//size+j][0] == 8 or 26<=d[i*image_clust.shape[1]//size+j][0]<=28:
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if -line+col >-size//2+(size//(2.5)*2) :
                            for zf in range(32-abs(line-col)*2) :
                                l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            for zf in range(abs(line-col)*2) :
                                l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                
                      
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rotation(rotation(rotation(petit_triangle(r, g, b, size, saturation, r2, g2, b2))))
            #"rectangle"
            elif d[i*image_clust.shape[1]//size+j][0] == 9 or 43<=d[i*image_clust.shape[1]//size+j][0]<=46:
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if size//2-size//3  < line < size//2+size//3 :
                            l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rectangle(r, g, b, size, saturation, r2, g2, b2)
            elif d[i*image_clust.shape[1]//size+j][0] == 10 or 47<=d[i*image_clust.shape[1]//size+j][0]<=50:
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if size//2-size//3  < col < size//2+size//3 :
                            l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rotation(rectangle(r, g, b, size, saturation, r2, g2, b2))
            #"demi rectangle"
            elif d[i*image_clust.shape[1]//size+j][0] == 11 or 29<=d[i*image_clust.shape[1]//size+j][0]<=35 :
                size = 16                
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if line < size//2 :
                            l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = demi_rectangle(r, g, b, size, saturation, r2, g2, b2)
            elif d[i*image_clust.shape[1]//size+j][0] == 12 or d[i*image_clust.shape[1]//size+j][0] == 15-1 or 36 <=d[i*image_clust.shape[1]//size+j][0]<=42 :
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if col < size//2 :
                            l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = rotation(demi_rectangle(r, g, b, size, saturation, r2, g2, b2))
            elif d[i*image_clust.shape[1]//size+j][0] == 13 :
                size = 16
                l=[]
                l2 = []
                for line in range(16) :
                    for col in range(16) :
                        if line < size//2 :
                            l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                        else :
                            l2.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                cmaj =couleur_majoritaire(l)
                r, g, b = cmaj[0], cmaj[1], cmaj[2]
                cmaj =couleur_majoritaire(l2)
                r2, g2, b2 = cmaj[0], cmaj[1], cmaj[2]
                
                image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = demi_rectangle(r, g, b, size, saturation, r2, g2, b2)
            
            elif d[i*image_clust.shape[1]//size+j][0] == 65 :
                l=[]
                for line in range(16) :
                    for col in range(16) :
                        l.append(np.ndarray.tolist(image_clust[line+(i)*size, j*size+(col)]))
                
                coul_maj = couleur_majoritaire(l)
                for lin in range(16) : 
                    for col in range(16) :
                        image_lego[i*size:(i+1)*size, j*size:(j+1)*size] = coul_maj
            else :
                print("Probleme !!!", d[i*image_clust.shape[1]//size+j])
                
    return int_tab(image_lego)


# In[ ]:





# In[ ]:





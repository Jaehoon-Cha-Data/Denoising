# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:52:56 2019

@author: jaehooncha
"""
import tensorflow as tf
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

### call data
samples_path = os.path.join(os.getcwd(), 'samples')
sample_dir = os.path.join(samples_path, 'sample.pickle')
with open(sample_dir, 'rb') as f:
    sample_x = pickle.load(f)
    
gsample_dir = os.path.join(samples_path, 'gsample.pickle')
with open(gsample_dir, 'rb') as f:
    gsample_x = pickle.load(f) 

psample = os.path.join(samples_path, 'psample.pickle')
with open(psample, 'rb') as f:
    psample_x = pickle.load(f)  

gpsample = os.path.join(samples_path, 'gpsample.pickle')
with open(gpsample, 'rb') as f:
    gpsample_x = pickle.load(f)

### draw result
def draw_result(model_name):    
    try:
        os.mkdir("recon20")
    except OSError:
        pass  
    nx = 10; ny = 8; # size
    
    canvas = np.zeros((28*nx+10, 28*ny + 60))
    canvas[0:10, :] = 1.

    j = 0;
    for i in range(nx):
        canvas[i*28+10:(i+1)*28+10, j*28:(j+1)*28] = sample_x[i].reshape(28,28)
    j = 1;        
    _, s_re, _ = model([sample_x, sample_x])
    s_re = np.array(s_re)
    for i in range(nx):        
        canvas[i*28+10:(i+1)*28+10, j*28:(j+1)*28] = s_re[i].reshape(28,28)
        
    j = 2;
    for i in range(nx):
        canvas[i*28+10:(i+1)*28+10, j*28+20:(j+1)*28+20] = gsample_x[i].reshape(28,28)
    j = 3;        
    _, g_re, _ = model([gsample_x, gsample_x])
    g_re = np.array(s_re)
    for i in range(nx):        
        canvas[i*28+10:(i+1)*28+10, j*28+20:(j+1)*28+20] = g_re[i].reshape(28,28)
    
    j = 4;
    for i in range(nx):
        canvas[i*28+10:(i+1)*28+10, j*28+40:(j+1)*28+40] = psample_x[i].reshape(28,28)
    j = 5;
    _, p_re, _ = model([psample_x, psample_x])        
    p_re = np.array(p_re)
    for i in range(nx):        
        canvas[i*28+10:(i+1)*28+10, j*28+40:(j+1)*28+40] = p_re[i].reshape(28,28)

    j = 6;
    for i in range(nx):
        canvas[i*28+10:(i+1)*28+10, j*28+60:(j+1)*28+60] = gpsample_x[i].reshape(28,28)
    j = 7;        
    _, gp_re, _ = model([gpsample_x, gpsample_x])
    gp_re = np.array(gp_re)
    for i in range(nx):        
        canvas[i*28+10:(i+1)*28+10, j*28+60:(j+1)*28+60] = gp_re[i].reshape(28,28)

    fig = plt.figure(figsize=(13, 11))        
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.axis('off')
    plt.text(0, 4, r'ground' '\n' r'truth(gt)', fontsize = 15)
    plt.text(29, 4, 'reconstr', fontsize = 15)
    plt.text(76, 4, r'gt+' '\n' r'N(0,25)', fontsize = 15)
    plt.text(104, 4, 'reconstr', fontsize = 15)
    plt.text(153, 4, r'gt+' '\n' r'P(30)', fontsize = 15)
    plt.text(182, 4, 'reconstr', fontsize = 15)
    plt.text(227, 4, 'gt+P(30)' '\n' r'+N(0,25)', fontsize = 15)
    plt.text(257, 4, 'reconstr', fontsize = 15)

    plt.tight_layout()
    save_name = 'reconstr_'+model_name+'.png'
    save_name = os.path.join('recon20', save_name)
    fig.savefig(save_name)
    plt.close()

    
model_name = ["AE", "VAE"]
io_type = ["c2c", "g2c", "p2c", "gp2c", "g2g", "p2p", "gp2gp"]
dataset = ["MNIST"]

for i in range(1):
    model_data = model_name[i]+'_'+dataset[0]
    for io in io_type:
        folder_dir =''
        folder_dir = os.path.join(model_name[i], io)
        folder_dir = os.path.join(folder_dir, model_data)
        model = tf.keras.models.load_model(folder_dir+'\model')
        draw_result(io)
        
        
        
        
        
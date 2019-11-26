# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 00:08:55 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

run 
"""
from models import AE, VAE
from networks import step_lr
import tensorflow as tf
import numpy as np
import os
from mnist import Mnist
import argparse
from collections import OrderedDict
np.random.seed(0)
tf.random.set_seed(0)
tf.keras.backend.set_floatx('float64')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = 'AE')
    parser.add_argument('--path_dir', type = str, default = os.getcwd())
    parser.add_argument('--i_type', type = str, default = 'c')
    parser.add_argument('--o_type', type = str, default = 'c')
    parser.add_argument('--datasets', type = str, default = 'MNIST') 
    parser.add_argument('--epochs', type = int, default = 400)
    parser.add_argument('--ep_set', type = list, default = [200, 300, 400])
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--lr_set', type = list, default = [0.005, 0.001, 0.0005])
    parser.add_argument('--archi', type = dict, default = {'h1':500,
                                                           'h2':500,
                                                           'latent':20})
    
    args = parser.parse_args()
    
    config = OrderedDict([
            ('model_name', args.model_name),
            ('path_dir', args.path_dir),
            ('i_type', args.i_type),
            ('o_type', args.o_type),
            ('datasets', args.datasets),
            ('epochs', args.epochs),
            ('batch_size', args.batch_size),
            ('lr_set', args.lr_set),
            ('ep_set', args.ep_set),
            ('archi', args.archi)])
    
    return config
    
config = parse_args()

config['io_type'] = config['i_type'] + '2' +config['o_type']

mnist = Mnist()

config['n_output'] = mnist.out_shape

n_samples = mnist.num_examples
iter_per_epoch = int(n_samples/config['batch_size']) 


train_x, train_y = mnist.train_images, mnist.train_labels
test_x, test_y = mnist.test_images, mnist.test_labels


if config['model_name'] == 'AE':
    print('Run AE')
    model = AE(config['n_output'], config['archi'])   
elif config['model_name'] == 'VAE':
    print('Run VAE')
    model = VAE(config['n_output'], config['archi'])

    
mother_folder = os.path.join(config['path_dir'], config['model_name'])
mother_folder = os.path.join(mother_folder, config['io_type'])
try:
    os.mkdir(mother_folder)
except OSError:
    pass    

folder_name = os.path.join(mother_folder, config['model_name']+'_'+config['datasets'])
try:
    os.mkdir(folder_name)
except OSError:
    pass    


optimizer = tf.keras.optimizers.Adam(lr=config['lr_set'][0])

train_loss = tf.keras.metrics.Mean(name='train_loss')

summary_writer = tf.summary.create_file_writer(folder_name)

Lr = step_lr(config['ep_set'], config['lr_set'])

@tf.function
def train_step(X, i):
    tf.keras.backend.set_value(optimizer.lr, i)
    with tf.GradientTape() as tape:
        if config['i_type'] == 'c':
            X = tf.identity(X)
        elif config['i_type'] == 'g':
            X = tf.clip_by_value(X*255 
                                 + tf.random.normal(tf.shape(X), 0., 25., dtype = tf.float64),0., 255.)/255.
        elif config['i_type'] == 'p':
            X = tf.clip_by_value(X*255 
                                 + tf.random.poisson(tf.shape(X), 30., dtype = tf.float64),0., 255.)/255.            
        elif config['i_type'] == 'gp':
            X = tf.clip_by_value(X*255 
                                 + tf.random.poisson(tf.shape(X), 30., dtype = tf.float64)
                                 + tf.random.normal(tf.shape(X), 0., 25., dtype = tf.float64),0., 255.)/255.                
       
        if config['o_type'] == 'c':
            Y = tf.identity(X)
        elif config['o_type'] == 'g':
            Y = tf.clip_by_value(X*255 
                                 + tf.random.normal(tf.shape(X), 0., 25., dtype = tf.float64),0., 255.)/255.
        elif config['o_type'] == 'p':
            Y = tf.clip_by_value(X*255 
                                 + tf.random.poisson(tf.shape(X), 30., dtype = tf.float64),0., 255.)/255.            
        elif config['o_type'] == 'gp':
            Y = tf.clip_by_value(X*255 
                                 + tf.random.poisson(tf.shape(X), 30., dtype = tf.float64)
                                 + tf.random.normal(tf.shape(X), 0., 25., dtype = tf.float64),0., 255.)/255.                 
        
        _, _, loss = model([X, Y])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

folder_name = os.path.join(mother_folder, config['model_name']+'_'+config['datasets'])
    
### run ###
def runs(log_freq = 1):
    for epoch in range(config['epochs']):
        for iter_in_epoch in range(iter_per_epoch):
            epoch_x, _ = mnist.next_train_batch(config['batch_size'])
            train_step(tf.constant(epoch_x), Lr[epoch])
            if tf.equal(optimizer.iterations % log_freq, 0):
                tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
                
        template = 'epoch: {}, completed out of: {}, train_loss: {}'
        print(template.format(epoch+1,
                              config['epochs'],
                                 train_loss.result()))   


        
train_summary_writer = tf.summary.create_file_writer(folder_name+'/train')

runs()

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 23:01:05 2019

@author: jaehooncha

@email: chajaehoon79@gmail.com

autoencoder-generative model TF2.0
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import networks

'''
Autoencoder
'''
class AE(Model):
    def __init__(self, n_output, archi):
        super(AE, self).__init__()
        self.n_output = n_output        
        self.archi = archi
        self.encoder = [Dense(archi[k], activation = 'softplus') 
                                for k in list(archi.keys())[:-1]]
        self.lat = Dense(archi['latent'])
        self.decoder = [Dense(archi[k], activation = 'softplus') 
                                for k in list(archi.keys())[::-1][1:]]
        self.out = Dense(self.n_output, activation = 'sigmoid')
        self.reconstr_loss = networks.cross_entropy
        
    def Encoder(self, x, name):
        with tf.name_scope(name):       
            for i in range(len(self.encoder)):
                x = self.encoder[i](x)
            return x
    
    def To_latent(self, x, name):
        with tf.name_scope(name):        
            x = self.lat(x)
            return x
         
    def Decoder(self, x, name):
        with tf.name_scope(name):       
            for i in range(len(self.decoder)):
                x = self.decoder[i](x)
            return x
        
    def Reconstruct(self, x):
        self.decoder_out = self.Decoder(x, 'decoder')
        return self.out(self.decoder_out)
        
    def call(self, x):
        self.encoder_out = self.Encoder(x[0], 'encoder')
        self.z = self.To_latent(self.encoder_out, 'latent')
        self.decoder_out = self.Decoder(self.z, 'decoder')
        self.reconstr = self.out(self.decoder_out)
        
        self.re_loss = self.reconstr_loss(x[1], self.reconstr)
        return self.z, self.reconstr, self.re_loss


'''
Variational Autoencoder
'''
class VAE(AE):
    def __init__(self, n_output, archi):
        super(VAE, self).__init__(n_output, archi)
        self.prior = networks.DiagonalGaussian(mu = tf.zeros(1, dtype = tf.float64), 
                                               logvar =tf.zeros(1, dtype = tf.float64))
        self.mu = Dense(self.archi['latent'])
        self.logvar = Dense(self.archi['latent'])
    
    def To_latent(self, x, name):
        with tf.name_scope(name):
            mu = self.mu(x)
            logvar = self.logvar(x)
            return networks.DiagonalGaussian(mu, logvar)
    
    def call(self, x):
        self.encoder_out = self.Encoder(x[0], 'encoder')
        self.encode_dist = self.To_latent(self.encoder_out, 'encoder_dist')
        self.z = self.encode_dist.sample()
        self.decoder_out = self.Decoder(self.z, 'decoder')
        self.reconstr = self.out(self.decoder_out)
 
        self.kl_loss = tf.reduce_mean(-self.prior.log_probability(self.z))
        self.lat_loss = tf.reduce_mean(self.encode_dist.log_probability(self.z))
        self.re_loss = self.reconstr_loss(x[1], self.reconstr)
        self.loss = tf.reduce_mean(self.re_loss + self.kl_loss + self.lat_loss)
        return self.z, self.reconstr, self.loss


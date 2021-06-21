#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of 
RRTMGP gas optics scheme

Right now just a placeholder, pasted some of the code I used in my paper

Contributions welcome!

@author: Peter Ukkonen
"""
import os
import gc
import numpy as np
from netCDF4 import Dataset
from gasopt_load_train_funcs import load_data_all,create_model, gptnorm_numba,gptnorm_numba_reverse
from gasopt_load_train_funcs import preproc_inputs,losses
from gasopt_load_train_funcs import optimizers, EarlyStopping, savemodel
from gasopt_load_train_funcs import ymeans_lw, ysigma_lw, ymeans_sw, ysigma_sw, ysigmas_sw, ysigmas_lw
from gasopt_load_train_funcs import ymeans_sw_ray, ysigma_sw_ray, ymeans_sw_abs, ysigma_sw_abs
from gasopt_load_train_funcs import plot_hist2d_T, plot_hist2d



# ------------ PREPROCESSING ---------------

# 1. Scale INPUTS

# The raw inputs are temperature, pressure, and volume mixing ratios
# Some inputs are first power-scaled, and then all inputs are min-max-scaled

# RRMTPG-NN Fortran code:
#character(32), dimension(16)              :: nn_gas_names_all = [character(len=32)  :: 'h2o', 'o3', 'co2', 'n2o', &
#      'ch4', 'cfc11', 'cfc12', 'co',  'ccl4',  'cfc22',  'hfc143a', 'hfc125', 'hfc23', 'hfc32', 'hfc134a', 'cf4'] 

#! The longwave inputs are:   tlay,    log(play),   h2o**(1/4), o3**(1/4), co2, n2o...
#    else if (ninputs == 7) then ! shortwave model using h2o, o3, co2, n2o and ch4"
#      nn_gas_names    = nn_gas_names_all(1:5)
#      scalemin   = scalemin_all(1:7)               
#      scalemax   = scalemax_all(1:7)

#real(sp), dimension(18)   :: scalemin_all =  (/ 1.60E2, 5.15E-3, 1.01E-2, 4.36E-3, 1.41E-4, 0.00E0, 2.55E-8, &
#0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0, 0.00E0 /)
#real(sp), dimension(18)   :: scalemax_all =  (/ 3.2047600E2, 1.1550600E1, 5.0775300E-1, 6.3168340E-2, &
#2.3000003E-3, 5.8135214E-7, 3.6000001E-6, 2.0000002E-9, 5.3385213E-10, 1.3127458E-6, 1.0316801E-10, 2.3845328E-10, &
#7.7914392E-10, 9.8880004E-10, 3.1067642E-11, 1.3642075E-11, 4.2330001E-10, 1.6702625E-10 /)
    
#nn_inputs(1,ilay,icol) =  (tlay(ilay,icol)     - scalemin(1)) / (scalemax(1) - scalemin(1))
#nn_inputs(2,ilay,icol) = (log(play(ilay,icol)) - scalemin(2)) / (scalemax(2) - scalemin(2))
#nn_inputs(3,ilay,icol) = ( sqrt(sqrt(gas_desc%concs(idx_h2o)%conc(ilay,icol))) - scalemin(3)) / (scalemax(3) - scalemin(3))
#nn_inputs(4,ilay,icol) = ( sqrt(sqrt(gas_desc%concs(idx_o3) %conc(ilay,icol))) - scalemin(4)) / (scalemax(4) - scalemin(4))
# ! OTHER INPUTS (GASES)
#              nn_inputs(igas,ilay,icol) = &
# (gas_desc%concs(idx_gas)%conc(ilay,icol) - scalemin(igas)) / (scalemax(igas) - scalemin(igas))



# 2. Scale OUTPUTS
# Following Ukkonen 2020, cross-sections are first power-scaled to reduce the
#                         dynamic range across different g-points
# after that, they are standard scaled (remove mean and scale to unit variance)
#                         
stdnorm = False # Regular standard scaling using sigmas of individual g-points
                # Otherwise, use the sigma across all-g-points (preserves correlations)

# shape(x_tr) = (nsamples, ninputs)
# shape(y_tr) = (nsamples, noutputs) = (nsamples,ngpt)
                
ngpt = y_tr.shape[1]

if shortwave:
    if predictand=='tau_sw_ray':
        neurons = [36,36]
        nfac = 8 # used to scale y = y_raw(1/nfac)
        y_means = ymeans_sw_ray 
        sigma = ysigma_sw_ray
    elif predictand=='tau_sw_abs':
        neurons = [36,36]
        nfac = 8
        y_means = ymeans_sw_abs
        sigma = ysigma_sw_abs
    y_tr    = gptnorm_numba(nfac,y_tr,y_means,sigma)
    y_val   = gptnorm_numba(nfac,y_val,y_means,sigma)
    y_test  = gptnorm_numba(nfac,y_test,y_means,sigma)
else:
    if predictand=='tau_lw':
        if stdnorm:
            sigma   = ysigmas_lw
        else:
            sigma   = ysigma_lw
        neurons = [58,58]
        nfac = 8
        y_means = ymeans_lw
        y_tr    = gptnorm_numba(nfac,y_tr,ymeans_lw,sigma)
        y_val   = gptnorm_numba(nfac,y_val,ymeans_lw,sigma)
        y_test   = gptnorm_numba(nfac,y_test,ymeans_lw,sigma)
    elif predictand=='planck_lw':
        neurons = [16,16]
        nfac      = 2
        y_tr    = np.power(y_tr, (1.0/nfac))
        y_val   = np.power(y_val, (1.0/nfac))
        y_test   = np.power(y_test, (1.0/nfac))

gc.collect()


# ------------ NEURAL NETWORK TRAINING ---------------

mymetrics   = ['mean_absolute_error']
valfunc     = 'val_mean_absolute_error'
activ       = 'softsign'
fpath       = rootdir+'data/tmp/tmp.h5'
epochs      = 800
patience    = 15
lossfunc    = losses.mean_squared_error
ninputs     = x_tr.shape[1]
lr          = 0.001 
batch_size  = 1024

neurons = [64,64]
neurons = [58,58]
# neurons = [52,52,52]

# batch_size  = 3*batch_size
# lr          = 2 * lr

optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 

# Create model
model = create_model(nx=ninputs,ny=ngpt,neurons=neurons,activ=activ,kernel_init='he_uniform')

model.compile(loss=lossfunc, optimizer=optim,
              metrics=mymetrics,  context= ["gpu(0)"])
model.summary()

gc.collect()
# Create earlystopper
earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)

# START TRAINING

history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                    validation_data=(x_val,y_val), callbacks=[earlystopper])
gc.collect()


# ------------- RECOMPILE WITH MEAN-ABS-ERR -------------
#model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
#earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)
#history2 = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val), callbacks=[earlystopper])
#gc.collect()


# ------------- RECOMPILE WITH MEAN-SQUARED_ERR -------------
#model.compile(loss=losses.mean_squared_error, optimizer=optim,metrics=['mean_absolute_error'])
#earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)
#history = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val), callbacks=[earlystopper])
#gc.collect()


# batch_size = 1024
# Epoch 133/800
# 6981852/6981852 [==============================] - 21s 3us/step - loss: 8.9374e-05 - mean_absolute_error: 0.0051 - val_loss: 7.8724e-05 - val_mean_absolute_error: 0.0049
# Restoring model weights from the end of the best epoch
# Epoch 00133: early stopping

# SCATTER PLOTS FOR OPTICAL DEPTH AND TRANSMITTANCE
#
#y_test_nn       = model.predict(x_test);  
#y_test_nn       = gptnorm_numba_reverse(nfac,y_test_nn, y_means,sigma)
#tau_test_nn     = y_test_nn * (np.repeat(col_dry_test[:,np.newaxis],ngpt,axis=1))
#    
#plot_hist2d(tau_test,tau_test_nn,20,True)        # 
#plot_hist2d_T(tau_test,tau_test_nn,20,True)      #  



# ------------------ SAVE AND LOAD MODELS ------------------ 


# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16_2.h5"
# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-lw-18-58-58_2.h5"
# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/pfrac-18-16-16.h5"

# savemodel(kerasfile, model)

# from keras.models import load_model
# model = load_model(kerasfile,compile=False)
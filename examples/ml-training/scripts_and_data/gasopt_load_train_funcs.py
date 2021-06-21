#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:22:19 2020

@author: peter
"""


import numpy as np
import time
import os
import h5py
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numba import njit, prange
from keras.models import Sequential
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import losses
from keras import layers
from keras import optimizers
from netCDF4 import Dataset

from sklearn.model_selection import train_test_split
import pprint
import gc

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def mse(y, y_pred):
    return np.mean(np.square(y - y_pred))
  
def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))
  
def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    EPSILON = 1e-10
    return _error(actual, predicted) / (actual + EPSILON) 

def plot_hist2d(y_true,y_pred,nbins,norm):
  
    x = y_true.flatten()
    y = y_pred.flatten()
    err =  np.corrcoef(x,y)[0,1]; err = err**2
    err2 = 100 * rmspe(x,y)
    if norm == True:
        fig, ax = plt.subplots()
        (counts, ex, ey, img) = ax.hist2d(x, y, bins=nbins, norm=LogNorm())
    else:
        plt.hist2d(x, y, bins=40)
    if (np.max(x) < 1.1) & (np.min(x) > -0.1):
        plt.xlabel('Transmittance')
        plt.ylabel('Transmittance (predicted)')
    elif (np.min(x) < 0.0):
        plt.xlabel('Normalized optical depth')
        plt.ylabel('Normalized optical depth (predicted)')
    else:
        plt.xlabel('Optical depth')
        plt.ylabel('Optical depth (predicted)')
        
    ymin, ymax = plt.gca().get_ylim()
    xmin, xmax = plt.gca().get_xlim()
    ax.set_xlim(np.min([ymin,xmin]),np.max([ymax,xmax]))
    ax.set_ylim(np.min([ymin,xmin]),np.max([ymax,xmax]))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')

    mse_err = mse(x,y)
    mae_err = np.mean(np.abs((x - y)))
    textstr0 = 'R-squared = {:0.5f}'.format(err)
    textstr1 = 'MSE = {:0.5f}'.format(mse_err)
    textstr2 = 'MAE = {:0.5f}'.format(mae_err)
    textstr3 = 'RMS % error = {:0.5f} %'.format(err2)
    plt.annotate(textstr0, xy=(-7.0, 0.15), xycoords='axes fraction')
    plt.annotate(textstr1, xy=(-7.0, 0.10), xycoords='axes fraction')
    plt.annotate(textstr2, xy=(-7.0, 0.05), xycoords='axes fraction')
    plt.annotate(textstr3, xy=(-7.0, 0.00), xycoords='axes fraction')
    ax.grid()
    ident = [xmin, xmax]
    ax.plot(ident,ident,'k')
    del x,y


def plot_hist2d_tau(y_true,y_pred,nbins,norm):
    x = y_true[(y_true<10)&(y_pred<10)].flatten()
    y = y_pred[(y_true<10)&(y_pred<10)].flatten()
    plot_hist2d(x,y,nbins,norm)
    del x,y

def plot_hist2d_T(y_true,y_pred,nbins,norm):
    y_true = np.exp(-y_true).flatten()
    y_pred = np.exp(-y_pred).flatten()
    plot_hist2d(y_true,y_pred,nbins,norm)
    del y_true,y_pred
    
# Means and standard deviations of outputs, used for scaling them
# longwave absorption cross-section 
ysigma_lw    = np.repeat(0.0008864219, 256)
ysigmas_lw  = np.array([4.37299255e-04, 5.26841090e-04, 5.78239735e-04, 6.14663819e-04,
       6.60302467e-04, 7.16811395e-04, 7.82822899e-04, 8.63865600e-04,
       9.55553434e-04, 1.02228683e-03, 1.03762641e-03, 1.04109594e-03,
       1.02161488e-03, 9.81304329e-04, 9.04381333e-04, 8.39698361e-04,
       3.44006432e-04, 3.76878830e-04, 4.18959913e-04, 4.52926586e-04,
       4.87430283e-04, 5.36457635e-04, 5.93752309e-04, 6.67204964e-04,
       7.78249931e-04, 8.71711643e-04, 9.09009308e-04, 9.50126501e-04,
       1.00429822e-03, 1.02111010e-03, 1.01376919e-03, 9.10839066e-04,
       2.83427158e-04, 2.84303271e-04, 2.77761865e-04, 2.67487456e-04,
       2.61551351e-04, 2.69951415e-04, 2.85730639e-04, 3.08865827e-04,
       3.24137072e-04, 3.31775227e-04, 3.31757939e-04, 3.31892894e-04,
       3.19441227e-04, 3.02275992e-04, 3.33752367e-04, 3.80316691e-04,
       2.57029023e-04, 2.75878643e-04, 2.83650443e-04, 2.88156793e-04,
       3.00391694e-04, 3.07757437e-04, 3.21745960e-04, 3.34062905e-04,
       3.22996697e-04, 3.27573536e-04, 3.25018162e-04, 3.13190336e-04,
       3.05996829e-04, 3.87850450e-04, 5.51637728e-04, 6.77014526e-04,
       1.67555350e-04, 1.65251753e-04, 1.63267279e-04, 1.68929400e-04,
       1.77968497e-04, 1.91097730e-04, 2.07610545e-04, 2.14189829e-04,
       2.05431599e-04, 2.09787482e-04, 2.10168684e-04, 2.03784948e-04,
       1.83591648e-04, 1.83350014e-04, 2.56995583e-04, 4.28540487e-04,
       1.64279190e-04, 1.62834011e-04, 1.63298959e-04, 1.68264902e-04,
       1.72428932e-04, 1.75483059e-04, 1.80116724e-04, 1.92946900e-04,
       2.20567861e-04, 2.51571706e-04, 2.64025701e-04, 2.80212989e-04,
       2.96684419e-04, 3.18566337e-04, 3.42430460e-04, 3.62690946e-04,
       1.09419969e-04, 1.00449935e-04, 9.77658347e-05, 9.85394290e-05,
       1.03576465e-04, 1.14851158e-04, 1.34463873e-04, 1.67545164e-04,
       2.32224411e-04, 2.95537990e-04, 3.17997969e-04, 3.39454447e-04,
       3.56933539e-04, 3.70330643e-04, 3.74660420e-04, 3.86928150e-04,
       1.01457197e-04, 9.76363881e-05, 9.52763585e-05, 9.38718731e-05,
       9.43289342e-05, 9.63170314e-05, 1.03159939e-04, 1.20234152e-04,
       1.48868057e-04, 1.69198669e-04, 1.77045746e-04, 1.88619684e-04,
       2.12944928e-04, 2.38789187e-04, 2.61871202e-04, 2.65984563e-04,
       1.38633754e-04, 1.59921809e-04, 1.88785052e-04, 2.11048216e-04,
       2.30200123e-04, 2.50673329e-04, 2.74246908e-04, 3.11643613e-04,
       3.61977669e-04, 3.98050790e-04, 4.08658263e-04, 4.18197596e-04,
       4.28431580e-04, 4.70186700e-04, 5.45153394e-04, 5.80056920e-04,
       3.68984154e-04, 3.96550517e-04, 4.18841897e-04, 4.37972980e-04,
       4.67797625e-04, 5.15082153e-04, 5.74897393e-04, 6.46317203e-04,
       7.26829632e-04, 7.62576470e-04, 7.62943120e-04, 7.49164727e-04,
       7.07328203e-04, 6.58436329e-04, 6.29353046e-04, 5.88736439e-04,
       3.92593356e-04, 4.25560633e-04, 4.51672735e-04, 4.84155084e-04,
       5.22198330e-04, 5.71732351e-04, 6.30297116e-04, 6.98455609e-04,
       7.60274590e-04, 7.88997160e-04, 7.88744073e-04, 7.70943007e-04,
       7.30904983e-04, 6.78054581e-04, 6.36220793e-04, 6.18106918e-04,
       2.96116486e-04, 3.32603609e-04, 3.69652698e-04, 4.06244217e-04,
       4.40582633e-04, 4.79989714e-04, 5.33083454e-04, 6.06299902e-04,
       7.26722181e-04, 8.34102393e-04, 8.80141393e-04, 9.32387076e-04,
       9.89418477e-04, 1.05255889e-03, 1.12367771e-03, 1.18428818e-03,
       1.16735901e-04, 1.09520239e-04, 1.10411493e-04, 1.14395698e-04,
       1.20297525e-04, 1.25826671e-04, 1.30226646e-04, 1.30812914e-04,
       1.58120849e-04, 2.33974846e-04, 2.50196754e-04, 3.14698613e-04,
       4.02720645e-04, 4.94655105e-04, 5.76312130e-04, 6.47833105e-04,
       2.21976341e-04, 2.52069411e-04, 2.96846789e-04, 3.47360969e-04,
       3.87380074e-04, 4.04242281e-04, 4.12160036e-04, 4.28739324e-04,
       4.31390828e-04, 4.19823802e-04, 3.99220706e-04, 3.89570370e-04,
       4.18784912e-04, 6.04440458e-04, 8.10706697e-04, 8.90113297e-04,
       2.17198845e-04, 2.40237583e-04, 2.54687766e-04, 2.67839903e-04,
       2.85624934e-04, 3.02617118e-04, 3.17192840e-04, 3.33671138e-04,
       3.57876153e-04, 3.77383956e-04, 3.85278458e-04, 3.94996459e-04,
       4.06229461e-04, 4.21439443e-04, 4.47071303e-04, 4.72978980e-04,
       1.29705557e-04, 1.36837974e-04, 1.53161920e-04, 1.70639425e-04,
       1.85338911e-04, 2.03972755e-04, 2.23354771e-04, 2.43709146e-04,
       2.65126728e-04, 2.75205472e-04, 2.70803546e-04, 2.56710075e-04,
       2.31562459e-04, 2.08078374e-04, 1.93804430e-04, 1.90146762e-04])
ymeans_lw  = np.array([0.0007013 , 0.00081638, 0.00088407, 0.0009376 , 0.00100272,
   0.00108871, 0.00119965, 0.00136057, 0.00162019, 0.00185225,
   0.00195079, 0.00206555, 0.00220559, 0.00240607, 0.00271468,
   0.00301905, 0.00046654, 0.00051556, 0.00058341, 0.00065088,
   0.00070464, 0.00076876, 0.00085896, 0.00097935, 0.00119935,
   0.00141034, 0.00151006, 0.00163442, 0.00180711, 0.00204042,
   0.00242276, 0.00280669, 0.00042887, 0.00046485, 0.00051584,
   0.00058201, 0.00065572, 0.00072477, 0.00080584, 0.00091423,
   0.00110042, 0.00127067, 0.00134649, 0.00144005, 0.00155864,
   0.0017359 , 0.00192395, 0.00202425, 0.00089971, 0.00101015,
   0.0010735 , 0.00113382, 0.00121697, 0.00131415, 0.0014548 ,
   0.00166414, 0.00201406, 0.0023286 , 0.00248189, 0.0026516 ,
   0.0028395 , 0.00309552, 0.00329059, 0.00340846, 0.00040419,
   0.00044149, 0.00049501, 0.00057646, 0.00065665, 0.00072959,
   0.0008178 , 0.00092567, 0.00110577, 0.00127773, 0.00135464,
   0.00144639, 0.00155456, 0.00172624, 0.00195144, 0.00219249,
   0.00037311, 0.00038832, 0.00039876, 0.00039357, 0.00039073,
   0.00039439, 0.00040139, 0.00041544, 0.00045041, 0.00048733,
   0.00050021, 0.000511  , 0.0005222 , 0.00053226, 0.00053905,
   0.00053686, 0.00045033, 0.00050789, 0.00056031, 0.0006055 ,
   0.00064596, 0.0006884 , 0.00073842, 0.00080476, 0.00091462,
   0.00100837, 0.00104174, 0.00107803, 0.00112003, 0.00116588,
   0.00121704, 0.00124574, 0.00043096, 0.00044609, 0.00045882,
   0.00047328, 0.00049173, 0.00051617, 0.00055129, 0.00060892,
   0.00070759, 0.00078732, 0.00081574, 0.00084897, 0.00089626,
   0.00096228, 0.00103707, 0.00107244, 0.00044382, 0.00047923,
   0.00052424, 0.00056825, 0.00061594, 0.00066949, 0.00073642,
   0.00083537, 0.00099988, 0.00115576, 0.00122428, 0.00130514,
   0.0013967 , 0.00150886, 0.00164095, 0.00173281, 0.0005398 ,
   0.00058346, 0.00061954, 0.00065103, 0.00069784, 0.00076837,
   0.00086104, 0.00099005, 0.00119531, 0.00135744, 0.00142613,
   0.00151644, 0.00163775, 0.0017866 , 0.00194151, 0.00203961,
   0.00062092, 0.00068568, 0.00072482, 0.00076629, 0.00081684,
   0.00088733, 0.00098261, 0.00111583, 0.00132001, 0.00149657,
   0.00157718, 0.00167974, 0.00181246, 0.00196578, 0.00211621,
   0.00222501, 0.00027775, 0.00031307, 0.00034595, 0.00037792,
   0.00040985, 0.00044745, 0.00049738, 0.00056636, 0.00068384,
   0.00079073, 0.00083684, 0.00089007, 0.00094933, 0.00101663,
   0.00109136, 0.00115981, 0.00041082, 0.00047212, 0.0005221 ,
   0.00057267, 0.00062689, 0.00068311, 0.00074801, 0.00083137,
   0.0009249 , 0.00097404, 0.00098701, 0.0009677 , 0.00091862,
   0.00092232, 0.00097919, 0.00099403, 0.00082178, 0.00097702,
   0.00115414, 0.00134279, 0.00149137, 0.00160743, 0.00175431,
   0.00198898, 0.00242579, 0.00281907, 0.00299135, 0.00319233,
   0.00345435, 0.00384145, 0.00410627, 0.00420549, 0.00022183,
   0.00025186, 0.00027004, 0.0002846 , 0.0003047 , 0.00032399,
   0.00034067, 0.00035919, 0.00038545, 0.00040524, 0.00041355,
   0.00042393, 0.00043629, 0.00045243, 0.0004797 , 0.00050912,
   0.00030737, 0.00035681, 0.00039763, 0.00043568, 0.00047533,
   0.00052387, 0.00058468, 0.00066867, 0.00081455, 0.00094435,
   0.00099829, 0.00107121, 0.00117441, 0.00127705, 0.00135903,
   0.00142707])

ysigma_sw   = np.repeat(0.0005998554,224)
ysigmas_sw  = np.array([1.27814161e-04, 1.24540881e-04, 1.41172712e-04, 1.89145269e-04,
       2.62712850e-04, 3.23056870e-04, 3.55804362e-04, 3.86234496e-04,
       4.11009660e-04, 4.39618265e-04, 4.40042773e-04, 4.35116332e-04,
       4.26761096e-04, 3.87661635e-04, 4.03894912e-04, 8.07047017e-04,
       1.23830417e-04, 1.37381488e-04, 1.56902863e-04, 1.76574088e-04,
       1.92512035e-04, 2.12301279e-04, 2.32688518e-04, 2.53983268e-04,
       2.76105686e-04, 2.86116877e-04, 2.81211366e-04, 2.66394162e-04,
       2.40595121e-04, 2.16147487e-04, 2.00186797e-04, 1.94676978e-04,
       2.37603373e-04, 2.89009099e-04, 3.44233174e-04, 3.88329096e-04,
       4.06594497e-04, 4.16530097e-04, 4.36349100e-04, 4.70867451e-04,
       5.08359322e-04, 5.32150402e-04, 5.25672085e-04, 5.10799599e-04,
       4.83576173e-04, 4.05817197e-04, 3.73896989e-04, 3.85701560e-04,
       8.18976988e-05, 9.08759505e-05, 1.01442121e-04, 1.14337555e-04,
       1.30766392e-04, 1.44241287e-04, 1.56067577e-04, 1.72221356e-04,
       1.96913703e-04, 2.23203475e-04, 2.36479604e-04, 2.50731819e-04,
       2.69974654e-04, 3.04055015e-04, 3.40785948e-04, 4.06201529e-04,
       7.31557410e-05, 9.46763784e-05, 1.13033546e-04, 1.28629883e-04,
       1.43667188e-04, 1.57838679e-04, 1.69634193e-04, 1.76631382e-04,
       1.71053566e-04, 1.58788352e-04, 1.50395342e-04, 1.39477244e-04,
       1.27494896e-04, 1.51993947e-04, 2.06523649e-04, 2.42232478e-04,
       4.19311709e-05, 6.73102931e-05, 1.00489899e-04, 1.67361911e-04,
       2.54074878e-04, 2.95042066e-04, 3.32043610e-04, 3.80501283e-04,
       4.50072154e-04, 5.00801946e-04, 5.11547767e-04, 5.16158038e-04,
       5.09433450e-04, 4.91208391e-04, 4.66760618e-04, 4.40874979e-04,
       4.44860585e-05, 7.85371548e-05, 1.27778642e-04, 1.79093341e-04,
       2.27646273e-04, 2.73791663e-04, 3.15229382e-04, 3.58114368e-04,
       4.03334436e-04, 4.37817733e-04, 4.49111904e-04, 4.59273220e-04,
       4.63792154e-04, 4.66039401e-04, 4.61428944e-04, 4.51644876e-04,
       4.07694900e-05, 4.57269388e-05, 5.13128045e-05, 5.79167072e-05,
       6.48957240e-05, 7.16704420e-05, 7.89783896e-05, 9.03737763e-05,
       9.96882181e-05, 8.84127119e-05, 5.69354242e-05, 2.56520585e-05,
       2.85639710e-05, 4.96997174e-05, 6.66791539e-05, 8.22545991e-05,
       1.84882328e-05, 2.13826588e-05, 3.21290042e-05, 5.00023471e-05,
       7.00605286e-05, 9.80905887e-05, 1.27864633e-04, 1.61017452e-04,
       2.14028149e-04, 2.70803458e-04, 2.93106775e-04, 3.12729240e-04,
       3.28561728e-04, 3.34229222e-04, 3.42639016e-04, 3.42321169e-04,
       2.04050181e-05, 2.23319317e-05, 2.78903404e-05, 3.49673421e-05,
       4.15210184e-05, 4.68700551e-05, 5.11649932e-05, 5.90531148e-05,
       6.97051857e-05, 1.01073307e-04, 1.15986701e-04, 1.27121385e-04,
       1.21010947e-04, 6.40740499e-05, 5.38448514e-05, 1.22012593e-04,
       3.43876783e-05, 2.88562729e-05, 3.01832448e-05, 3.49710866e-05,
       4.12761956e-05, 4.36928035e-05, 4.45287905e-05, 4.36617841e-05,
       4.30529678e-05, 4.30917925e-05, 4.34794957e-05, 4.48052412e-05,
       4.80746174e-05, 5.55976663e-05, 6.82312561e-05, 8.46260321e-05,
       5.60645383e-07, 5.72529241e-07, 5.83844192e-07, 5.94286101e-07,
       6.03677767e-07, 6.11910920e-07, 6.18756567e-07, 6.25703479e-07,
       6.25800019e-07, 6.29673493e-07, 6.29992156e-07, 6.30235745e-07,
       6.30411102e-07, 6.30541964e-07, 6.30613282e-07, 6.29795625e-07,
       9.39450744e-06, 4.77085336e-05, 1.09209674e-04, 1.77969836e-04,
       2.35875975e-04, 2.75188760e-04, 2.98505001e-04, 3.11339560e-04,
       3.18148198e-04, 3.20237621e-04, 3.20672514e-04, 3.20939993e-04,
       3.21082885e-04, 3.21137079e-04, 3.21170469e-04, 3.21149591e-04,
       1.10989079e-04, 1.23178570e-04, 1.56121705e-04, 1.91726469e-04,
       2.20147219e-04, 2.38825047e-04, 2.43783008e-04, 2.45615778e-04,
       2.47220540e-04, 2.47980992e-04, 2.48164290e-04, 2.48593253e-04,
       2.49192537e-04, 2.49370683e-04, 2.49482193e-04, 2.49547888e-04])
ymeans_sw = np.array([0.00037083, 0.00044209, 0.00050536, 0.00058699, 0.00069377,
       0.0008147 , 0.00095259, 0.00118586, 0.00160582, 0.00188952,
       0.00201833, 0.00219009, 0.00244024, 0.00283895, 0.00348068,
       0.00406385, 0.00034179, 0.00038664, 0.00042815, 0.00046809,
       0.0005097 , 0.00056119, 0.00062559, 0.00071393, 0.00086488,
       0.00099768, 0.00105177, 0.0011247 , 0.00122712, 0.00132632,
       0.00140509, 0.00147116, 0.0004506 , 0.00053909, 0.00064168,
       0.00074254, 0.00084421, 0.00096252, 0.00108908, 0.00123447,
       0.00148458, 0.00171693, 0.00181891, 0.00194639, 0.00210757,
       0.00235399, 0.00260144, 0.00272683, 0.00032192, 0.00034846,
       0.00037568, 0.00040586, 0.00044164, 0.00048103, 0.00052855,
       0.00059311, 0.00069642, 0.00079178, 0.00083278, 0.00087989,
       0.00093853, 0.00101786, 0.00110595, 0.00112647, 0.00032898,
       0.00037803, 0.00042608, 0.00047477, 0.00052442, 0.00057716,
       0.00063745, 0.00071833, 0.00085972, 0.00099369, 0.00105385,
       0.00113107, 0.00124225, 0.00137246, 0.00152802, 0.00163304,
       0.00032347, 0.00035042, 0.00038449, 0.00044857, 0.00053532,
       0.00059162, 0.00065327, 0.00074091, 0.00089425, 0.00103798,
       0.00110064, 0.00117579, 0.00127228, 0.00139873, 0.00152858,
       0.00161931, 0.00033517, 0.00037287, 0.0004173 , 0.00046665,
       0.00052037, 0.0005756 , 0.00063536, 0.00071477, 0.00085556,
       0.00099234, 0.00105448, 0.00112842, 0.00121555, 0.00132714,
       0.00145095, 0.00155022, 0.00036895, 0.00038357, 0.00039398,
       0.00040467, 0.00041701, 0.0004305 , 0.00044344, 0.00046376,
       0.0005305 , 0.00061535, 0.00067116, 0.00073367, 0.00079331,
       0.00085168, 0.00090718, 0.00094346, 0.00038343, 0.0003924 ,
       0.00040473, 0.00042312, 0.00044495, 0.00047225, 0.00050571,
       0.00054339, 0.00060592, 0.00065974, 0.00068728, 0.00073093,
       0.00079389, 0.00088332, 0.00098721, 0.00108659, 0.00045222,
       0.00046618, 0.00047975, 0.00049424, 0.00050797, 0.00052239,
       0.0005392 , 0.00056305, 0.00062147, 0.00068353, 0.00071958,
       0.00078041, 0.00088791, 0.00110135, 0.00138173, 0.00152975,
       0.0005231 , 0.00053445, 0.00054292, 0.00055498, 0.00056824,
       0.00057908, 0.00058575, 0.00059095, 0.00059543, 0.0005999 ,
       0.00060232, 0.00060586, 0.00061152, 0.0006206 , 0.00063354,
       0.00065011, 0.00057636, 0.00058859, 0.00060019, 0.00061094,
       0.00062062, 0.00062907, 0.00063611, 0.00064161, 0.00064556,
       0.00064732, 0.00064764, 0.00064789, 0.00064809, 0.00064823,
       0.0006483 , 0.00064833, 0.0006635 , 0.00070939, 0.00078375,
       0.00088054, 0.00098333, 0.00106954, 0.00112823, 0.00116335,
       0.00118329, 0.0011896 , 0.00119042, 0.00119094, 0.00119124,
       0.00119147, 0.00119165, 0.0011917 , 0.00104226, 0.0010737 ,
       0.00112755, 0.00119405, 0.00125461, 0.00129682, 0.00131468,
       0.00132099, 0.00132339, 0.00132437, 0.00132464, 0.00132507,
       0.00132562, 0.00132582, 0.00132593, 0.00132591])

# shortwave rayleigh cross-section
ymeans_sw_ray = np.array([0.00016408, 0.00016821, 0.00016852, 0.00016616, 0.0001631 ,
       0.0001615 , 0.00016211, 0.00016632, 0.00017432, 0.00017609,
       0.00017617, 0.00017683, 0.00017806, 0.00017891, 0.00017938,
       0.00017905, 0.00020313, 0.000203  , 0.00020417, 0.00020546,
       0.00020597, 0.00020647, 0.0002067 , 0.0002069 , 0.00020719,
       0.00020752, 0.00020766, 0.00020783, 0.00020801, 0.00020828,
       0.00020884, 0.00020988, 0.00022147, 0.00022575, 0.00022777,
       0.00022846, 0.00022824, 0.00022803, 0.00022816, 0.00022841,
       0.0002286 , 0.00022876, 0.00022883, 0.00022887, 0.00022888,
       0.00022891, 0.00022922, 0.00023004, 0.00025017, 0.00024942,
       0.00024824, 0.00024734, 0.00024655, 0.00024587, 0.00024539,
       0.00024486, 0.00024454, 0.0002441 , 0.00024381, 0.00024343,
       0.00024307, 0.00024265, 0.00024179, 0.00024042, 0.00025942,
       0.00026145, 0.00026296, 0.00026379, 0.00026454, 0.00026518,
       0.00026548, 0.00026566, 0.00026578, 0.00026592, 0.00026607,
       0.00026617, 0.00026612, 0.00026633, 0.00026634, 0.00026667,
       0.00028838, 0.00028652, 0.00028487, 0.00028311, 0.00027978,
       0.00027901, 0.00027885, 0.00027866, 0.000278  , 0.00027733,
       0.00027734, 0.00027694, 0.00027574, 0.00027526, 0.0002754 ,
       0.00027594, 0.00030356, 0.00031213, 0.00031563, 0.00031476,
       0.00031548, 0.00031693, 0.00031758, 0.00031764, 0.00031757,
       0.00031747, 0.0003175 , 0.0003176 , 0.00031767, 0.00031787,
       0.00031921, 0.00032022, 0.00033161, 0.00033401, 0.00033486,
       0.0003345 , 0.00033421, 0.00033408, 0.00033402, 0.00033377,
       0.00033366, 0.00033365, 0.0003337 , 0.0003337 , 0.00033372,
       0.00033368, 0.0003336 , 0.00033342, 0.00038102, 0.00038465,
       0.00038598, 0.00038884, 0.00039175, 0.00039174, 0.00039248,
       0.00039248, 0.00039339, 0.00038445, 0.00037872, 0.00037472,
       0.00037253, 0.00036779, 0.00036238, 0.00035383, 0.0004347 ,
       0.00044431, 0.00045121, 0.00045676, 0.00046053, 0.00046292,
       0.00046443, 0.00046354, 0.00045597, 0.0004476 , 0.00044526,
       0.00044227, 0.00043994, 0.00043726, 0.00043139, 0.00043131,
       0.00050991, 0.0005198 , 0.00052429, 0.00052806, 0.00052854,
       0.00052917, 0.00053388, 0.00053857, 0.0005396 , 0.00053686,
       0.00053488, 0.0005327 , 0.00052904, 0.00052511, 0.00052043,
       0.00051689, 0.00057637, 0.0005886 , 0.0006002 , 0.00061095,
       0.00062064, 0.00062908, 0.00063611, 0.00064162, 0.00064557,
       0.00064733, 0.00064765, 0.0006479 , 0.0006481 , 0.00064824,
       0.00064831, 0.00064833, 0.00065669, 0.00067188, 0.00068705,
       0.00070096, 0.00071349, 0.00072444, 0.00073358, 0.00074076,
       0.00074614, 0.00074835, 0.00074795, 0.00074786, 0.0007479 ,
       0.00074804, 0.00074818, 0.00074823, 0.0008143 , 0.00081451,
       0.00081412, 0.00081407, 0.00081408, 0.00081299, 0.00081158,
       0.00081498, 0.00081472, 0.0008145 , 0.00081539, 0.00081426,
       0.00081398, 0.000814  , 0.00081404, 0.00081415])
ysigma_sw_ray = np.repeat(0.00019679657,224)

# shortwave absorption cross-section
ymeans_sw_abs = np.array([3.64390580e-04, 4.35663940e-04, 4.98018635e-04, 5.77545608e-04,
       6.80800469e-04, 7.98740832e-04, 9.35279648e-04, 1.16656872e-03,
       1.58452173e-03, 1.86584354e-03, 1.99465151e-03, 2.16701976e-03,
       2.41802959e-03, 2.82146805e-03, 3.48183908e-03, 4.09035478e-03,
       3.24113556e-04, 3.74707161e-04, 4.17389121e-04, 4.57456830e-04,
       4.98836453e-04, 5.49621007e-04, 6.13025972e-04, 7.00094330e-04,
       8.49446864e-04, 9.81244841e-04, 1.03521883e-03, 1.10830076e-03,
       1.21134310e-03, 1.31195760e-03, 1.39195414e-03, 1.45876186e-03,
       4.28469037e-04, 5.20646572e-04, 6.22550666e-04, 7.22033263e-04,
       8.23189737e-04, 9.40993312e-04, 1.06700219e-03, 1.21110224e-03,
       1.45994173e-03, 1.69180706e-03, 1.79443962e-03, 1.92319078e-03,
       2.08631344e-03, 2.33873748e-03, 2.59446981e-03, 2.72375043e-03,
       2.89453164e-04, 3.26674141e-04, 3.59543861e-04, 3.93101625e-04,
       4.30800777e-04, 4.71213047e-04, 5.19042369e-04, 5.83244429e-04,
       6.85371691e-04, 7.79234222e-04, 8.19451292e-04, 8.65648268e-04,
       9.23064887e-04, 1.00047945e-03, 1.08587136e-03, 1.09644048e-03,
       2.97961291e-04, 3.59470578e-04, 4.12600290e-04, 4.63586446e-04,
       5.14341518e-04, 5.67600771e-04, 6.28228823e-04, 7.09333457e-04,
       8.51527497e-04, 9.86502739e-04, 1.04738004e-03, 1.12565351e-03,
       1.23939372e-03, 1.37201988e-03, 1.52829266e-03, 1.63304247e-03,
       2.70115474e-04, 3.13259574e-04, 3.55943455e-04, 4.24126105e-04,
       5.12095110e-04, 5.70286124e-04, 6.33014424e-04, 7.20241107e-04,
       8.71218799e-04, 1.01254229e-03, 1.07443938e-03, 1.14905764e-03,
       1.24553905e-03, 1.37227518e-03, 1.50288385e-03, 1.59471075e-03,
       2.49097910e-04, 3.04214394e-04, 3.61522223e-04, 4.22842306e-04,
       4.84134798e-04, 5.45158167e-04, 6.10073039e-04, 6.93159061e-04,
       8.35262705e-04, 9.70904832e-04, 1.03256141e-03, 1.10617711e-03,
       1.19320781e-03, 1.30544091e-03, 1.43013091e-03, 1.53011072e-03,
       3.03399313e-04, 3.29578412e-04, 3.45331355e-04, 3.61360639e-04,
       3.78781464e-04, 3.96694435e-04, 4.13389760e-04, 4.41241835e-04,
       5.20797505e-04, 6.10202318e-04, 6.68037275e-04, 7.32506509e-04,
       7.93701038e-04, 8.53195263e-04, 9.09500406e-04, 9.46514192e-04,
       2.32592269e-04, 2.71835335e-04, 3.08167830e-04, 3.44223663e-04,
       3.79800709e-04, 4.20017983e-04, 4.63830249e-04, 5.07036340e-04,
       5.73535392e-04, 6.31669944e-04, 6.62838283e-04, 7.10128399e-04,
       7.75139430e-04, 8.65899900e-04, 9.70544817e-04, 1.06985983e-03,
       3.41864652e-04, 3.69071873e-04, 3.96323914e-04, 4.22754820e-04,
       4.46886756e-04, 4.71655425e-04, 4.98428126e-04, 5.32940670e-04,
       6.06888614e-04, 6.70523092e-04, 7.07189436e-04, 7.71613559e-04,
       8.81720975e-04, 1.09840697e-03, 1.38371368e-03, 1.53473706e-03,
       4.17014409e-04, 4.30032291e-04, 4.34701447e-04, 4.33250068e-04,
       4.38496791e-04, 4.55915550e-04, 4.31207794e-04, 4.22994781e-04,
       4.37038223e-04, 4.48761188e-04, 4.56356967e-04, 4.66349302e-04,
       4.80149902e-04, 4.99643327e-04, 5.27186319e-04, 5.57484163e-04,
       2.16507342e-05, 2.14800675e-05, 2.42409842e-05, 2.30929109e-05,
       2.50050962e-05, 2.49029163e-05, 2.54020069e-05, 2.31895119e-05,
       2.51217079e-05, 2.50334833e-05, 2.48085526e-05, 2.50862649e-05,
       2.36565447e-05, 2.40919053e-05, 2.22349281e-05, 2.54304305e-05,
       4.07712301e-04, 5.41551271e-04, 6.77760807e-04, 8.20003042e-04,
       9.50566900e-04, 1.05036958e-03, 1.11506274e-03, 1.15274324e-03,
       1.17376540e-03, 1.18031248e-03, 1.18122390e-03, 1.18179375e-03,
       1.18207792e-03, 1.18228467e-03, 1.18242903e-03, 1.18247792e-03,
       1.02293247e-03, 1.05753809e-03, 1.11542153e-03, 1.18556281e-03,
       1.24850264e-03, 1.29191973e-03, 1.30981358e-03, 1.31571468e-03,
       1.31824624e-03, 1.31927885e-03, 1.31954963e-03, 1.31999550e-03,
       1.32057443e-03, 1.32077874e-03, 1.32089714e-03, 1.32086105e-03])
ysigma_sw_abs = np.repeat(0.00065187697,224)

@njit(parallel=True)
def gptnorm_numba(nfac,y,means,sigma):
    # y has shape (nobs,gpts)
    (nobs,ngpt) = y.shape
    nfacc = 1/nfac
    for iobs in prange(nobs):
        for igpt in prange(ngpt):
            y[iobs,igpt] = np.power(y[iobs,igpt],nfacc)
            y[iobs,igpt] = (y[iobs,igpt] - means[igpt]) / sigma[igpt]
    return y

@njit(parallel=True)
def gptnorm_numba_reverse(nfac,y,means,sigma):
    # y has shape (nobs,gpts)
    (nobs,ngpt) = y.shape
    for iobs in prange(nobs):
        for igpt in prange(ngpt):
            y[iobs,igpt] = (y[iobs,igpt] * sigma[igpt]) + means[igpt]
            y[iobs,igpt] = np.power(y[iobs,igpt],nfac)

    return y


def minmaxscale(x,datamin,datamax):
    for i in range(x.shape[1]):
        x[:,i] =  (x[:,i] - datamin[i]) / (datamax[i] - datamin[i] )
    return x


# Sqrt-scaling of H2O and O3 + min max scaling of all inputs
def preproc_inputs(x, datamin, datamax):
    x[:,2] = x[:,2]**(1.0/4) 
    x[:,3] = x[:,3]**(1.0/4) 
    x = minmaxscale(x,datamin,datamax)
    return x

def load_inp_outp_RFMIP(fname,ychoice=1,shortwave=False,remove_CKDMIP=False):
    dat = Dataset(fname)
    
   # ychoice= 0 for planck fraction (longwave) or ssa(shortwave), 
   # 1 for optical depth (sw/lw), 2 for tau_rau (sw), 3 for tau_abs(sw)
    if (shortwave):
        if ychoice==0:  varname = 'ssa'
        else:           varname = 'tau_sw'
    else:
        if ychoice==0:  varname = 'planck_frac'
        else:           varname = 'tau_lw'
    print("Extracting {} as output".format(varname))
    # Last lev has nans
    x       = dat.variables['nn_input'][:,:,0:-1,:].data
    col_dry = dat.variables['col_dry'][:,:,0:-1].data
    if ychoice != 0: x = np.concatenate((x, col_dry[:,:,:,np.newaxis]),axis=3)
    
    if (shortwave) & (ychoice > 1): 
        # tau_sw = tau_abs + tau_ray
        # ssa  = tau_ray / tau_sw = tau_ray / (tau_abs + tau_ray)
        # tau_rau = ssa * tau_sw
        # tau_abs = (1-ssa) * tau_sw
        if ychoice == 2: # tau_rau
            y_ssa       = dat.variables['ssa'][:,:,0:-1,:].data
            y = y_ssa * dat.variables['tau_sw'][:,:,0:-1,:].data
        elif ychoice == 3: # tau_abs
            y_ssa       = dat.variables['ssa'][:,:,0:-1,:].data
            y = (1-y_ssa) * dat.variables['tau_sw'][:,:,0:-1,:].data
    else: # ssa or tau_tot
        y  = dat.variables[varname][:,:,0:-1,:].data
    
    matshape = y.shape
    nexp = matshape[0];    ncol = matshape[1]
    nlev = matshape[2];    ngpt = matshape[3]
    ngas = x.shape[3];
    print( "there are {} profiles (expt*col) and {} columns in this dataset".format(nexp*ncol,ncol))
  
    if remove_CKDMIP:
        print("removing CKDMIP exps")
        expt_label = dat.variables['expt_label'][:]
        inds_ckdmip = [i for i, s in enumerate(expt_label) if 'CKDMIP' in s]
        x = np.delete(x,inds_ckdmip,axis=0); y = np.delete(y,inds_ckdmip,axis=0)
        nexp = y.shape[0]

    inds_test = np.array([ 0,  6, 14, 19, 30, 31, 36, 46, 52, 53, 58, 74, 75, 82, 96])
    inds_tr = np.setdiff1d(np.arange(ncol),inds_test)
    ntr = inds_tr.size; ntest = inds_test.size 

    x_tr = np.reshape(x[:,inds_tr,:,:], (nexp*ntr*nlev,ngas))
    x_test = np.reshape(x[:,inds_test,:,:], (nexp*ntest*nlev,ngas))

    y_tr   = np.reshape(y[:,inds_tr,:,:], (nexp*ntr*nlev,ngpt))
    y_test = np.reshape(y[:,inds_test,:,:], (nexp*ntest*nlev,ngpt))
    dat.close();
    del x, y
    
    return x_tr,x_test,y_tr,y_test

def load_inp_outp(fname,ychoice=1,dcol=1,shortwave=False,remove_CKDMIP=False,
                  skip_lastlev=False):
    dat = Dataset(fname)
    
    if (shortwave):
        if ychoice==0:  varname = 'ssa'
        else:           varname = 'tau_sw'
    else:
        if ychoice==0:  varname = 'planck_frac'
        else:           varname = 'tau_lw'
        
    # Last lev has nans
    x = dat.variables['nn_input'][:,:,:,:].data
    col_dry = dat.variables['col_dry'][:,:,:].data
    if ychoice != 0: x = np.concatenate((x, col_dry[:,:,:,np.newaxis]),axis=3)
    # y = dat.variables[varname][:,:,:,:].data
    if (shortwave) & (ychoice > 1): 
        if ychoice == 2: # tau_rau
            y_ssa       = dat.variables['ssa'][:,:,:,:].data
            y = y_ssa * dat.variables['tau_sw'][:,:,:,:].data
        elif ychoice == 3: # tau_abs
            y_ssa       = dat.variables['ssa'][:,:,:,:].data
            y = (1-y_ssa) * dat.variables['tau_sw'][:,:,:,:].data
    else: #ssa or tau_tot
        y  = dat.variables[varname][:,:,:,:].data

    if skip_lastlev: x = x[:,:,0:-1,:]; y = y[:,:,0:-1,:]
    y  = y[:,::dcol,:,:]; x  = x[:,::dcol,:,:]
    # y = y[23:,:,:,:]; x = x[23:,:,:,:]

    if remove_CKDMIP:
        expt_label = dat.variables['expt_label'][:]
        inds_ckdmip = [i for i, s in enumerate(expt_label) if 'CKDMIP' in s]
        x = np.delete(x,inds_ckdmip,axis=0); y = np.delete(y,inds_ckdmip,axis=0)
        nexp = y.shape[0]

    matshape = y.shape
    
    nexp = matshape[0]; ncol = matshape[1]
    nlev = matshape[2]; ngpt = matshape[3]
    ngas = x.shape[3];  nobs = nexp*ncol*nlev
    print( "there are {} profiles (expt*col) and {} columns in this dataset".format(nexp*ncol,ncol))

    y = np.reshape(y, (nobs,ngpt)); x = np.reshape(x, (nobs,ngas))
    dat.close()
    return x,y

def load_inp_outp_GCM(fname,ychoice=1,dcol=1,shortwave=False,remove_CKDMIP=False):
    dat = Dataset(fname)
    
    if (shortwave):
        if ychoice==0:  varname = 'ssa'
        else:           varname = 'tau_sw'
    else:
        if ychoice==0:  varname = 'planck_frac'
        else:           varname = 'tau_lw'
        
    # Last lev has nans
    x = dat.variables['nn_input'][:,:,:,:].data
    col_dry = dat.variables['col_dry'][:,:,:].data
    if ychoice != 0: x  = np.concatenate((x, col_dry[:,:,:,np.newaxis]),axis=3)
    # y = dat.variables[varname][:,:,:,:].data
    if (shortwave) & (ychoice > 1): 
        if ychoice == 2: # tau_rau
            y_ssa       = dat.variables['ssa'][:,:,:,:].data
            y = y_ssa * dat.variables['tau_sw'][:,:,:,:].data
        elif ychoice == 3: # tau_abs
            y_ssa       = dat.variables['ssa'][:,:,:,:].data
            y = (1-y_ssa) * dat.variables['tau_sw'][:,:,:,:].data
    else: #ssa or tau_tot
        y  = dat.variables[varname][:,:,:,:].data
        
    y  = y[:,::dcol,:,:]; x  = x[:,::dcol,:,:]
    (nexp,nsite,nlay,ngas) = np.shape(x)
    
    p_sfc = dat.variables['ps'][:,:].data
    p = dat.variables['pres_layer'][:].data
    # Pressure levels below the surface need to be removed
    p_big = np.broadcast_to(p,(nexp,nsite,nlay))
    p_sfc_big = np.repeat(np.reshape(p_sfc,(nexp,nsite,1)),nlay,axis=2)
    inds_belowsfc = p_big[:,:,:] > p_sfc_big[:,:,:]
    inds_belowsfc = inds_belowsfc.flatten()
    
    if remove_CKDMIP:
        expt_label = dat.variables['expt_label'][:]
        inds_ckdmip = [i for i, s in enumerate(expt_label) if 'CKDMIP' in s]
        x = np.delete(x,inds_ckdmip,axis=0); y = np.delete(y,inds_ckdmip,axis=0)
        nexp = y.shape[0]

    matshape = y.shape
    nexp = matshape[0]; ncol = matshape[1]
    nlev = matshape[2]; ngpt = matshape[3]
    ngas = x.shape[3];  nobs = nexp*ncol*nlev
    print( "there are {} profiles (expt*col) and {} columns in this dataset".format(nexp*ncol,ncol))

    y = np.reshape(y, (nobs,ngpt)); x = np.reshape(x, (nobs,ngas))
    
    y = y[~inds_belowsfc,:]
    x = x[~inds_belowsfc,:]
    
    del p_big,p_sfc_big,inds_belowsfc
    dat.close()

    return x,y

def load_data_all(fname_rfmip, fname_cams, fname_nwp, fname_GCM, fname_Garand, fname_CKDMIP,
              include_rfmip, include_cams, include_nwpsaf, include_GCM, include_Garand, include_CKDMIP,
              shortwave=False,ychoice=0, dcol=1,frac_val=0.1, frac_cams=0.3, seed=7, 
              remove_CKDMIP_exps=False, garand_val=True):
    
    
    np.random.seed(seed) # fix random seed for reproducibility
    
    # ------------ LOAD DATA: RFMIP ------------ L
    print("Loading data: RFMIP....")
    x_rfmip, x_test, y_rfmip, y_test \
                    = load_inp_outp_RFMIP(fname_rfmip, ychoice, shortwave=shortwave,
                                          remove_CKDMIP=remove_CKDMIP_exps)
    print( "max x[:,3] (water vapor) ; x[:,1] (pres) : {:.2f} | {:.2f}".format(np.max(x_rfmip[:,3]),
                                                                               np.max(x_rfmip[:,1])))
    inds = {}
    
    if include_rfmip:
        x_tr,x_val,y_tr,y_val = train_test_split(x_rfmip, \
                               y_rfmip, test_size = frac_val, random_state = 42)
        y_rfmip_tr_size = y_tr.size
        inds['rfmip'] = [0, y_tr.shape[0]]
        n = y_tr.shape[0]
    else:
        x_tr = x_rfmip[0:1,:]; y_tr = y_rfmip[0:1,:]
        x_val = x_rfmip[0:1,:]; y_val = y_rfmip[0:1,:]
        n = 0
        
    del x_rfmip, y_rfmip
    
    # inds_nwp = [0,1,2,3,4,5,6,8,10,11] # if less_gases=True, use only selected inputs

    ngas  = x_test.shape[1]
    ngpt  = y_test.shape[1]

    
    # ------------ LOAD DATA: NWPSAF ------------ L
    
    if include_nwpsaf: 
        print("Loading data: NWPSAF...")
        x_nwpsaf, y_nwpsaf  = load_inp_outp(fname_nwp, ychoice, dcol=1, 
                                    shortwave=shortwave,remove_CKDMIP=remove_CKDMIP_exps,
                                    skip_lastlev=True)
        
        x_nwpsaf_tr, x_nwpsaf_val, y_nwpsaf_tr, y_nwpsaf_val  = train_test_split(x_nwpsaf,  \
                             y_nwpsaf,  test_size = frac_val, random_state = 42)
        print( "max x[:,3] (water vapor) ; x[:,1] (pres) : {:.2f} | {:.2f}".format(np.max(x_nwpsaf[:,3]),
                                                                                   np.max(x_nwpsaf[:,1])))      
        x_tr = np.concatenate((x_tr,x_nwpsaf_tr),axis=0)
        x_val = np.concatenate((x_val,x_nwpsaf_val),axis=0)
        y_tr =  np.concatenate((y_tr,y_nwpsaf_tr),axis=0)
        y_val =  np.concatenate((y_val,y_nwpsaf_val),axis=0)
        
        y_nwpsaf_tr_size = y_nwpsaf_tr.size
        inds['nwpsaf'] = [n, y_tr.shape[0]]
        n = y_tr.shape[0]

        del x_nwpsaf, y_nwpsaf, x_nwpsaf_tr, x_nwpsaf_val, y_nwpsaf_tr, y_nwpsaf_val
        
    
    # ------------ LOAD DATA: CAMS ------------ L
      
    if include_cams:
        print("Loading data: CAMS...")
        x_cams, y_cams  = load_inp_outp(fname_cams, ychoice, dcol=dcol, 
                                    shortwave=shortwave,remove_CKDMIP=remove_CKDMIP_exps,
                                    skip_lastlev=True)
        print( "max x[:,3] (water vapor) ; x[:,1] (pres) : {:.2f} | {:.2f}".format(np.max(x_cams[:,3]),
                                                                                   np.max(x_cams[:,1])))
        nrows = x_cams.shape[0]
        inds_rand = np.sort(np.random.choice(np.arange(nrows),np.int(frac_cams*nrows),replace=False))
        x_cams = x_cams[inds_rand,:]
        y_cams = y_cams[inds_rand,:]
          
        x_cams_tr, x_cams_val, y_cams_tr, y_cams_val  = train_test_split(x_cams,  \
                             y_cams,  test_size = frac_val, random_state = 42)
            
        x_tr = np.concatenate((x_tr,x_cams_tr),axis=0)
        x_val = np.concatenate((x_val,x_cams_val),axis=0)
          
        y_tr =  np.concatenate((y_tr, y_cams_tr),axis=0)
        y_val =  np.concatenate((y_val, y_cams_val),axis=0)
          
        y_cams_tr_size = y_cams_tr.size
        
        inds['cams'] = [n, y_tr.shape[0]]
        n = y_tr.shape[0]
          
        del x_cams, y_cams, y_cams_tr, y_cams_val, x_cams_tr, x_cams_val
        
      
    if include_GCM: 
        print("Loading data: GCM...")
        x_GCM, y_GCM  = load_inp_outp_GCM(fname_GCM, ychoice, dcol=dcol, 
                                    shortwave=shortwave,remove_CKDMIP=remove_CKDMIP_exps)
        x_GCM_tr, x_GCM_val, y_GCM_tr, y_GCM_val  = train_test_split(x_GCM,  \
                             y_GCM,  test_size = frac_val, random_state = 42)
        
        print( "max x[:,3] (water vapor) ; x[:,1] (pres) : {:.2f} | {:.2f}".format(np.max(x_GCM[:,3]),
                                                                                   np.max(x_GCM[:,1])))
            
        x_tr = np.concatenate((x_tr, x_GCM_tr),axis=0)
        x_val = np.concatenate((x_val, x_GCM_val),axis=0)
          
        y_tr =  np.concatenate((y_tr, y_GCM_tr),axis=0)
        y_val =  np.concatenate((y_val, y_GCM_val),axis=0)
                  
        y_GCM_tr_size = y_GCM_tr.size
        
        inds['gcm'] = [n, y_tr.shape[0]]
        n = y_tr.shape[0]
          
        del x_GCM, y_GCM, y_GCM_tr, y_GCM_val, x_GCM_tr, x_GCM_val 
        
        
    if include_Garand: 
        print("Loading data: Garand...")
        x_garand, y_garand  = load_inp_outp(fname_Garand, ychoice, dcol=1,  
                                    shortwave=shortwave,remove_CKDMIP=remove_CKDMIP_exps)
        print( "max x[:,3] (water vapor) ; x[:,1] (pres) : {:.2f} | {:.2f}".format(np.max(x_garand[:,3]),
                                                                           np.max(x_garand[:,1])))   
        if garand_val: # Use Garand for validation
            # x_garand_tr = x_garand; y_garand_tr = y_garand
            # x_garand_val = np.copy(x_garand_tr); y_garand_val = np.copy(y_garand_tr)
            x_val = np.concatenate((x_val, x_garand),axis=0)
            y_val =  np.concatenate((y_val, y_garand),axis=0)
            y_garand_tr_size = 0
            y_garand_val_size = y_garand.size
        else:
            x_garand_tr, x_garand_val, y_garand_tr, y_garand_val  = train_test_split(x_garand,  \
                                 y_garand,  test_size = frac_val, random_state = 42)
            x_tr = np.concatenate((x_tr, x_garand_tr),axis=0)
            x_val = np.concatenate((x_val, x_garand_val),axis=0)
            y_tr =  np.concatenate((y_tr, y_garand_tr),axis=0)
            y_val =  np.concatenate((y_val, y_garand_val),axis=0)
            y_garand_tr_size = y_garand_tr.size
            y_garand_val_size = y_garand_val.size
            x_garand_tr, x_garand_val, y_garand_tr, y_garand_val
    
        inds['garand'] = [n, y_tr.shape[0]]
        n = y_tr.shape[0]

        # print( "{:.2f}% of the training data comes from garand".format((100*y_garand_tr.size/y_tr.size)))
        del x_garand, y_garand
        
    if include_CKDMIP: 
        print("Loading data: CKDMIP...")
        x_CKDMIP, y_CKDMIP  = load_inp_outp(fname_CKDMIP, ychoice, dcol=1,  
                                    shortwave=shortwave,remove_CKDMIP=remove_CKDMIP_exps)
        x_CKDMIP_tr, x_CKDMIP_val, y_CKDMIP_tr, y_CKDMIP_val  = train_test_split(x_CKDMIP,  \
                             y_CKDMIP,  test_size = frac_val, random_state = 42)
        print( "max x[:,3] (water vapor) ; x[:,1] (pres) : {:.2f} | {:.2f}".format(np.max(x_CKDMIP[:,3]),
                                                                                   np.max(x_CKDMIP[:,1])))   
        x_tr = np.concatenate((x_tr, x_CKDMIP_tr),axis=0)
        x_val = np.concatenate((x_val, x_CKDMIP_val),axis=0)
        y_tr =  np.concatenate((y_tr, y_CKDMIP_tr),axis=0)
        y_val =  np.concatenate((y_val, y_CKDMIP_val),axis=0)
        
        y_CKDMIP_tr_size = y_CKDMIP_tr.size
        inds['CKDMIP'] = [n, y_tr.shape[0]]
        n = y_tr.shape[0]

        # print( "{:.2f}% of the training data comes from CKDMIP".format((100*y_CKDMIP_tr.size/y_tr.size)))
        del x_CKDMIP, y_CKDMIP, x_CKDMIP_tr, x_CKDMIP_val, y_CKDMIP_tr, y_CKDMIP_val
        
    if include_rfmip:
        print( "{:.2f}% of the training data comes from RFMIP".format((100*y_rfmip_tr_size/y_tr.size)))
    if include_nwpsaf:
        print( "{:.2f}% of the training data comes from NWPSAF".format((100*y_nwpsaf_tr_size/y_tr.size)))
    if include_cams:
        print( "{:.2f}% of the training data comes from CAMS".format((100*y_cams_tr_size/y_tr.size)))
    if include_GCM:
        print( "{:.2f}% of the training data comes from GCM".format((100*y_GCM_tr_size/y_tr.size)))
    if include_Garand:
        ratio_tr = 100*y_garand_tr_size/y_tr.size; ratio_val = 100*y_garand_val_size/y_val.size
        print( "{:.2f}% of the training, {:.2f}%  of the validation data comes from Garand".format(ratio_tr,ratio_val))
    if include_CKDMIP:
        print( "{:.2f}% of the training data comes from CKDMIP".format((100*y_CKDMIP_tr_size/y_tr.size)))
        
    size_tot = y_tr.shape[0] + y_val.shape[0] + y_test.shape[0]
    ratio_tr = 100*y_tr.shape[0]/size_tot; ratio_val = 100*y_val.shape[0]/size_tot; ratio_test = 100*y_test.shape[0]/size_tot
    print( "In total, {:.2f}% - {:.2f}% - {:.2f}% of the data is used for training - validation - testing".format(ratio_tr,ratio_val,ratio_test))

    # if less_gases:
    #     x_tr = x_tr[:,inds_nwp]
    #     x_val = x_val[:,inds_nwp]
    #     x_test = x_test[:,inds_nwp]

    ngas  = x_test.shape[1]
    ngpt  = y_tr.shape[1]
    
    print( "x_tr: {}".format(x_tr.shape))
    print( "y_tr: {}".format(y_tr.shape))
    print( "x_val: {}".format(x_val.shape))
    print( "y_val: {}".format(y_val.shape))
    print( "x_test: {}".format(x_test.shape))
    print( "y_test: {}".format(y_test.shape))
    
    return x_tr, x_val, x_test, y_tr, y_val, y_test, ngas, ngpt, inds



def get_available_layers(model_layers, available_model_layers=[b"dense"]):
    parsed_model_layers = []
    for l in model_layers:
        for g in available_model_layers:
            if g in l:
                parsed_model_layers.append(l)
    return parsed_model_layers

# KERAS HDF5 NEURAL NETWORK MODEL FILE TO NEURAL-FORTRAN ASCII MODEL FILE
def h5_to_txt(weights_file_name, output_file_name=''):

    #check and open file
    with h5py.File(weights_file_name,'r') as weights_file:

        weights_group_key=list(weights_file.keys())[0]

        # activation function information in model_config
        model_config = weights_file.attrs['model_config'].decode('utf-8') # Decode using the utf-8 encoding
        model_config = model_config.replace('true','True')
        model_config = model_config.replace('false','False')

        model_config = model_config.replace('null','None')
        model_config = eval(model_config)

        model_layers = list(weights_file['model_weights'].attrs['layer_names'])
        model_layers = get_available_layers(model_layers)
        print("names of layers in h5 file: %s \n" % model_layers)

        # attributes needed for .txt file
        # number of model_layers + 1(Fortran includes input layer),
        #   dimensions, biases, weights, and activations
        num_model_layers = len(model_layers)+1

        dimensions = []
        bias = {}
        weights = {}
        activations = []

        print('Processing the following {} layers: \n{}\n'.format(len(model_layers),model_layers))
        if 'Input' in model_config['config']['layers'][0]['class_name']:
            model_config = model_config['config']['layers'][1:]
        else:
            model_config = model_config['config']['layers']

        for num,l in enumerate(model_layers):
            layer_info_keys=list(weights_file[weights_group_key][l][l].keys())

            #layer_info_keys should have 'bias:0' and 'kernel:0'
            for key in layer_info_keys:
                if "bias" in key:
                    bias.update({num:np.array(weights_file[weights_group_key][l][l][key])})

                elif "kernel" in key:
                    weights.update({num:np.array(weights_file[weights_group_key][l][l][key])})
                    if num == 0:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[0]))
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))
                    else:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))

            if 'Dense' in model_config[num]['class_name']:
                activations.append(model_config[num]['config']['activation'])
            else:
                print('Skipping bad layer: \'{}\'\n'.format(model_config[num]['class_name']))

    if not output_file_name:
        # if not specified will use path of weights_file with txt extension
        output_file_name = weights_file_name.replace('.h5', '.txt')

    with open(output_file_name,"w") as output_file:
        output_file.write(str(num_model_layers) + '\n')

        output_file.write("\t".join(dimensions) + '\n')
        if bias:
            for x in range(len(model_layers)):
                bias_str="\t".join(list(map(str,bias[x].tolist())))
                output_file.write(bias_str + '\n')
        if weights:
            for x in range(len(model_layers)):
                weights_str="\t".join(list(map(str,weights[x].T.flatten())))
                output_file.write(weights_str + '\n')
        if activations:
            for a in activations:
                if a == 'softmax':
                    print('WARNING: Softmax activation not allowed... Replacing with Linear activation')
                    a = 'linear'
                output_file.write(a + "\n")

def savemodel(kerasfile, model):
   model.summary()
   newfile = kerasfile[:-3]+".txt"
   model.save(kerasfile)
   print("saving to {}".format(newfile))
   h5_to_txt(kerasfile,newfile)
   
def create_model(nx,ny,neurons=[40,40], activ0='softsign',activ='softsign',
                 kernel_init='he_uniform',activ_last='linear'):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=nx, kernel_initializer=kernel_init, activation=activ0))
    # hidden layers
    for i in range(1,np.size(neurons)):
      model.add(Dense(neurons[i], activation=activ,kernel_initializer=kernel_init))
      
    model.add(Dense(ny, activation=activ_last,kernel_initializer=kernel_init))
    
    return model

def create_callbacks(fpath,valfunc, patience=10):
    checkpointer = ModelCheckpoint(filepath=fpath, monitor='val_loss',verbose=1,period=1)
    earlystopper = EarlyStopping(monitor=valfunc, patience=patience, verbose=1, mode='min', restore_best_weights=True)
    return checkpointer,earlystopper
  
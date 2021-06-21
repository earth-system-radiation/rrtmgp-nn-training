#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing training data for neural network emulators of 
RRTMGP gas optics scheme; to be run from /examples/ml-training within the 
RTE+RRTMGP library

Another script (1/3) generates INPUT data for RRTMGP (saving to a netCDF file) 
according to user choices and can be run several times with different options 
to create different datasets.

THIS SCRIPT generate corresponding OUTPUTS (optical properties) by calling RRTMGP

Finally, these can be combined with ml_train_emulators (3/3) to train neural networks

Contributions welcome!

@author: Peter Ukkonen
"""

import os
import gc
import numpy as np
from netCDF4 import Dataset
#from sklearn.preprocessing import MinMaxScaler, StandardScaler

# rte_rrtmgp_dir   = os.path.join("..", "..")
this_dir        = os.getcwd() + '/'
ml_example_dir  = this_dir + '../'
#os.chdir(ml_example_dir)
rte_rrtmgp_dir  = ml_example_dir + '../../'
# This files lives in $RRTMGP_ROOT/examples/all-sky/

# --------------------------------------------------------
# ------------ 1. LONGWAVE OR SHORTWAVE ------------------
# --------------------------------------------------------
# Two output variables which depend on this choice will be generated with RRTMGP
# So far the following outputs are supported:
# Longwave (LW) variables
#   "lw_abs" --> Absorption optical depth / cross-section (LW)
#   "lw_planck" --> Planck fraction, needed to compute Planck source functions
#                   from temperature
# Shortwave (SW) variables
#   "sw_abs" --> Absorption optical depth / cross-section (SW)
#   "sw_ray"--> Rayleigh optical depth / cross-section (SW), measuring
#   scattering by gas molecules
# These output variables are 1D arrays which include all g-points, so have sizes
# NGPT which depends on the k-distribution.
spectrum = 'longwave'
#spectrum = 'shortwave'


# RRTMGP provides optical depth, ssa, and tau
# Planck fraction is a temporary variable, meaning the Fortran code needs to be 
# modified so this can be saved (for instance inside the source derived type)
# In the shortwave, sw_abs = (optical_props%ssa * optical_props%tau) 

# Furthermore, if we want to predict cross-section instead of optical depth
# (which I think we should), col_dry needs to be either saved within Fortran or 
# computed here from vmr_h2o and plev like in get_col_dry


# --------------------------------------------------------
# ------------ 2. SPECIFY RRTMGP K-DISTRIBUTION ----------
# --------------------------------------------------------
lw_kdist_file = "rrtmgp-data-lw-g256-2018-12-04.nc"
sw_kdist_file = "rrtmgp-data-sw-g224-2018-12-04.nc"
lw_kdist_path   = rte_rrtmgp_dir + "rrtmgp/data/" + lw_kdist_file
sw_kdist_path   = rte_rrtmgp_dir + "rrtmgp/data/" + sw_kdist_file

if spectrum=='shortwave':
    gas_kdist_path = sw_kdist_path
else:
    gas_kdist_path = lw_kdist_path

kdist           = Dataset(gas_kdist_path)
ngpt            = kdist.dimensions['gpt'].size
# Temperature and pressure range of the LUT, for checking that the input
# data does not exceed these
kdist_temp_ref  = kdist.variables['temp_ref'][:]
kdist_pres_ref  = kdist.variables['press_ref'][:]
kdist_gases_raw = kdist.variables['gas_names'][:]
kdist_gases     = ''.join(str(s, encoding='UTF-8') for s in kdist_gases_raw)
kdist_gases     = kdist_gases.split()
kdist.close()


# --------------------------------------------------------
# ------------ SPECIFY INPUT DATA  -----------------------
# --------------------------------------------------------
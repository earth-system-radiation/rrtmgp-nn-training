#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing training data for neural network emulators of 
RRTMGP gas optics scheme; to be run from /examples/ml-training within the 
RTE+RRTMGP library

This script (1/3) generates INPUT data for RRTMGP (saving to a netCDF file) 
according to user choices and can be run several times with different options 
to create different datasets.
For now it uses some existing data as source, and extends this by hypercube
sampling.

It can be combined with ml_generate_outputs (2/3) to generate corresponding 
OUTPUTS (optical properties) by calling RRTMGP

Finally, these can be combined with ml_train_emulators (3/3) to train neural networks

These files live in $RRTMGP_ROOT/examples/ml-training/scripts_and_data/

Contributions welcome!

@author: Peter Ukkonen
"""

import os
import gc
import numpy as np
from netCDF4 import Dataset
from doepy import build, read_write
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# --------------------------------------------------------
# (2. LOAD RRTMGP K-DISTRIBUTION TO CHECK THAT TEMP, PRES WILL BE in RANGE?)
# --------------------------------------------------------
#Zlw_kdist_file = "rrtmgp-data-lw-g256-2018-12-04.nc"
#sw_kdist_file = "rrtmgp-data-sw-g224-2018-12-04.nc"
#lw_kdist_path   = rte_rrtmgp_dir + "rrtmgp/data/" + lw_kdist_file
#sw_kdist_path   = rte_rrtmgp_dir + "rrtmgp/data/" + sw_kdist_file
#
#if spectrum=='shortwave':
#    gas_kdist_path = sw_kdist_path
#else:
#    gas_kdist_path = lw_kdist_path
#
#kdist           = Dataset(gas_kdist_path)
#ngpt            = kdist.dimensions['gpt'].size
## Temperature and pressure range of the LUT, for checking that the input
## data does not exceed these
#kdist_temp_ref  = kdist.variables['temp_ref'][:]
#kdist_pres_ref  = kdist.variables['press_ref'][:]
#kdist_gases_raw = kdist.variables['gas_names'][:]
#kdist_gases     = ''.join(str(s, encoding='UTF-8') for s in kdist_gases_raw)
#kdist_gases     = kdist_gases.split()
#kdist.close()

# --------------------------------------------------------
# ------------ 3. CHOOSE INITIAL INPUT DATA  -------------
# --------------------------------------------------------
# the inputs for NN training ( gas concentrations, temperature, and pressure)
# will be generated according to the specified method. 

# The range and number of these inputs (how many minor gases are included)
# will depend on the intended application. Emulating the full sensitivities
# of RRTMGP will require sampling a larger hypercube in both the number and size
# of dimensions (corresponding to each input) and therefore lots more data

# For many gases, the concentrations can be sampled independently, but for
# water vapor and probably ozone, the co-dependency with temperature should
# be accounted for.
# Easiest way to do is to provide "real" atmospheric profiles from observations,
# climate models, or a reanalysis.

# PRELIMINARY IMPLEMENTATION: lets use the RFMIP "experiment" structure to sample
# some inputs synthetically, while other variables with the "site" dimension
# assume to be already provided and are not sampled here.

# 'ckdmip_mmm' = "Minimum, maximum, mean" dataset in CKDMIP, which originally
#             includes the MMM profiles of T, H2O, O3 (in present-day climate)
#   'inputs_CKDMIP-MMM-extended.nc' additionally includes MMM values of CH4
#    and CO2 as derived from RFMIP = 5**3 = 243 combinations. 
#    The sampling of other gases (which have the experiment dimension)
#    will be specified by user.
# 'rfmip'      = RFMIP (Radiative Forcing Model Intercomparison Project) data
#             spanning 100 sites around the globe and 18 experiments
# 'cams'       = (NOT YET IMPLEMENTED) CAMS reanalysis data
# 'gcm'        = (NOT YET IMPLEMENTED) climate model data from a high emissions
#                scenario
# 'garand'     = (NOT YET IMPLEMENTED) Garand profiles used in RRTMGP tuning

# 'SYNTHETIC'  = (NOT YET IMPLEMENTED) create data completely on-the-fly by
# specifying gases and their range, as well as the range of temperatures;
# the dependencies T(p), H2O(T,p), O3(T,p) need to be parameterized

input_file_source = 'inputs_CKDMIP_MMM_extended.nc'
#input_file_source = 'inputs_RFMIP_RRTMGP_gases.nc'

inputs_dir = this_dir + 'rrtmgp_inp_outp/'

if(len(input_file_source)>1):
    input_file_path = inputs_dir + input_file_source
    initial_data_provided = True
    ds_source = Dataset(input_file_path)
else:
    initial_data_provided = False
    # Need some way of sampling T, H2O, O3
    import sys
    sys.exit("Need some initial data")


# ----------------------------------------------------------
# ------------ 4. SPECIFY HYPERCUBE SAMPLING OF GASES  -----
# ---------------------------------------------------------
# the inital dataset can be extended synthetically by sampling the concentrations of 
# user-specified gases
# In the longwave, this can amount up to a hypercube with up to 14 dimensions
# since RRTMGP supports so many minor gases!
# We want to cover this huge space evenly with a small number of samples, for 
# which Halton sequences are effective (used also in Ukkonen et al. 2020)
hypercube_sampling = True

# RRTMGP gases for which profiles are already provided and can't be sampled 
varlist_site = []
for variable in ds_source.variables.keys():
    if ((ds_source[variable].dimensions[0] == 'site') and 
    (variable not in ['site_label','temp_level','surface_temperature',
                      'surface_emissivity'])):
         varlist_site.append(variable)     
        
vars_provided = " ".join(varlist_site)

# RRTMGP gases available for hypercube sampling
varlist_expt = []
for variable in ds_source.variables.keys():
    if ((ds_source[variable].dimensions[0] == 'expt') and 
    ('site' not in ds_source[variable].dimensions) and 
    (variable not in ['expt_label','oxygen_GM','nitrogen_GM'])):
         varlist_expt.append(variable)
         
vars_available = " ".join(varlist_expt)     
        
print("Input variables already sampled in "+
      "the provided data (site-dependence): \n {} ".format(vars_provided))
print("---------------------------------------------------------")
print("Input variables AVAILABLE for hypercube sampling "+
      "(expt-dependence): \n {} ".format(vars_available))
print("---------------------------------------------------------")

# USER-SELECTED GASES, THEIR RANGES, AND NUM OF SAMPLES FOR HYPERCUBE SAMPLING
# provided as 'gas_name": [minval,maxval] in the same units as in input_file_source
        
# do NOT include nitrogen or oxygen (considered constant)
nsample = 200         # NuMBER OF HYPERCUBE SAMPLES

#gas_conc_ranges = {  
#                'carbon_dioxide_GM':    [142, 2300.],
#                'methane_GM':           [375, 2500.],
#                'cfc11_GM':             [0, 234.0],
#                'cfc12_GM':             [0, 521.0],
#                'nitrous_oxide_GM':     [200, 389],
#                'carbon_monoxide_GM':   [0, 1.8e-07],
#              }

# CKDMIP source file and CKDMIP style experiments which only include 
# CO2, CH4, N2O, CFC-11-eq and CFC-12: 
gas_conc_ranges = {  
                'cfc11eq':              [0.0,2000.0] , # CFC-11-equivalent
#                'cfc11':              [0.0,234.0] , 
                'cfc12':                [0, 521.0],
                'nitrous_oxide':        [200, 405],
                #'carbon_monoxide_GM':  [0, 1.8e-07],
              }

# RFMIP min-max ranges, where the minimum usually corresponds to Last Glacial
# Maximum, and the maximum is usually a future climate scenario, are
#gas_conc_ranges_rfmip_lw = {
##        'water_vapor': [8.21e-07, 0.0404],    # h2o,      unit 1
##        'ozone': [7.68e-09, 1.15e-05],        # o3,       unit 1
#        'carbon_dioxide_GM': [142, 2.27e+03],  # co2,      unit 1.e-6
#        'nitrous_oxide_GM': [200, 389],        # n2o,      unit 1.e-9
#        'methane_GM': [375, 2.48e+03],         # ch4,      unit 1.e-9
#        'cfc11_GM': [0, 233],                  # cfc11,    unit 1.e-12
##        'cfc11eq_GM': [0, 1.94e+03],
#        'cfc12_GM': [0, 521],                  # cfc12,    unit 1.e-12
##        'cfc12eq_GM': [0, 1.05e+03],
#        'carbon_monoxide_GM': [0, 1.8e-07],    # co,       unit 1
#        'carbon_tetrachloride_GM': [0, 83.1],  # ccl4,     unit 1.e-12
#        'hcfc22_GM': [0, 230],                 # cfc22,    unit 1.e-12
#        'hfc143a_GM': [0, 714],                # hfc143,   unit 1.e-12
#        'hfc125_GM': [0, 966],                 # hfc125,   unit 1.e-12
#        'hfc23_GM': [0, 26.9],                 # hfc23,    unit 1.e-12
#        'hfc32_GM': [0, 8.34],                 # hfc32,    unit 1.e-12
#        'hfc134a_GM': [0, 421],                # hfc134,   unit 1.e-12
#        'cf4_GM': [0, 127],                    # cf4,      unit 1.e-12
#        }
# dat_rfmip = Dataset("multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc")
#for v in dat_rfmip.variables:
#    if ('units' in dat_rfmip.variables[v].ncattrs()):
#        datt = dat_rfmip.variables[v][:].data; unit = dat_rfmip.variables[v].units
#        print("'{}': [{:.3g}, {:.3g}],  # unit {}".format(v,np.min(datt), np.max(datt), unit))


vars_selected = " ".join(gas_conc_ranges.keys())
vars_selected_list = vars_selected.split()

print("Input variables SELECTED for hypercube sampling: (make sure the names "+
      "match the available gases!) \n {} ".format(vars_selected))
print("---------------------------------------------------------")

gas_concs_hypercube = build.halton(gas_conc_ranges, num_samples = nsample)
# We could also use another sampling method supported by DOEPY

# 3D plotting
#var1 = 'cfc11'; var2 = 'cfc12'; var3 ='nitrous_oxide_GM'
#x = gas_concs_hypercube[var1].values; y = gas_concs_hypercube[var2].values
#z = gas_concs_hypercube[var3].values
#fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
#ax.scatter(x, y, z, marker='o'); ax.set_xlabel(var1); ax.set_ylabel(var2); ax.set_zlabel(var3)
#plt.show()


# ---------------------------------------------------------
# ------------ 5. SAVE NEW DATA TO NETCDF  ----------------
# ---------------------------------------------------------
keep_original_experiments = True

# Which experiment is used to copy over variables that vary with both expt and site
# into the new experiments (these variables are not sampled)
iexp_ref = 0

ngases = len(gas_conc_ranges)
print("A {}-dimensional hypercube with n = {} samples was generated ".format(
   ngases,nsample) +
  "using a Halton sequence. For each gas, if that gas has a vertical " +
  "dependency in the initial input file, the dependency is kept in the " +
  "new file by scaling the initial profile at expt=0 so that the " +
  "concentration at the surface corresponds to the sampled value. \n"+
  "For any variables that vary with BOTH expt and site (not sampled), the "+
  "data from iexp_ref={} is copied over to the new experiments".format(iexp_ref))

inp_file_final = inputs_dir + (str(input_file_source.split(".")[0]) +
                  "_hyper_ndim{}_nsamp{}.nc".format(ngases,nsample))

print("\nAttempting to save the new data to {}".format(inp_file_final))

ds_final = Dataset(inp_file_final,"w")

gases_excluded = list(set(vars_available.split()) - set(vars_selected.split()))

vars_selected_shortnames = vars_selected_list.copy()
for ivar in range(0,len(vars_selected_shortnames)):
    var = vars_selected_shortnames[ivar]
    if 'original_name' in ds_source[var].ncattrs():
        vars_selected_shortnames[ivar] = ds_source[var].original_name
vars_selected_shortnames = " ".join(vars_selected_shortnames)

# COPY attributes
for name in ds_source.ncattrs():
    ds_final.setncattr(name, ds_source.getncattr(name))
# COPY dimensions
for name, dimension in ds_source.dimensions.items():
    ds_final.createDimension(
        name, (len(dimension) if not dimension.isunlimited else None))
    
# CREATE variables and COPY all data except for the excluded gases
if keep_original_experiments:
    for name, variable in ds_source.variables.items():
        if (name not in gases_excluded):
            x = ds_final.createVariable(name, variable.datatype, variable.dimensions)
            ds_final[name][:] = ds_source[name][:]
            ds_final[name].setncatts(ds_source[name].__dict__)
    # NEW EXPERIMENTS (HYPERCUBE SAMPLES)
    iexp0 = ds_final.dimensions['expt'].size; iexp1 = iexp0 + nsample
    i = 0
    for iexp in range(iexp0,iexp1):
        # Sampled gases
        for var in vars_selected.split():
            new_val = gas_concs_hypercube[var][i]
            # if the variable has a vertical profile, the new profile
            # will scale this so that the sampled value will be at surface 
            if (len(ds_final[var].dimensions) == 2 and ds_final[var][:].max()>0.0):
                nlay = ds_final[var].shape[1]
                factor = new_val / ds_final[var][0,-1].data
                ds_final[var][iexp,0:nlay] = factor*ds_final[var][0,:]  
            else:
                ds_final[var][iexp] = new_val
        # Other variables with expt-dependence that were not sampled: copy over
        # from iexp_ref
        for var in ds_final.variables.keys():
            dims = ds_final[var].dimensions
            if (var not in vars_selected.split() and dims[0] == 'expt'):
                if (len(dims)==1):
                    ds_final[var][iexp] = ds_final[var][iexp_ref]
                elif (len(dims)==2):
                    (ndim1,ndim2) = ds_final[var].shape
                    ds_final[var][iexp,0:ndim2] = ds_final[var][iexp_ref,0:ndim2]
                elif (len(dims)==3):
                    (ndim1,ndim2,ndim3) = ds_final[var].shape
                    ds_final[var][iexp,0:ndim2,0:ndim3] = ds_final[var][iexp_ref,0:ndim2,0:ndim3]
        ds_final['expt_label'][iexp] = 'Hypercube sample {}/{} varying {}'.format(i+1,nsample,vars_selected_shortnames)
        i = i + 1
                
#else:  # do NOT include original experiments - not yet implemented

#for v in ds_final.variables:
#    print(v); 
#    datt = ds_final.variables[v][:]
#    print("MIN ",np.min(datt),"   MAX ",np.max(datt))  
#    
ds_final.close()
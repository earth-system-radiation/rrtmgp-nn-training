# ml-training-rrtmgp
ML training example built on the RFMIP clear-sky program

Idea is to have streamlined tools we can use to build NN emulators of RRTMGP for a given k-distribution (specifying the number of g-points = number of NN outputs) for given applications.
This requires (presumably lots of) training data consisting of RRTMGP inputs and outputs.

Preliminary implementation: given some user-provided atmospheric profiles with T,p, and gas concentrations,
we can make a larger dataset from this for ML training by hypercube sampling of gas concentrations. The RFMIP
"experiments" can be used to for such samples (variables which vary by "site" are already provided and not sampled). 
Alternatively, we could generate everything synthetically, but then the co dependence of T,P,H2O and O3 needs to be parameterized?

Three scripts located in `scripts_and_data/` (should the first two be combined?):
1. `ml_generate_inputs.py` - create RRTMGP input data (needs data but otherwise self contained)
2. `ml_generate_outputs.py` - (NOT DONE) create RRTMGP output data by calling a modified version of the RFMIP longwave/shortwave Fortran programs
3. `ml_train_emulators.py`- (NOT DONE)  train NNs using the data from 1-2 and user specified choices: complexity could be avoided by using the hyperparameters from Ukkonen 2020 / Menno 2021, with the user just specifying the data, potential early stopping, and basic architecture (number of neurons, layers and activation functions. Use existing code in Keras or write something completely new in PyTorch?


# TO-DO:
- [x] `ml_generate_inputs.py`  tested with two source files in `rrtmgp_inp_outp/` (RFMIP data and extended CKDMIP "MMM" data), seems to work
- [ ] `ml_generate_outputs.py` works and can call RFMIP programs 
- [ ] Modify `mo_rfmip_io.F90` to be more flexible so that it will understand the new input files: load only those RRTMGP gases which are present in the file, which can have the "_GM" suffix or not, and have various shapes not previously supported ((site,layer)). I did something similar [in my NN fork](https://github.com/peterukk/rte-rrtmgp-nn/blob/nn_dev/examples/rfmip-clear-sky/mo_rfmip_io.F90#L70)  but I used column-last dimensions for gas concentrations so the code needs to be changed 
- [ ] Modify `rrtmgp_rfmip_lw.F90`, `rrtmgp_rfmip_sw.F90` and potentially source code to save the RRTMGP outputs to a netCDF file (could be the same as input file, or another file given as command line argument). In Ukkonen 2020 the NN outputs were absorption cross-section (LW/SW), rayleigh cross-section (SW), and Planck fraction (LW). The first two could be computed within Python from regular RRTMGP optical properties (optical depths) and path number of dry air molecules (col_dry).
- [/] Modify source code to get Planck fraction as an optional output (done for kernel `compute_Planck_source`)
- [ ] `ml_train_emulators.py` works and is reasonably flexible
- [ ] Fortran Neural Network code to use the trained models! Part of this repo or another? Peter's previous code can be used but changes are needed to adapt to column-first structure of reference code; making the kernels more generic is also a good idea.


------------

# Original instructions:

1. Build the RTE+RRTMGP libraries in `../../build/`. This will require setting
environmental variables `FC` for the Fortran compiler and `FCFLAGS`.
2. Build the executables in this directory, which will require providing the
locations of the netCDF C and Fortran libraries and module files as environmental
variables `NCHOME` and `NFHOME`, as well a variable `RRTMGP_ROOT` pointing to the root of the installation
(the absolute path to `../../`).
3. Use Python script `stage_files.py` to download relevant files from the
[RFMIP web site](https://www.earthsystemcog.org/projects/rfmip/resources/).This script invokes another Python script to create empty output files.
4. Use Python script `run-rfmip-examples.py` to run the examples. The script takes
some optional arguments, see `run-rfmip-examples.py -h`
5. Python script `compare-to-reference.py` will compare the results to reference
answers produced on a Mac with Intel 19 Fortran compiler. Differences are normally
within 10<sup>-6</sup> W/m<sup>2</sup>.

The Python scripts require modules `netCDF4`, `numpy`, `xarray`, and `dask`.
Install with `pip` requires `pip install dask[array]` for the latter.

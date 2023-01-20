import os
import shutil
import sys
import argparse
path_to_PyHEADTAIL = '/s/ls4/users/kssagan/PyFRIENDS/PyHEADTAIL'#'/home/kiruha/PyHEADTAIL'
#sys.path.append(path_to_PyHEADTAIL)
import numpy as np
from scipy.constants import c, e, m_e, pi
import time
#import pycuda.autoinit
import PyHEADTAIL
#from PyHEADTAIL.general.contextmanager import GPU
import traceback
import pandas as pd
import tempfile

from scipy.constants import e, c, m_e, epsilon_0, hbar

from scipy import interpolate

import matplotlib.pyplot as plt
np.random.seed(int(time.time()))

#from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.longitudinal_tracking import LinearMap, RFSystems
from PyHEADTAIL.particles.particles import Particles
import PyHEADTAIL.particles.generators as generators
from PyHEADTAIL.machines.synchrotron import Synchrotron
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.aperture.aperture import RectangularApertureZ

from Visualisations_new import plot_longitudinal_phase_space_color, plot_sigma_z_sigma_E_mean_z_mean_E,\
                           plot_ex_ey, plot_mx_my, plot_ex_ey_current, plot_mx_my_current, plot_sigma_z_sigma_E_charge
from get_Ekin import get_Ekin
from make_WW import make_WW
from make_Impedance import make_Impedance
from get_WW import get_WW
from make_radiation import make_radiation
from update_bunch import update_bunch
from make_dict import make_dict
from load_obj import load_obj
from save_obj import save_obj
from generate_bunch import generate_bunch
from get_sigma_E import get_sigma_E
from get_parameters_dict import get_parameters_dict
from get_magnetic_structure_twi import get_magnetic_structure_twi
from ParticleLoss import ParticleLoss

from PyHEADTAIL.impedances.impedances_smooth import Impedance, ImpedanceTable
from PyHEADTAIL.particles.slicing import UniformBinSlicer, UniformChargeSlicer

from PyHEADTAIL.radiation.radiation import SynchrotronRadiationTransverse,SynchrotronRadiationLongitudinal

## Reading arguments from command-line

## Reading the parameters of the machine and the beam from the file README.md
parameters_list = ['Energy', 'Circumference', 'Revolution time', 'Betatron tune H',
                  'Betatron tune V', 'Momentum Compaction Factor', 'Chromaticity H', 
                   'Chromaticity V', 'Synchrotron Integral 1', 'Synchrotron Integral 2',
                  'Synchrotron Integral 3', 'Synchrotron Integral 4','Synchrotron Integral 5',
                  'Damping Partition H', 'Damping Partition V', 'Damping Partition E',
                  'Radiation Loss', 'Natural Energy Spread', 'Natural Emittance', 'Radiation Damping H',
                  'Radiation Damping V', 'Radiation Damping E', 'Slip factor', 'Assuming cavities Voltage',
                  'Frequency', 'Harmonic Number', 'Synchronous Phase', 'Synchrotron Tune', 'Bunch Length',
                  'Emitty from Dy', 'Emitty 1/gamma cone limit','spec_dir']
path =  '/home/kiruha/science_repo/compton/'
path_input = path + 'input/'
path_to_readme = path_input + 'README.md' 
parameters_dict = get_parameters_dict(path_to_readme)

## Specifying the paths to the files
spec_dir = parameters_dict['spec_dir']
print(spec_dir)
path =  '/s/ls4/users/kssagan/compton/'#'/home/kiruha/science_repo/compton/'
path_input = path + 'input/'
path_output = path + spec_dir#'output files/'

filename_magnetic = path_input + 'twi.txt'
path_to_obj = path_output + 'obj/'
filename_geom = path_input + 'ebs_geom_full.txt'
filename_rw = path_input + 'ebs_rw_full.txt'
path_to_readme = path_input + 'README.md' 
path_to_fig = path_output + 'figures/'
# Monitors
monitor_path = path_output + 'monitors/'
bunch_filename = monitor_path + 'bunch_mon/'




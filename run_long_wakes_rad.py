import os
## Specifying the paths to the files
spec_dir ='long_wakes_rad/'
#path_to_repo = ''
#path = path_to_repo + 'PyHEADTAIL/projects/'+ spec_dir
path = '/s/ls4/users/kssagan/compton/'#'/home/kiruha/scince_repo/report/'
path_input = path + 'input/'
path_output = path + spec_dir#'output files/'

filename_magnetic = path_input + 'twi.txt'
path_to_obj = path_output + 'obj/'
filename_geom = path_input + 'ebs_geom_full.txt'
filename_rw = path_input + 'ebs_rw_full.txt'
path_to_readme = path_input + 'README.md'
path_to_p_loss = path_output + 'ParticleLoss/' 
# Monitors
monitor_path = path_output + 'monitors/'
bunch_filename = monitor_path + 'bunch_mon/'

for dir_ in [bunch_filename,path_to_obj, path_to_p_loss]:
    if not os.path.exists(dir_):
        try:
            os.makedirs(dir_)
        except:
            pass  
            
import sys
import argparse
path_to_PyHEADTAIL = '/s/ls4/users/kssagan/PyFRIENDS/PyHEADTAIL'#'/home/kiruha/PyHEADTAIL'
sys.path.append(path_to_PyHEADTAIL)
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

from Visualisations import plot_longitudinal_phase_space, plot_sigma_z_sigma_E, plot_ex_ey, plot_mx_my,\
                           plot_ex_ey_current, plot_mx_my_current, plot_sigma_z_sigma_E_current
from get_Ekin import get_Ekin
from make_WW import make_WW
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

from PyHEADTAIL.impedances.wakes import WakeField, WakeTable
from PyHEADTAIL.particles.slicing import UniformBinSlicer, UniformChargeSlicer

from PyHEADTAIL.radiation.radiation import SynchrotronRadiationTransverse,SynchrotronRadiationLongitudinal

## Reading arguments from command-line
def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('i',type=int)
    parser.add_argument('charge_min_nC',type=float)
    parser.add_argument('charge_max_nC',type=float)
    parser.add_argument('n_scan',type=int)
    return parser
    
parser = createParser()
args = parser.parse_args()
charge_min = args.charge_min_nC*1e-9
charge_max = args.charge_max_nC*1e-9


## Reading the parameters of the machine and the beam from the file README.md
parameters_list = ['Energy', 'Circumference', 'Revolution time', 'Betatron tune H',
                  'Betatron tune V', 'Momentum Compaction Factor', 'Chromaticity H', 
                   'Chromaticity V', 'Synchrotron Integral 1', 'Synchrotron Integral 2',
                  'Synchrotron Integral 3', 'Synchrotron Integral 4','Synchrotron Integral 5',
                  'Damping Partition H', 'Damping Partition V', 'Damping Partition E',
                  'Radiation Loss', 'Natural Energy Spread', 'Natural Emittance', 'Radiation Damping H',
                  'Radiation Damping V', 'Radiation Damping E', 'Slip factor', 'Assuming cavities Voltage',
                  'Frequency', 'Harmonic Number', 'Synchronous Phase', 'Synchrotron Tune', 'Bunch Length',
                  'Emitty from Dy', 'Emitty 1/gamma cone limit']

parameters_dict = get_parameters_dict(path_to_readme)

Ekin = parameters_dict['Energy']*1e9
gamma = 1 + Ekin * e / (m_e * c**2)
beta = np.sqrt(1 - gamma**-2)
p0 = beta*c*m_e*gamma

circumference = parameters_dict['Circumference']

t = parameters_dict['Revolution time']*1e-9

Q_x = parameters_dict['Betatron tune H']
Q_y = parameters_dict['Betatron tune V']

alpha_mom_compaction = parameters_dict['Momentum Compaction Factor']

Qpx = parameters_dict['Chromaticity H']
Qpy = parameters_dict['Chromaticity V']
print(f'Chromaticity x: {Qpx}\nChromaticity y: {Qpy}') 

I1 = parameters_dict['Synchrotron Integral 1']
I2 = parameters_dict['Synchrotron Integral 2']
I3 = parameters_dict['Synchrotron Integral 3']
I4 = parameters_dict['Synchrotron Integral 4']
I5 = parameters_dict['Synchrotron Integral 5']

E_loss_ev = parameters_dict['Radiation Loss']*1e6

dE = parameters_dict['Natural Energy Spread']

natural_emmitance = parameters_dict['Natural Emittance']

Radiation_Damping_x = parameters_dict['Radiation Damping H']*1e-3
Radiation_Damping_y = parameters_dict['Radiation Damping V']*1e-3
Radiation_Damping_z = parameters_dict['Radiation Damping E']*1e-3

V_RF = parameters_dict['Assuming cavities Voltage']*1e3

h_RF = parameters_dict['Harmonic Number']

phi_s = np.pi - parameters_dict['Synchronous Phase']
dphi_RF = phi_s
RF_at = 'end_of_transverse'

Q_s = parameters_dict['Synchrotron Tune']

p_increment = 0

sigma_z = parameters_dict['Bunch Length']*1e-3

epsn_x = 7.436459488204655e-09*beta*gamma # [m rad]
epsn_y = 7.436459488204655e-09/14*beta*gamma

## Getting the twiss functions 
s, betax, betay, alphax, alphay, Dx, Dy, accQx, accQy = get_magnetic_structure_twi(filename_magnetic)


## Creating an instance of the machine 
machine = Synchrotron(optics_mode = 'non-smooth',charge= -e,
		        s=s,
		        mass=m_e,
		        p0=p0,
		        name=None,
		        alpha_x=alphax,
		        beta_x=betax,
		        D_x=Dx,
		        alpha_y=alphay,
		        beta_y=betay,
		        D_y=Dy,
		        accQ_x=accQx,
		        accQ_y=accQy,
		        Qp_x = Qpx, 
		        Qp_y = Qpy,
		        longitudinal_mode='non-linear',
		        alpha_mom_compaction=alpha_mom_compaction,
		        h_RF=h_RF,
		        V_RF=V_RF,
		        dphi_RF=dphi_RF,
		        RF_at = RF_at,
		        p_increment=p_increment)
		        
long_map = machine.one_turn_map[-1]
## Creating an instance of the bunch

charge = 1.5e-9
intensity = charge/e
n_macroparticles = int(1e6)

n_scan = 16
bunch_scan = list()
for i in range(n_scan):
    bunch_scan.append(generate_bunch(intensity, n_macroparticles, machine.transverse_map.alpha_x[0], 
                       machine.transverse_map.alpha_y[0],machine.transverse_map.beta_x[0], 
                       machine.transverse_map.beta_y[0], machine.longitudinal_map,
                       machine.transverse_map.D_x[0],machine.transverse_map.D_y[0],
                       sigma_z,gamma,p0,epsn_x,epsn_y,t))

bunch = bunch_scan[0]
bunch_dict = make_dict(bunch)

## Creating an instance of the object responsible for radiation losses
radiation_long, radiation_transverse = make_radiation(E_loss_ev, machine, Ekin, alpha_mom_compaction, 
                                                      epsn_x, epsn_y, Radiation_Damping_z/t,\
                                                      Radiation_Damping_x/t, Radiation_Damping_y/t,I2,I3,I4,None,None)


## Creating an instance of the object associated with wake fields
list_of_wake_sources_long = list()
list_of_wake_sources_x = list()
list_of_wake_sources_y = list()

n_slices = 1000
slicing_mode = 'n_sigma_z'#'fixed_cuts'
fixed_cuts_perc_min_max = 0.5
factor = 0.015
factor_x = 1
factor_y = 1
inverse = -1
n_sigma_z = 3
ratio_interp = 4
NumberPoints = int(3291*ratio_interp)
min_z = -ratio_interp*9e-2
max_z = ratio_interp*9e-2

##geom  
#long                                   
fd, tmp_filename = tempfile.mkstemp(suffix='.txt', text=True)
get_WW(machine.beta, sigma_z, inverse=inverse, factor=factor, del_negative_t = False,
       filename = filename_geom, list_ = ['time','longitudinal'], new_filename = tmp_filename,
       NumberPoints = NumberPoints, min_z = min_z, max_z = max_z)

wake_table_geom_long,slicer = make_WW(tmp_filename, bunch,n_slices = n_slices, 
                                      fixed_cuts_perc_min_max = fixed_cuts_perc_min_max,
                                      list_ = ['time','longitudinal'],  slicing_mode = slicing_mode,
                                      n_sigma_z = n_sigma_z)

list_of_wake_sources_long.append(wake_table_geom_long)
os.close(fd)
os.unlink(tmp_filename)


## rw
#long
fd, tmp_filename = tempfile.mkstemp(suffix='.txt', text=True)
get_WW(machine.beta, sigma_z, inverse=inverse, factor=factor, del_negative_t = False,
       filename = filename_rw, list_ = ['time','longitudinal'], new_filename = tmp_filename,
       NumberPoints = NumberPoints, min_z = min_z, max_z = max_z)

wake_table_rw_long,slicer = make_WW(tmp_filename, bunch, n_slices = n_slices, 
                                    fixed_cuts_perc_min_max = fixed_cuts_perc_min_max,
                                    list_ = ['time','longitudinal'],  slicing_mode = slicing_mode,
                                    n_sigma_z = n_sigma_z)

list_of_wake_sources_long.append(wake_table_rw_long)
os.close(fd)
os.unlink(tmp_filename)


wake_fields_long = WakeField(slicer, *list_of_wake_sources_long)

"""## Putting everything at an instance of our ring (machine.one_turn_map)
machine.one_turn_map.insert(1, wake_fields_long)
machine.one_turn_map.insert(2, wake_fields_x)
machine.one_turn_map.insert(3, wake_fields_y)"""

## Setting Intensity and necessary calculation parameters
charge_scan = np.linspace(charge_min, charge_max, n_scan)
intensity_scan = charge_scan/e
n_turns = int(2e4)
write_every = 10
write_buffer_every = 250
## Values to be recorded in the calculation
bunch_monitor_scan = list()
for charge in charge_scan:
    charge = charge*1e9 
    new_bunch_filename = bunch_filename+f'charge={charge:.3}nC'.replace('.',',')
    bunch_monitor_scan.append(BunchMonitor(
        filename=new_bunch_filename,n_steps=int(n_turns/write_every),
        write_buffer_every=write_every,
        parameters_dict={'Q_x': Q_x,'Q_y':Q_y},
        stats_to_store = [
            'mean_z', 'mean_dp',
            'sigma_z', 'sigma_dp']))

## The function that performs the calculation with different intensities
def run(bunch, intensity, bunch_monitor):  
    update_bunch(bunch,intensity,
                 bunch_dict,beta,gamma,p0)
    for i in range(n_turns):
        long_map.track(bunch)
        wake_fields_long.track(bunch)
        radiation_long.track(bunch)
        if (i+1)%write_every == 0:
            bunch_monitor.dump(bunch)
    return [1]

iterable = list()
for bunch_i,intensity_i, bunch_monitor_i in zip(bunch_scan, intensity_scan, bunch_monitor_scan):
    iterable.append((bunch_i,intensity_i,bunch_monitor_i))

## Let's start
from multiprocessing import Process, Pool
processes = n_scan ## Numbers of threads used for calculation
print('start tracking!')
t0 = time.time()
with Pool(processes = processes) as pool:
    results = list(pool.starmap(run,iterable))
    
print(f'compute time = {time.time()-t0}')

## Plot dependence of sigma z and sigma E on currents
sigma_z_plt = list()
sigma_E_plt = list()
[[sigma_z_plt.append(result[0]), sigma_E_plt.append(result[1])] for result in results]

charge_scan = charge_scan*1e9
fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
line1, = ax1.plot(charge_scan,sigma_z_plt,'-', c='r')
ax1.set_xlabel('charge [nC]')
ax1.set_ylabel('sigma_z [mm]')
ax1.set_title(f'dependence of sigma_z on charge')
ax1.grid()

line3, = ax3.plot(charge_scan,sigma_E_plt,'-', c='r')
ax3.set_xlabel('charge [nC]')
ax3.set_ylabel('sigma_E [1e-4]')
ax3.set_title(f'dependence of sigma_E on current')
ax3.grid()
ax1.scatter(current_scan, sigma_z_plt, color='r', s=20, marker='s')
ax3.scatter(current_scan, sigma_E_plt, color='r', s=20, marker='s')

plt.tight_layout()
plt.show()

fig.savefig(path_to_fig+'sigma_z_sigma_E_charge.jpg')
save_obj(path_to_obj, [sigma_z_plt,sigma_E_plt],'data_for_plot')

print(f'computing time per turn = {(time.time()-t0)/60/n_turns} min')


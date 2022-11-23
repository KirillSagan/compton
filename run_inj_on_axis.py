import os
## Specifying the paths to the files
spec_dir ='inj_on_axis/'
#path_to_repo = ''
#path = path_to_repo + 'PyHEADTAIL/projects/'+ spec_dir
path = '/s/ls4/users/kssagan/compton/'#'/home/kiruha/science_repo/compton/'
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

for dir_ in [bunch_filename,path_to_obj]:
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
import pycuda.autoinit
import PyHEADTAIL
from PyHEADTAIL.general.contextmanager import GPU
import traceback
import pandas as pd
import tempfile

from scipy.constants import e, c, m_e, epsilon_0, hbar

from scipy import interpolate

import matplotlib.pyplot as plt
np.random.seed(int(time.time()))

from PyHEADTAIL.general import pmath as pm
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

def rearrange_arr(arr, index_start, index_end, index_to_insert):
    tmp_arr = arr[index_start:index_end]
    arr = arr[index_end:]
    if index_to_insert != -1:
        arr = np.insert(arr, index_to_insert, tmp_arr)
    else:
        arr = np.append(arr, tmp_arr)
    return arr
    
parser = createParser()
args = parser.parse_args()
charge_min = args.charge_min_nC*1e-9
charge_max = args.charge_max_nC*1e-9
n_scan = args.n_scan
i = args.i

## Reading the parameters of the machine and the beam from the file README.md
parameters_list = ['Energy', 'Circumference', 'Revolution time', 'Betatron tune H',
                  'Betatron tune V', 'Momentum Compaction Factor', 'Chromaticity H', 
                   'Chromaticity V', 'Synchrotron Integral 1', 'Synchrotron Integral 2',
                  'Synchrotron Integral 3', 'Synchrotron Integral 4','Synchrotron Integral 5',
                  'Damping Partition H', 'Damping Partition V', 'Damping Partition E',
                  'Radiation Loss', 'Natural Energy Spread', 'Natural Emittance', 'Radiation Damping H',
                  'Radiation Damping V', 'Radiation Damping E', 'Slip factor', 'Assuming cavities Voltage',
                  'Frequency', 'Harmonic Number', 'Synchronous Phase', 'Synchrotron Tune', 'Bunch Length',
                  'Emitty from Dy', 'Emitty 1/gamma cone limit', 'betaxAve', 'betayAve']

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

Radiation_Damping_x = parameters_dict['Radiation Damping H']
Radiation_Damping_y = parameters_dict['Radiation Damping V']
Radiation_Damping_z = parameters_dict['Radiation Damping E']

V_RF = parameters_dict['Assuming cavities Voltage']*1e3

h_RF = parameters_dict['Harmonic Number']

phi_s = np.pi - parameters_dict['Synchronous Phase']
dphi_RF = phi_s
RF_at = 'end_of_transverse'

Q_s = parameters_dict['Synchrotron Tune']

p_increment = 0

sigma_z = parameters_dict['Bunch Length']*1e-3

septum_dx = 3e-3
epsn_x = 7.436459488204655e-09*beta*gamma # [m rad]
epsn_y = 7.436459488204655e-09/14*beta*gamma

## Getting the twiss functions 
s, betax, betay, alphax, alphay, Dx, Dy, accQx, accQy = get_magnetic_structure_twi(filename_magnetic)


betax_avr = parameters_dict['betaxAve']
betay_avr = parameters_dict['betayAve']
n_betax_avr = 10
n_betay_avr = 8

n_max_Dx = np.argmax(Dx)
betax = rearrange_arr(betax,0,n_max_Dx,-1)
betay = rearrange_arr(betay,0,n_max_Dx,-1)
alphay = rearrange_arr(alphay,0,n_max_Dx,-1)
Dx = rearrange_arr(Dx,0,n_max_Dx,-1)
Dy = rearrange_arr(Dy,0,n_max_Dx,-1)
accQx = rearrange_arr(accQx,0,n_max_Dx,-1)
accQy = rearrange_arr(accQy,0,n_max_Dx,-1)

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

n_RF_insert = 49
machine.one_turn_map.insert(n_RF_insert,machine.longitudinal_map)
machine.one_turn_map.pop(-1)		        
## Creating an instance of the bunch

S = 3e-3 #Septum thickness
Dx_max = Dx.max()
p_inj = S/Dx_max*p0+p0
E_inj = ((p_inj*c)**2+(m_e*c**2)**2)**0.5 
Ekin_inj = (E_inj - m_e*c**2)/e
gamma_inj = E_inj/(m_e*c**2)
beta_inj = np.sqrt(1 - gamma_inj**-2)

charge = 1.5e-9
intensity = charge/e
n_macroparticles = int(1e5)

bunch = generate_bunch(intensity, n_macroparticles, machine.transverse_map.alpha_x[0], 
               machine.transverse_map.alpha_y[0],machine.transverse_map.beta_x[0], 
               machine.transverse_map.beta_y[0], machine.longitudinal_map,
               machine.transverse_map.D_x[0],machine.transverse_map.D_y[0],
               sigma_z,gamma,p0,epsn_x,epsn_y,t)

bunch.dp += (p_inj-p0)/p0
bunch.x  += machine.transverse_map.D_x[0]*(p_inj-p0)/p0

bunch_dict = make_dict(bunch)

## Creating an instance of the object responsible for radiation losses
radiation_long, radiation_transverse = make_radiation(E_loss_ev, machine, Ekin, alpha_mom_compaction, 
                                                      epsn_x, epsn_y, Radiation_Damping_z/t,\
                                                      Radiation_Damping_x/t, Radiation_Damping_y/t,I2,I3,I4,Dx[-1],Dy[-1])



## Setting Intensity and necessary calculation parameters
charge_scan = np.linspace(charge_min, charge_max, n_scan)
charge = charge_scan[i]
intensity = charge/e
n_turns = int(5e7)
write_every = 20
write_buffer_every = 1000
## Values to be recorded in the calculation
#cons = range(1,11)
#names_long = ['S_n_long_'+f'{i}' for i in cons]
#names_trans = ['S_n_trans_'+f'{i}' for i in cons]
#names_trans_y = ['S_n_trans_y_'+f'{i}' for i in cons]
#bunch_monitor_scan = list()

charge = charge*1e9 
new_bunch_filename = bunch_filename+f'charge={charge:.3}nC'.replace('.',',')
bunch_monitor = BunchMonitor(
		filename=new_bunch_filename,n_steps=int(n_turns/write_every),
		write_buffer_every=write_every,
		parameters_dict={'Q_x': Q_x,'Q_y':Q_y},
		stats_to_store = [
		    'mean_z', 'mean_dp','mean_x','mean_y',
		    'sigma_z', 'sigma_dp', 'sigma_x','sigma_y',
		    'epsn_x', 'epsn_y'])


## Let's start
print('start tracking!')
t0 = time.time()

try:
    update_bunch(bunch, intensity,
                 bunch_dict, beta, gamma, p0)
    print(f'intensity = {intensity:.3e}')
    for i in range(n_turns):
        if (time.time()-t0)/60/60 >= (24*3-0.1):
            raise RuntimeError      
        machine.longitudinal_map.track(bunch)
        radiation_long.track(bunch)
        if (i+1)%write_every == 0:
            bunch_monitor.dump(bunch)
except:
    filename_err = path_to_obj + f'charge={charge:.3e}nC_err_logs.txt'.replace('.',',')
    log_info = traceback.format_exc()
    print(log_info)
    with open(filename_err, 'w') as f:
        f.write(log_info)
    
finally:
    print(f'Qpx = {Qpx}\tQpy = {Qpy}\nCharge={charge:.3}nC\nTurn={i}\nComputing time per turn = {(time.time()-t0)/60/n_turns} min')
    bunch_dict = make_dict(bunch)
    save_obj(path=path_to_obj, obj=bunch_dict, name=f'turns={i},charge={charge:.3e}nC')

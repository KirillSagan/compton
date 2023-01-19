import os
import shutil
## Specifying the paths to the files
spec_dir ='long_10sigma_1e6_precise/'
#path_to_repo = ''
#path = path_to_repo + 'PyHEADTAIL/projects/'+ spec_dir
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

for dir_ in [path_to_obj, path_to_fig, bunch_filename]:
    if not os.path.exists(dir_):
        try:
            os.makedirs(dir_)
        except:
            pass
    else:
        shutil.rmtree(dir_)  
        os.makedirs(dir_)   
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
from PyHEADTAIL.aperture.aperture import RectangularApertureZ

from Visualisations_new import plot_longitudinal_phase_space_color, plot_sigma_z_sigma_E_mean_z_mean_E,\
                           plot_ex_ey, plot_mx_my, plot_ex_ey_current, plot_mx_my_current, plot_sigma_z_sigma_E_charge
from get_Ekin import get_Ekin
from make_WW import make_WW
from make_Impedance_smooth_precise import make_Impedance
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

from PyHEADTAIL.impedances.impedances_smooth_precise import Impedance, ImpedanceTable
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
n_scan = args.n_scan
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

sigma_z =1.5e-3#parameters_dict['Bunch Length']*1e-3

epsn_x = 7.436459488204655e-09*beta*gamma # [m rad]
epsn_y = 7.436459488204655e-09*beta*gamma

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
list_of_impedance_sources_long = list()
list_of_impedance_sources_x = list()
list_of_impedance_sources_y = list()

n_slices = 2000
slicing_mode = 'n_sigma_z'#'fixed_cuts'
fixed_cuts_perc_min_max = 0.5
factor = 0.015
factor_x = 1
factor_y = 1
inverse = -1
n_sigma_z = 10
ratio_interp = 1.5
NumberPoints = None
min_z = None
max_z = None

##geom  
#long                                   
fd, tmp_filename = tempfile.mkstemp(suffix='.txt', text=True)
get_WW(machine.beta, sigma_z, inverse=inverse, factor=factor, del_negative_t = False,
       filename = filename_geom, list_ = ['time','longitudinal'], new_filename = tmp_filename,
       NumberPoints = NumberPoints, min_z = min_z, max_z = max_z)

impedance_table_geom_long,slicer = make_Impedance(tmp_filename, bunch,n_slices = n_slices, 
                                      fixed_cuts_perc_min_max = fixed_cuts_perc_min_max,
                                      list_ = ['time','longitudinal'],  slicing_mode = slicing_mode,
                                      n_sigma_z = n_sigma_z)

list_of_impedance_sources_long.append(impedance_table_geom_long)
os.close(fd)
os.unlink(tmp_filename)


## rw
#long
fd, tmp_filename = tempfile.mkstemp(suffix='.txt', text=True)
get_WW(machine.beta, sigma_z, inverse=inverse, factor=factor, del_negative_t = False,
       filename = filename_rw, list_ = ['time','longitudinal'], new_filename = tmp_filename,
       NumberPoints = NumberPoints, min_z = min_z, max_z = max_z)

impedance_table_rw_long,slicer = make_Impedance(tmp_filename, bunch, n_slices = n_slices, 
                                    fixed_cuts_perc_min_max = fixed_cuts_perc_min_max,
                                    list_ = ['time','longitudinal'],  slicing_mode = slicing_mode,
                                    n_sigma_z = n_sigma_z)

list_of_impedance_sources_long.append(impedance_table_rw_long)
os.close(fd)
os.unlink(tmp_filename)


Impedance_long = Impedance(slicer, *list_of_impedance_sources_long)

"""## Putting everything at an instance of our ring (machine.one_turn_map)
machine.one_turn_map.insert(1, wake_fields_long)
machine.one_turn_map.insert(2, wake_fields_x)
machine.one_turn_map.insert(3, wake_fields_y)"""

# Creating z Aperture 
z_lost = 25e-3
Aperture_z = RectangularApertureZ(z_low = -z_lost, z_high = z_lost)

## Setting Intensity and necessary calculation parameters
charge_scan = np.linspace(charge_min, charge_max, n_scan)
intensity_scan = charge_scan/e
n_turns = int(5e4)
write_every = 5
write_buffer_every = 500
write_obj_every = 10000
check_aperture_every = 50
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
def run(bunch, intensity,bunch_monitor):  
    charge = intensity*e
    sigma_z_scan = list()
    sigma_E_scan = list()
    mean_z_scan = list()
    mean_E_scan = list()
    update_bunch(bunch,intensity,
                 bunch_dict,beta,gamma,p0)
    
    sigma_z_scan.append(bunch.sigma_z())
    sigma_E_scan.append(bunch.sigma_dp())
    mean_z_scan.append(bunch.mean_z())
    mean_E_scan.append(bunch.mean_dp())
    
    bunch_dict_new = make_dict(bunch)
    try:
        for i in range(n_turns):
            long_map.track(bunch)
            Impedance_long.track(bunch)
            radiation_long.track(bunch)
            if (i+1)%check_aperture_every == 0:
                Aperture_z.track(bunch)
            if (i+1)%write_every == 0:
                sigma_z_scan.append(bunch.sigma_z())
                sigma_E_scan.append(bunch.sigma_dp())
                mean_z_scan.append(bunch.mean_z())
                mean_E_scan.append(bunch.mean_dp())
                bunch_monitor.dump(bunch)
            if (i+1)%write_obj_every == 0:
                bunch_dict_new = make_dict(bunch)
                save_obj(path_to_obj,bunch_dict_new,f'bunch_data_charge={intensity*e*1e9:.3}nC_turn={i}')
            if (i+1)%n_turns == 0:
                plot_sigma_z_sigma_E_mean_z_mean_E(np.array(sigma_z_scan),np.array(sigma_E_scan),
                                                np.array(mean_z_scan), np.array(mean_E_scan), n_turns,
                                                write_every, charge,path=path_to_fig, savefig = True) 
                plot_longitudinal_phase_space_color(bunch, charge, path=path_to_fig, savefig = True,
                                                    name=f'longitudinal_phase_space_after_{n_turns}_turns')
    except:
        filename_err = path_to_obj + f'charge={charge*1e9:.3e}nC_err_logs.txt'.replace('.',',')
        log_info = traceback.format_exc()
        print(log_info)
        with open(filename_err, 'w') as f:
            f.write(log_info)
            
    return [np.array(sigma_z_scan[-100:-1]).mean(), np.array(sigma_E_scan[-100:-1]).mean()]

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

plot_sigma_z_sigma_E_charge(np.array(sigma_z_plt),np.array(sigma_E_plt),charge_scan, 
                                 path=path_to_fig,name=f'sigma_z_sigma_E_charges_after_{n_turns}_turns.jpg')

save_obj(path_to_obj, [sigma_z_plt,sigma_E_plt],'data_for_plot')

print(f'computing time per turn = {(time.time()-t0)/60/n_turns} min')


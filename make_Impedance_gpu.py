
from PyHEADTAIL.impedances.impedances_gpu import Impedance, ImpedanceTable
from PyHEADTAIL.particles.slicing import UniformBinSlicer, UniformChargeSlicer
import numpy as np


def make_Impedance(filename, bunch,
            n_slices, n_sigma_z = None, fixed_cuts_perc_min_max = None,
            list_ = ['time','longitudinal','dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'],
            slicing_mode = 'n_sigma_z'):
    try:
        wake_file_columns = list_
        table = ImpedanceTable(filename, wake_file_columns)
    except ValueError as exc:
        print ('Error message:\n' + str(exc))
        
    if slicing_mode == 'fixed_cuts':
        initial_cut_tail_z = np.min(bunch.z) - fixed_cuts_perc_min_max*(np.max(bunch.z)-np.min(bunch.z))
        initial_cut_head_z = np.max(bunch.z) + fixed_cuts_perc_min_max*(np.max(bunch.z)-np.min(bunch.z))
        slicer = UniformBinSlicer(n_slices, z_cuts=(initial_cut_tail_z,initial_cut_head_z))
    elif slicing_mode == 'n_sigma_z':
        slicer = UniformBinSlicer(n_slices=n_slices, n_sigma_z=n_sigma_z)
    else:
        print('Smth goes wrong with slicing mode')
        
    return table, slicer

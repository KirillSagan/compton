import pandas as pd
import numpy as np
from scipy.constants import c, e, m_e, pi

def get_magnetic_structure_twi(filename_magnetic):
    data = np.loadtxt(filename_magnetic, skiprows = 84, usecols = (0,1,2,3,4,7,8,9,10))
    length = np.diff(data[:,0])
    data = data[1:,:]
    data = data[length>0]

    s = np.concatenate(([0], data[:,0]))
    betax = np.concatenate((data[:,1], [data[0,1]]))
    alphax = np.concatenate((data[:,2], [data[0,2]]))
    accQx = np.concatenate(([0], data[:,3])) / (2*np.pi)
    Dx = np.concatenate((data[:,4], [data[0,4]]))
    betay = np.concatenate((data[:,5], [data[0,5]]))
    alphay = np.concatenate((data[:,6], [data[0,6]]))
    accQy = np.concatenate(([0], data[:,7])) / (2*np.pi)
    Dy = np.concatenate((data[:,8], [data[0,8]]))
    
    return [np.array(s), np.array(betax), np.array(betay), np.array(alphax), 
            np.array(alphay), np.array(Dx), np.array(Dy), np.array(accQx), np.array(accQy)]

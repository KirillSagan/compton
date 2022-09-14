import numpy as np
from scipy.constants import m_e,c,e

def get_Ekin(bunch):
    p = bunch.dp*bunch.p0+bunch.p0
    return np.sqrt((p*c)**2+(m_e*c**2)**2)/e

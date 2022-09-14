from scipy.constants import c, e, m_e
from get_Ekin import get_Ekin
import numpy as np
def get_sigma_E(bunch):
    E_kin0 = np.sqrt((bunch.p0*c)**2+(m_e*c**2)**2)/e
    Ekin_norm = get_Ekin(bunch)/E_kin0
    return Ekin_norm.std()*1e4

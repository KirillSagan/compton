import PyHEADTAIL.particles.generators as generators
import numpy as np
from scipy.constants import c,e,m_e
def generate_bunch(intensity, n_macroparticles, alpha_x, alpha_y, beta_x, beta_y, linear_map,Dx,Dy,sigma_z,gamma,p0,epsn_x,epsn_y, t, k = 1):

    beta_z = (linear_map.eta(dp=0, gamma=gamma) * linear_map.circumference / 
              (2 * np.pi * linear_map.Q_s))
    epsn_z = 4 * np.pi * sigma_z**2 * p0 / (beta_z * e) * k
    beta_z = 4 * np.pi * sigma_z**2 * p0 / (epsn_z * e)
    bunch = generators.generate_Gaussian6DTwiss(
        macroparticlenumber=n_macroparticles, intensity=intensity, charge=-e,
        gamma=gamma, mass=m_e, circumference=linear_map.circumference,
        alpha_x=alpha_x, beta_x=beta_x, epsn_x=epsn_x,
        alpha_y=alpha_y, beta_y=beta_y, epsn_y=epsn_y,
        beta_z=beta_z, epsn_z=epsn_z,dispersion_x=Dx,
        dispersion_y=Dy)
    
    return bunch 

import pandas as pd
import numpy as np
from scipy.constants import c, e, m_e, pi
from scipy.interpolate import interp1d

def get_magnetic_structure_interp(filename_magnetic, Q_x, Q_y, n_interp = 100):
    n_cells = 40
    n_interp_precise = int(1e4)
    row_data = pd.read_excel(filename_magnetic)

    data = row_data[row_data.Length > 0]
    s = list(data.s)
    s_interp = np.linspace(s[0], s[-1], n_interp)
    s_interp_precise = np.linspace(s[0], s[-1], n_interp_precise)
    s_interp_precise_full = np.linspace(s[0], s[-1]*n_cells, n_interp_precise*n_cells)
    s_interp_full = np.linspace(s[0], s[-1]*n_cells, n_interp*n_cells)

    
    length  = np.array(data.Length)
    R = length/(2*np.pi)
    R_interp = interp1d(s,R)
    
    
    betax = list(data.betax)
    betax_interp = interp1d(s,betax)
    betax_interp_full = list(betax_interp(s_interp))*n_cells
    betax_interp_full = betax_interp_full + [betax_interp_full[0]]

    accQx = []
    summ = 0 
    for beta,dR in zip(list(betax_interp(s_interp_precise))*n_cells,\
                       list(R_interp(s_interp_precise))*n_cells):
        summ += dR/beta
        accQx.append(summ)   
    accQx_interp = interp1d(s_interp_precise_full, accQx)(s_interp_full)
    accQx_interp = np.concatenate(([0], accQx_interp))
    accQx_interp[-1] = Q_x
    
    
    betay = list(data.betaz)
    betay_interp = interp1d(s,betay)
    betay_interp_full = list(betay_interp(s_interp))*n_cells
    betay_interp_full = betay_interp_full + [betay_interp_full[0]]

    accQy = []
    summ = 0 
    for beta,dR in zip(list(betay_interp(s_interp_precise))*n_cells,\
                       list(R_interp(s_interp_precise))*n_cells):
        summ += dR/beta
        accQy.append(summ)   
    accQy_interp = interp1d(s_interp_precise_full,accQy)(s_interp_full)
    accQy_interp = np.concatenate(([0], accQy_interp))
    accQy_interp[-1] = Q_y
  

    Dx = data.D
    Dx_interp = list(interp1d(s,Dx)(s_interp))*n_cells
    Dx_interp = np.concatenate((Dx_interp,[Dx_interp[0]]))

    Dpx = data.Dp
    Dpx_interp = list(interp1d(s,Dpx)(s_interp))*n_cells
    Dpx_interp = np.concatenate((Dpx_interp,[Dpx_interp[0]]))*0

    Dy_interp = np.array(Dx_interp)*0
    Dpy_interp = np.array(Dpx_interp)*0
    
    
    betax_diff = np.diff(betax+[betax[0]])
    derivative_betax = betax_diff/length
    
    alphax = -0.5*derivative_betax
    alphax_interp = list(interp1d(s,alphax)(s_interp))*n_cells
    alphax_interp = alphax_interp + [alphax_interp[0]] 
    
    betay_diff = np.diff(betay + [betay[0]])
    derivative_betay = betay_diff/length
    
    alphay = -0.5*derivative_betay
    alphay_interp = list(interp1d(s,alphay)(s_interp))*n_cells
    alphay_interp = alphay_interp + [alphay_interp[0]]
    
    s_interp_full = np.concatenate(([0], s_interp_full))
    
    return [np.array(s_interp_full), np.array(betax_interp_full),
            np.array(betay_interp_full), np.array(alphax_interp), 
            np.array(alphay_interp), np.array(Dx_interp), np.array(Dy_interp),
            np.array(accQx_interp), np.array(accQy_interp)]

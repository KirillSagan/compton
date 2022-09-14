import numpy as np
from PyHEADTAIL.radiation.radiation import SynchrotronRadiationTransverse,SynchrotronRadiationLongitudinal
from scipy.constants import e, c, m_e, epsilon_0, hbar

def make_radiation(E_loss_ev, machine, Ekin,alpha_mom_compaction, 
                   eq_emit_x, eq_emit_y,damping_time_z_turns, 
                   damping_time_x_turns, damping_time_y_turns,
                   I2,I3,I4,D_x,D_y):

  
    Cq = 55.0 * hbar / (32.0 * np.sqrt(3.0) * m_e * c)
     
    eq_sig_dp = np.sqrt( Cq * machine.gamma**2.0 * I3 / (2.0*I2+I4) ) / machine.beta**2.0

    radiation_long = SynchrotronRadiationLongitudinal(
            eq_sig_dp=eq_sig_dp, damping_time_z_turns=damping_time_z_turns, 
            E_loss_eV=E_loss_ev, D_x=D_x, D_y=D_y)
    
    radiation_transverse = SynchrotronRadiationTransverse(
            eq_emit_x=eq_emit_x, eq_emit_y=eq_emit_y, 
            damping_time_x_turns=damping_time_x_turns, 
            damping_time_y_turns=damping_time_y_turns, 
            beta_x=machine.transverse_map.beta_x.mean(), 
            beta_y=machine.transverse_map.beta_y.mean())
    print(f'eq_sig_dp = {eq_sig_dp:.2e}')

    return radiation_long, radiation_transverse

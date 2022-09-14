import numpy as np
from scipy.constants import c, e, m_e

def del_negative_t_func(row_data,i, wake_type, time_i):
    len_index_negative = len(np.where(row_data[:,time_i]<0)[0])
    if len_index_negative > 0:
        print('There is negative t!')
    else:
        len_index_negative = 1

    if wake_type == 'longitudinal':
        value_at_zero_time = row_data[len_index_negative,i]
    else:
        value_at_zero_time = 0
            
    wake_data = row_data.copy()[len_index_negative-1:,i]
    wake_data[0] = value_at_zero_time
    
    time = row_data.copy()[len_index_negative-1:,time_i]
    time[0] = 0.
    return time, wake_data 
    
      
def get_WW(beta, sigma_z, list_ = [], filename = None,min_z=None, max_z=None,
           NumberPoints=None, new_filename = 'prepared_data.txt', del_negative_t = True, 
           skiprows = 0, inverse = 1, factor = 1, factor_x = None, factor_y = None,
           list_wake_types_in_file = ['time','longitudinal','dipole_x', 'dipole_y', 
                                      'quadrupole_x', 'quadrupole_y'] ):
    
    
    convert_to_V_per_C = 1e12
    convert_to_V_per_Cm = 1e15
    
    dict_wake_i = dict()
    for wake_type in list_:
        dict_wake_i[wake_type] = list_wake_types_in_file.index(wake_type)

    row_data = np.loadtxt(filename,skiprows = skiprows)
    time_i = dict_wake_i.pop('time')
    z_range = row_data[:,time_i]
    
    linspace_max_z = z_range
    if NumberPoints != None and min_z != None and max_z != None:
        linspace_max_z = np.linspace(min_z, max_z, NumberPoints)
    tmp_list = list([linspace_max_z])
    
    
    for wake_type, i in dict_wake_i.items():
        
        index_nan = np.where(np.isnan(row_data[:,i])==True)[0]
        if len(index_nan)>0:
            print('There is nan value!')
            
        if factor_x != None and (wake_type == 'dipole_x' or wake_type == 'quadrupole_x'):
            row_data[:,i] = row_data[:,i]*factor_x   
            
        if factor_y != None and (wake_type == 'dipole_y' or wake_type == 'quadrupole_y'):
            row_data[:,i] = row_data[:,i]*factor_y 
        
        if wake_type == 'longitudinal':
            row_data[:,i] = row_data[:,i]/convert_to_V_per_C
            print('There are long wake')
        else:
            row_data[:,i] = row_data[:,i]/convert_to_V_per_Cm
            
        wake_data = row_data[:,i]
        if del_negative_t:
            z_range, wake_data = del_negative_t_func(row_data, i, wake_type, time_i)
            
        tmp_list.append(np.interp(linspace_max_z, z_range, wake_data, right = 0, left = 0))
    
    data = np.array(tmp_list).transpose()
    
    convert_to_ns = 1e9
    data[:,0] = data[:,0]/(c*beta)*convert_to_ns
    data[:,1:] = data[:,1:]*factor*inverse
    
    np.savetxt(new_filename,data)

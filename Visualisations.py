from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def plot_transerse_phase_space(bunch,current, path=None,name='transerse_phase_space'):
	fig, (ax1, ax2) = plt.subplots(2, figsize=(7,12))
	ax1.scatter(bunch.x, bunch.xp, marker='o', s = 0.05)
	ax2.scatter(bunch.y, bunch.yp, marker='o', s = 0.05)
	ax1.set_xlabel('x [m]')
	ax1.set_ylabel('xp')
	ax1.set_title(name.strip('.jpg'))
	
	ax2.set_title(name.strip('.jpg'))
	ax2.set_xlabel('y [m]')
	ax2.set_ylabel('yp')
	ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
	ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
	plt.tight_layout()
	plt.show()
	if path != None:
		fig.savefig(path+name+f'current = {current:.2e} mA'.replace('.',',')+'.jpg')


def plot_particle_loss(dict_, name=None, path=None):
	fig, ax1 = plt.subplots(1, figsize=(7,7))
	bunch_x = dict_['x']
	bunch_y = dict_['y']
	ax1.scatter(bunch_x, bunch_y, marker='o', s = 1)
	ax1.set_xlabel('x [m]')
	ax1.set_ylabel('y [m]')
	ax1.set_title(name.strip('.jpg'))

	plt.tight_layout()
	if path != None:
		fig.savefig(path+name+'.jpg')
	
	
def plot_longitudinal_phase_space(bunch,current,path=None,name='longitudinal_phase_space'):
    fig, (ax1) = plt.subplots(1)
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.scatter(bunch.z, bunch.dp, marker='o', s =0.05)
    ax1.set_xlabel('z [m]')
    ax1.set_ylabel('dp')
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.set_title(name.strip('.jpg'))
    plt.tight_layout()                                          
    plt.show()
    if path != None:
        fig.savefig(path+name+f'current = {current:.3e} mA'.replace('.',',')+'.jpg')
        

def plot_sigma_z_sigma_E(sigma_z_scan,sigma_E_scan,n_turns,write_every,
                         current,path=None,name='sigma_z_sigma_E'):
    x = np.arange(0, n_turns+1, write_every)
    fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.plot(x,sigma_z_scan,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('sigma_z [mm]')
    ax1.set_title(f'dependence of sigma_z on turns')
    ax1.grid()

    ax3.plot(x,sigma_E_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('sigma_E [1e-4]')
    ax3.set_title(f'dependence of sigma_E on turns')
    ax3.grid()
    
    ax1.legend(title = f'current {current:.2e} mA', facecolor = 'w')
    ax3.legend(title = f'current {current:.2e} mA', facecolor = 'w')

    ax1.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    plt.show()
    if path != None:
        fig.savefig(path+name+f' current = {current:.3e} mA'.replace('.',',')+'.jpg')

def plot_ex_ey(ex_scan,ey_scan,n_turns,write_every,
                         current,path=None,name='ex_ey'):
    x = np.arange(0, n_turns+1, write_every)
    fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.plot(x,ex_scan,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('epsn_x [m rad]')
    ax1.set_title(f'dependence of epsn_x on turns')
    ax1.grid()

    ax3.plot(x,ey_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('epsn_y [m rad]')
    ax3.set_title(f'dependence of epsn_y on turns')
    ax3.grid()
    
    ax1.legend(title = f'current {current:.2e} mA', facecolor = 'w')
    ax3.legend(title = f'current {current:.2e} mA', facecolor = 'w')

    ax1.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    plt.show()
    if path != None:
        fig.savefig(path+name+f' current = {current:.2e} mA'.replace('.',',')+'.jpg')
    
def plot_mx_my(mx_scan,my_scan,n_turns,write_every,
                         current,path=None,name='mx_my'):
    x = np.arange(0, n_turns+1, write_every)
    mx_scan = mx_scan*1e3
    my_scan = my_scan*1e3
    fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.plot(x,mx_scan,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('mean_x [mm]')
    ax1.set_title(f'dependence of mean_x on turns')
    ax1.grid()

    ax3.plot(x,my_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('mean_y [mm]')
    ax3.set_title(f'dependence of epsn_y on turns')
    ax3.grid()
    
    ax1.legend(title = f'current {current:.2e} mA', facecolor = 'w')
    ax3.legend(title = f'current {current:.2e} mA', facecolor = 'w')

    ax1.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    plt.show()
    if path != None:
        fig.savefig(path+name+f' current = {current:.2e} mA'.replace('.',',')+'.jpg')

def plot_sigma_z_sigma_E_current(sigma_z_plt,sigma_E_plt,current_scan, 
                                 path=None,name='sigma_z_sigma_E_headtail.jpg'):

    fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.plot(current_scan,sigma_z_plt,'-', c='r')
    ax1.set_xlabel('current [mA]')
    ax1.set_ylabel('sigma_z [mm]')
    ax1.set_title(f'dependence of sigma_z on current')
    ax1.grid()

    ax3.plot(current_scan,sigma_E_plt,'-', c='r')
    ax3.set_xlabel('current [mA]')
    ax3.set_ylabel('sigma_E [1e-4]')
    ax3.set_title(f'dependence of sigma_E on current')
    ax3.grid()
    ax1.scatter(current_scan, sigma_z_plt, color='r', s=20, marker='s')
    ax3.scatter(current_scan, sigma_E_plt, color='r', s=20, marker='s')

    plt.tight_layout()
    plt.show()
    if path != None:
        fig.savefig(path+name)

def plot_mx_my_current(mx_plt,my_plt,current_scan, 
                                 path=None,name='mx_my_current.jpg'):
    mx_plt = mx_plt*1e3
    my_plt = my_plt*1e3

    fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.plot(current_scan,mx_plt,'-', c='r')
    ax1.set_xlabel('current [mA]')
    ax1.set_ylabel('mean_x [mm]')
    ax1.set_title(f'dependence of mean_x on current')
    ax1.grid()

    ax3.plot(current_scan,my_plt,'-', c='r')
    ax3.set_xlabel('current [mA]')
    ax3.set_ylabel('mean_y [mm]')
    ax3.set_title(f'dependence of mean_y on current')
    ax3.grid()
    ax1.scatter(current_scan, mx_plt, color='r', s=20, marker='s')
    ax3.scatter(current_scan, my_plt, color='r', s=20, marker='s')

    plt.tight_layout()
    plt.show()
    if path != None:
        fig.savefig(path+name)

def plot_ex_ey_current(ex_plt,ey_plt,current_scan, 
                                 path=None,name='ex_ey_current.jpg'):
    ex_plt = ex_plt*1e3
    ey_plt = ey_plt*1e3

    fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.plot(current_scan,ex_plt,'-', c='r')
    ax1.set_xlabel('current [mA]')
    ax1.set_ylabel('epsn_x [mm]')
    ax1.set_title(f'dependence of epsn_x on current')
    ax1.grid()

    ax3.plot(current_scan,ey_plt,'-', c='r')
    ax3.set_xlabel('current [mA]')
    ax3.set_ylabel('epsn_y [mm]')
    ax3.set_title(f'dependence of epsn_y on current')
    ax3.grid()
    ax1.scatter(current_scan, ex_plt, color='r', s=20, marker='s')
    ax3.scatter(current_scan, ey_plt, color='r', s=20, marker='s')

    plt.tight_layout()
    plt.show()
    if path != None:
        fig.savefig(path+name)
                                    

    

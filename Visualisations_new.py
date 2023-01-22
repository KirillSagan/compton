from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


MEDIUM_SIZE = 20
plt.rc('font', size = MEDIUM_SIZE)          
plt.rc('axes', titlesize = MEDIUM_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)    
plt.rc('xtick', labelsize=MEDIUM_SIZE)   
plt.rc('ytick', labelsize=MEDIUM_SIZE)    
plt.rc('legend', fontsize=MEDIUM_SIZE)               
    #plt.rc('figure', titlesize = MEDIUM_SIZE)  
plt.rc('lines', linewidth = 2) 

def plot_transerse_phase_space(ax1, ax2, bunch,current, path=None,name='transerse_phase_space', fig = None):
    #fig, (ax1, ax2) = plt.subplots(2,1,figsize=(7,12))
    ax1.cla()
    ax2.cla()
    ax1.scatter(bunch.x, bunch.xp, marker='o', s = 0.05, c = 'b')
    ax2.scatter(bunch.y, bunch.yp, marker='o', s = 0.05, c= 'r')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('xp')
    ax1.set_title(name.strip('.jpg'))
    
    ax2.set_title(name.strip('.jpg'))
    ax2.set_xlabel('y [m]')
    ax2.set_ylabel('yp')
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    plt.tight_layout()
    if fig != None:
        fig.savefig(path+name+f' current = {current:.2e} mA'.replace('.',',')+'.jpg')
    
def plot_longitudinal_phase_space_color(bunch, charge, path=None,
                                        name='longitudinal_phase_space', savefig = None):

    fig, ax1 = density_scatter(x = bunch.z*1e3, y= bunch.dp, bins = 600, 
                        s = 0.07,  cmap=cm.viridis, n_sigma = 3.5)
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.set_title(name.strip('.jpg'))
    plt.tight_layout()                                          
    plt.show()
    if savefig != None:
        fig.savefig(path+name+f'_charge_=_{charge*1e9:.3e}nC'.replace('.',',')+'.jpg')

def plot_longitudinal_phase_space(ax1, bunch, charge, path=None,
                                        name='longitudinal_phase_space', fig = None):
    #fig, (ax1) = plt.subplots(1)
    ax1.cla()
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.scatter(bunch.z, bunch.dp, marker='o', s =0.05, c = 'b')
    ax1.set_xlabel('z [m]')
    ax1.set_ylabel('dp')
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.set_title(name.strip('.jpg'))
    plt.tight_layout()                                          
    plt.show()
    if fig != None:
        fig.savefig(path+name+f' charge = {charge*1e9:.3e} nC'.replace('.',',')+'.jpg')
        

def plot_sigma_z_sigma_E_mean_z_mean_E(sigma_z_scan, sigma_E_scan,
					            mean_z_scan, mean_E_scan, n_turns, write_every,
                         		charge, path=None, name='sigma_z_sigma_E', savefig = None):
    fig, (axt,axb) = plt.subplots(2,2,figsize = (20,10))
    ax1, ax3 = axt
    ax2, ax4 = axb

    x = np.arange(0, n_turns+1, write_every)

    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.plot(x,sigma_z_scan*1e3,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('sigma z [mm]')
    ax1.set_title(f'dependence of sigma z on turns')
    ax1.grid(True)
    
    ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax3.plot(x,sigma_E_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('sigma dE/E []')
    ax3.set_title(f'dependence of sigma E on turns')
    ax3.grid(True)

    ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax2.plot(x,mean_z_scan*1e3,'-', c='r')
    ax2.set_xlabel('turns')
    ax2.set_ylabel('mean z [mm]')
    ax2.set_title(f'dependence of mean z on turns')
    ax2.grid(True)
    
    ax4.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax4.plot(x,mean_E_scan,'-', c='r')
    ax4.set_xlabel('turns')
    ax4.set_ylabel('mean dE/E []')
    ax4.set_title(f'dependence of mean E on turns')
    ax4.grid(True)
    
    ax1.set_xlim(x.min(),x.max()+1)
    ax2.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    ax4.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    #plt.show()
    if savefig != None:
        fig.savefig(path+name+f' charge = {charge*1e9:.2e} nC'.replace('.',',')+'.jpg')
        
        
def plot_mean_z_sigma_z(ax1, ax3, mean_z_scan,sigma_z_scan,n_turns,write_every,
                         charge,path=None,name='mean_z_sigma_z', fig = None):
    ax3.cla()
    x = np.arange(0, n_turns+1, write_every)
    #fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax3.plot(x,sigma_z_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('sigma_z [mm]')
    ax3.set_title(f'dependence of sigma z on turns')
    #ax3.grid(True)
    
    ax1.cla()
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.plot(x,mean_z_scan,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('mean_z [mm]')
    ax1.set_title(f'dependence of mean z on turns')
    #ax1.grid(True)
    

    ax1.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    #plt.show()
    if fig != None:
        fig.savefig(path+name+f' charge = {charge:.2e} nC'.replace('.',',')+'.jpg')


def plot_ex_ey(ax1, ax3, ex_scan,ey_scan,n_turns,write_every,
                         current,path=None,name='ex_ey', fig = None):
    x = np.arange(0, n_turns+1, write_every)
    #fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.cla()
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.plot(x,ex_scan,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('epsn_x [m rad]')
    ax1.set_title(f'dependence of epsn_x on turns')
    #ax1.grid(True)
    
    ax3.cla()
    ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax3.plot(x,ey_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('epsn_y [m rad]')
    ax3.set_title(f'dependence of epsn_y on turns')
    #ax3.grid(True)
    
    #ax1.legend(title = f'current {current:.2e} mA', facecolor = 'w')
    #ax3.legend(title = f'current {current:.2e} mA', facecolor = 'w')

    ax1.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    plt.show()
    if fig != None:
        fig.savefig(path+name+f' current = {current:.2e} mA'.replace('.',',')+'.jpg')
    
def plot_mx_my(ax1, ax3, mx_scan,my_scan,n_turns,write_every,
                         current,path=None,name='mx_my', fig = None):
    x = np.arange(0, n_turns+1, write_every)
    #fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.cla()
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.plot(x,mx_scan,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('mean_x [mm]')
    ax1.set_title(f'dependence of mean_x on turns')
    #ax1.grid(True)
    
    ax3.cla()
    ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax3.plot(x,my_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('mean_y [mm]')
    ax3.set_title(f'dependence of mean_y on turns')
    #ax3.grid(True)
    
    #ax1.legend(title = f'current {current:.2e} mA', facecolor = 'w')
    #ax3.legend(title = f'current {current:.2e} mA', facecolor = 'w')

    ax1.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    plt.show()
    if fig != None:
        fig.savefig(path+name+f' current = {current:.2e} mA'.replace('.',',')+'.jpg')
        
def plot_sigma_xp_sigma_yp(ax1, ax3, mxp_scan,myp_scan,n_turns,write_every,
                         current,path=None,name='mx_my', fig = None):
    x = np.arange(0, n_turns+1, write_every)
    #fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.cla()
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.plot(x,mxp_scan,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('sigma_xp [1]')
    ax1.set_title(f'dependence of sigma_xp on turns')
    #ax1.grid(True)
    
    ax3.cla()
    ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax3.plot(x,myp_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('sigma_yp [1]')
    ax3.set_title(f'dependence of sigma_yp on turns')
    #ax3.grid(True)
    
    #ax1.legend(title = f'current {current:.2e} mA', facecolor = 'w')
    #ax3.legend(title = f'current {current:.2e} mA', facecolor = 'w')

    ax1.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    plt.show()
    if fig != None:
        fig.savefig(path+name+f' current = {current:.2e} mA'.replace('.',',')+'.jpg')
        
        
def plot_sigma_x_sigma_y(ax1, ax3, sigma_x_scan,sigma_y_scan,n_turns,write_every,
                         current,path=None,name='mx_my', fig = None):
    x = np.arange(0, n_turns+1, write_every)
    #fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.cla()
    ax1.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax1.plot(x,sigma_x_scan,'-', c='r')
    ax1.set_xlabel('turns')
    ax1.set_ylabel('sigma_x [mm]')
    ax1.set_title(f'dependence of sigma_x on turns')
    #ax1.grid(True)
    
    ax3.cla()
    ax3.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax3.plot(x,sigma_y_scan,'-', c='r')
    ax3.set_xlabel('turns')
    ax3.set_ylabel('sigma_y [mm]')
    ax3.set_title(f'dependence of sigma_y on turns')
    #ax3.grid(True)
    
    #ax1.legend(title = f'current {current:.2e} mA', facecolor = 'w')
    #ax3.legend(title = f'current {current:.2e} mA', facecolor = 'w')

    ax1.set_xlim(x.min(),x.max()+1)
    ax3.set_xlim(x.min(),x.max()+1)
    plt.tight_layout()
    plt.show()
    if fig != None:
        fig.savefig(path+name+f' current = {current:.2e} mA'.replace('.',',')+'.jpg')

def plot_sigma_z_sigma_E_charge(sigma_z_plt,sigma_E_plt,charge_scan, 
                                 path=None,name='sigma_z_sigma_E_headtail.jpg'):

    fig, (ax1, ax3) = plt.subplots(2,1,figsize = (10,6))
    ax1.plot(charge_scan*1e9,sigma_z_plt*1e3,'-', c='r')
    ax1.set_xlabel('charge [nC]')
    ax1.set_ylabel('sigma z [mm]')
    ax1.set_title(f'dependence of sigma_z on charge')
    ax1.grid(True)

    ax3.plot(charge_scan*1e9,sigma_E_plt,'-', c='r')
    ax3.set_xlabel('charge [nC]')
    ax3.set_ylabel('sigma E []')
    ax3.set_title(f'dependence of sigma_E on charge')
    ax3.grid(True)
    ax1.scatter(charge_scan*1e9, sigma_z_plt*1e3, color='r', s=25, marker='s')
    ax3.scatter(charge_scan*1e9, sigma_E_plt, color='r', s=25, marker='s')

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
    #ax1.grid()

    ax3.plot(current_scan,my_plt,'-', c='r')
    ax3.set_xlabel('current [mA]')
    ax3.set_ylabel('mean_y [mm]')
    ax3.set_title(f'dependence of mean_y on current')
    #ax3.grid()
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
    #ax1.grid()

    ax3.plot(current_scan,ey_plt,'-', c='r')
    ax3.set_xlabel('current [mA]')
    ax3.set_ylabel('epsn_y [mm]')
    ax3.set_title(f'dependence of epsn_y on current')
    #ax3.grid()
    ax1.scatter(current_scan, ex_plt, color='r', s=20, marker='s')
    ax3.scatter(current_scan, ey_plt, color='r', s=20, marker='s')

    plt.tight_layout()
    plt.show()
    if path != None:
        fig.savefig(path+name)
                                    
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde
import matplotlib.gridspec as gridspec

def density_scatter( x, y, ax = None, sort = True, bins = 20, cmap=cm.plasma,
                    xlabel = 'z [mm]',ylabel = 'dp []', n_sigma = 7, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig = plt.figure(figsize=(10,9))
    gs = gridspec.GridSpec(3, 3)
    ax = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2],sharex=ax)
    
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) 
                , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, cmap=cmap, **kwargs )
    n_sigma_x = n_sigma
    n_sigma_y = n_sigma
    ax.set_xlim(-np.std(x)*n_sigma_x+np.mean(x), np.std(x)*n_sigma_x+np.mean(x))
    ax.set_ylim(-np.std(y)*n_sigma_y+np.mean(y), np.std(y)*n_sigma_y+np.mean(y))
    plt.tight_layout()
    ax_xDist.hist(x,bins=bins,align='mid')
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    return fig, ax
    

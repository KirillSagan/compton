"""
@class WakeKick
@author Kevin Li, Michael Schenk
@date July 2014
@brief Implementation of the wake kicks, i.e. of the elementary objects
       describing the effects of a wake field.
@copyright CERN
"""

import matplotlib.pyplot as plt

import numpy as np
from scipy.constants import c
from abc import ABCMeta, abstractmethod

from PyHEADTAIL.general import pmath as pm
from PyHEADTAIL.general.element import Printing
from scipy.interpolate import interp1d

class ImpedanceKick(Printing, metaclass=ABCMeta):
    """Abstract base class for wake kick classes, like e.g. the
    DipoleWakeKickX.
    Provides the basic and universal methods to calculate the strength
    of a wake kick. Two implementations of the convolution are
    available. Based on what slicer mode (uniform_bin, uniform_charge)
    is used, the self._convolution method is bound to one or the other.
    The self.apply(bunch, slice_set) method calculates and applies the
    corresponding kick to the particles of the bunch that are located
    inside the slicing region defined by a slice_set. This should be
    the only method to be implemented for a child class inheriting from
    the WakeKick class.
    """

    def __init__(self, wake_function, slicer, n_turns_wake,
                 *args, **kwargs):
        """Universal constructor for WakeKick objects. The slicer_mode
        is passed only to decide about which of the two implementations
        of the convolution the self._convolution method is bound to.
        """
        self.wake_function = wake_function

        if (slicer.mode == 'uniform_bin' and
                (n_turns_wake == 1 or slicer.z_cuts)):
            self._convolution = self._convolution_numpy
        else:
            self._convolution = self._convolution_dot_product
            if n_turns_wake > 1:
                self.warns(
                    'You use multiturn wakes. The stored '
                    'SliceSet instances from previous turns are converted '
                    'from z to dt according to the current turn\'s gamma. '
                    'If you accelerate this is only an approximation which '
                    'becomes invalid when gamma changes much over '
                    'n_turns_wake.')

        self.n_turns_wake = n_turns_wake

    @abstractmethod
    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply the corresponding wake kick to the
        bunch conjugate momenta using the given slice_set. Only
        particles within the slicing region, i.e particles_within_cuts
        (defined by the slice_set) experience a kick.
        """
        pass

    @staticmethod
    def _wake_factor(bunch):
        """Universal scaling factor for the strength of a wake field
        kick.
        """
        wake_factor = (-(bunch.charge)**2 / (bunch.mass * bunch.gamma *
                       (bunch.beta * c)**2) * bunch.particlenumber_per_mp)
        return wake_factor

    def _convolution_dot_product(self, target_times, source_times,
                                 source_moments, source_beta):
        """Implementation of the convolution of wake and source_moments
        (beam profile) using the numpy dot product. To be used with the
        'uniform_charge' slicer mode.
        """
        dt_to_target_slice = (
            [target_times] - np.transpose([source_times]))
        wake = self.wake_function(dt_to_target_slice, beta=source_beta)

        return np.dot(source_moments, wake)

    def _convolution_numpy(self, target_times, source_times,
                           source_moments, source_beta):
        """Implementation of the convolution of wake and source_moments
        (longitudinal beam profile) using the numpy built-in
        numpy.convolve method. Recommended use with the 'uniform_bin'
        slicer mode (in case of multiturn wakes, additional conditions
        must be fulfilled: fixed z_cuts and no acceleration!) for
        higher performance. Question: how about interpolation to avoid
        expensive dot product in most cases?
        """
        # Currently target_times/source_times are on the GPU --> np.concatenate
        # doesnt work. Temporary fix before checking if rewrite of
        # np.concatenate is required on GPU (if this is bottleneck), is to
        # get the arrays to the cpu via .get()
        try:
            target_times = target_times.get()
        except AttributeError:
            pass # is already on CPU
        try:
            source_times = source_times.get()
        except AttributeError:
            pass #is already on CPU
        dt_to_target_slice = np.concatenate(
            (target_times - source_times[-1],
            (target_times - source_times[0])[1:]))
        wake = self.wake_function(dt_to_target_slice, beta=source_beta)
        #print 'len convolution', len(source_moments), len(wake)
        #print 'type moments', type(source_moments[0])
        #print 'type wake', type(wake[0]), wake
        return pm.convolve(source_moments, wake, 'valid')

    def _accumulate_source_signal(self, bunch, times_list, ages_list,
                                  moments_list, betas_list):
        """Accumulate (multiturn-)wake signals left by source slices.
        Takes a list of slice set attributes and adds up all
        convolutions weighted by the respective moments. Also updates
        the age of each slice set.
        """
        target_times = times_list[0]
        accumulated_signal = 0

        if len(ages_list) < self.n_turns_wake:
            n_turns = len(ages_list)
        else:
            n_turns = self.n_turns_wake

        for i in range(n_turns):
            source_times = times_list[i] + ages_list[i]
            source_beta = betas_list[i]
            source_moments = moments_list[i]
            accumulated_signal += self._convolution(
                target_times, source_times, source_moments, source_beta)
        return self._wake_factor(bunch) * accumulated_signal


""" Constant wake kicks """

class ConstantImpedanceKickX(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a constant wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = lambda_z_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0
        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)

        constant_kick = self._wake_factor(bunch)*new_wake_potential

        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.xp[p_idx] += pm.take(constant_kick, s_idx)


class ConstantImpedanceKickY(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a constant wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = lambda_z_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0
        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)

        constant_kick = self._wake_factor(bunch)*new_wake_potential

        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.yp[p_idx] += pm.take(constant_kick, s_idx)


class ConstantImpedanceKickZ(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a constant wake kick to bunch.dp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = lambda_z_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0
        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)

        constant_kick = self._wake_factor(bunch)*new_wake_potential

        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.dp[p_idx] += pm.take(constant_kick, s_idx)


""" Dipolar wake kicks """

class DipoleImpedanceKickX(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a dipolar wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)

        moments_list = [s.n_macroparticles_per_slice*s.mean_x
                        for s in slice_set_list]

        moments_fft = np.fft.rfft(moments_list[0]/slice_set_list[0].slice_widths)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = moments_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0

        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)
        plt.plot(times/c*1e3, new_wake_potential)
        
        dipole_kick_x = self._wake_factor(bunch)*new_wake_potential
        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.xp[p_idx] += pm.take(dipole_kick_x, s_idx)

class DipoleImpedanceKickXY(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a dipolar (cross term x-y) wake kick
        to bunch.xp using the given slice_set. Only particles within
        the slicing region, i.e particles_within_cuts (defined by the
        slice_set) experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)

        moments_list = [s.n_macroparticles_per_slice*s.mean_y
                        for s in slice_set_list]

        moments_fft = np.fft.rfft(moments_list[0]/slice_set_list[0].slice_widths)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = moments_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0

        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)
        plt.plot(times/c*1e3, new_wake_potential)    

        dipole_kick_xy = self._wake_factor(bunch)*new_wake_potential
        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.xp[p_idx] += pm.take(dipole_kick_xy, s_idx)


class DipoleImpedanceKickY(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a dipolar wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)

        moments_list = [s.n_macroparticles_per_slice*s.mean_y
                        for s in slice_set_list]

        moments_fft = np.fft.rfft(moments_list[0]/slice_set_list[0].slice_widths)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = moments_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0

        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)
        plt.plot(times/c*1e3, new_wake_potential)

        dipole_kick_y = self._wake_factor(bunch)*new_wake_potential
        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.yp[p_idx] += pm.take(dipole_kick_y, s_idx)

class DipoleImpedanceKickYX(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a dipolar (cross term y-x) wake kick
        to bunch.yp using the given slice_set. Only particles within
        the slicing region, i.e particles_within_cuts (defined by the
        slice_set) experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)

        moments_list = [s.n_macroparticles_per_slice*s.mean_x
                        for s in slice_set_list]

        moments_fft = np.fft.rfft(moments_list[0]/slice_set_list[0].slice_widths)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = moments_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0

        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)
        plt.plot(times/c*1e3, new_wake_potential)

        dipole_kick_yx = self._wake_factor(bunch)*new_wake_potential

        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.yp[p_idx] += pm.take(dipole_kick_yx, s_idx)


""" Quadrupolar wake kicks """

class QuadrupoleImpedanceKickX(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a quadrupolar wake kick to bunch.xp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = lambda_z_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0

        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)
        plt.plot(times/c*1e3, new_wake_potential)

        quadrupole_kick_x = self._wake_factor(bunch)*new_wake_potential

        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.xp[p_idx] += pm.take(quadrupole_kick_x, s_idx) * bunch.x[p_idx]


class QuadrupoleImpedanceKickXY(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a quadrupolar (cross term x-y) wake
        kick to bunch.xp using the given slice_set. Only particles
        within the slicing region, i.e particles_within_cuts (defined by
        the slice_set) experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = lambda_z_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0
        
        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)
        plt.plot(times/c*1e3, new_wake_potential)

        quadrupole_kick_xy = self._wake_factor(bunch)*new_wake_potential

        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.xp[p_idx] += pm.take(quadrupole_kick_xy, s_idx) * bunch.y[p_idx]


class QuadrupoleImpedanceKickY(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a quadrupolar wake kick to bunch.yp
        using the given slice_set. Only particles within the slicing
        region, i.e particles_within_cuts (defined by the slice_set)
        experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = lambda_z_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0
        
        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)
        plt.plot(times/c*1e3, new_wake_potential)

        quadrupole_kick_y = self._wake_factor(bunch)*new_wake_potential

        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = slice_set_list[0].slice_index_of_particle[p_idx]
        bunch.yp[p_idx] += pm.take(quadrupole_kick_y, s_idx) * bunch.y[p_idx]


class QuadrupoleImpedanceKickYX(ImpedanceKick):

    def apply(self, bunch, times, lambda_z_fft, lambda_z_wake_fft, slice_set_list):
        """Calculate and apply a quadrupolar (cross term y-x) wake
        kick to bunch.yp using the given slice_set. Only particles
        within the slicing region, i.e particles_within_cuts (defined by
        the slice_set) experience the kick.
        """
        wake_potential_interp = self.wake_function(times)
        impedance = np.fft.rfft(wake_potential_interp)
        with np.errstate(invalid='ignore'):
            lambda_fft_ratio = lambda_z_fft/lambda_z_wake_fft
            lambda_fft_ratio[np.isnan(lambda_fft_ratio)] = 0
        
        new_impedance = impedance * lambda_fft_ratio
        new_wake_potential = np.fft.irfft(new_impedance)
        plt.plot(times/c*1e3, new_wake_potential)

        quadrupole_kick_yx = self._wake_factor(bunch)*new_wake_potential

        p_idx = slice_set_list[0].particles_within_cuts_slice
        s_idx = pm.take(slice_set_list[0].slice_index_of_particle, p_idx)
        bunch.yp[p_idx] += pm.take(quadrupole_kick_yx, s_idx) * bunch.x[p_idx]

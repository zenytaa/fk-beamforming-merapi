import os

from obspy.signal.tf_misfit import cwt
from obspy.imaging.cm import obspy_sequential
from obspy.signal.trigger import trigger_onset
from obspy.signal import util

import datetime as dt

import matplotlib.pyplot as plt
from matplotlib.pyplot import rc

from matplotlib.colors import Normalize
from matplotlib.dates import DateFormatter, MinuteLocator, SecondLocator

import numpy as np
import math


def plot_trigger(trace, cft, thr_on, thr_off, show=True):
    import matplotlib.pyplot as plt
    df = trace.stats.sampling_rate
    npts = trace.stats.npts
    tt = (trace.stats.starttime).timestamp + trace.times()
    t=[]
    for i in range(len(tt)):
      t.append(dt.datetime.utcfromtimestamp(tt[i]))
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(t, trace.data, 'k')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(t, cft, 'k')
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))
    i, j = ax1.get_ylim()
    trigger_on = dt.datetime.utcfromtimestamp(int(on_off[:, 0]/df) + 
                                              trace.stats.starttime.timestamp)
    trigger_off = dt.datetime.utcfromtimestamp(int(on_off[:, 1]/df) +
                                               trace.stats.starttime.timestamp)
    try:
        ax1.vlines(trigger_on,
                   i,
                   j,
                   color='r',
                   lw=2,
                   label="Trigger On")
        ax1.vlines(trigger_off,
                   i,
                   j,
                   color='b',
                   lw=2,
                   label="Trigger Off")
        ax1.legend()
    except IndexError:
        pass
    ax2.axhline(thr_on, color='red', lw=1, ls='--')
    ax2.axhline(thr_off, color='blue', lw=1, ls='--')
    ax2.set_xlabel("Trigger on : %s [s]" % trigger_on,
                   fontsize=15)
    ax2.set_ylabel("Threshold", fontsize=15)
    ax1.set_ylabel("Velocity (m/s)", fontsize=15)
    fig.suptitle(trace.id, fontsize=15)
    fig.canvas.draw()
    if show:
        plt.show()


#get trigger on and off times
def get_trig_times(cft, trace, thr_on, thr_off):
  trig = trigger_onset(cft, thr_on, thr_off)
  trig_on = dt.datetime.utcfromtimestamp(int(trig[:, 0]/trace.stats.sampling_rate) + 
                                         trace.stats.starttime.timestamp)
  trig_off = dt.datetime.utcfromtimestamp(int(trig[:, 1]/trace.stats.sampling_rate) + 
                                          trace.stats.starttime.timestamp)
  return [trig_on, trig_off]


def get_array_coords(st, ref_station):
    '''
    Returns the array coordinates for an array, in km with respect to the reference array provided
    
    Inputs:
    st - ObsPy Stream object containing array data
    ref_station - A String containing the name of the reference station
    
    Outputs:
    X - [Nx2] NumPy array of array coordinates in km
    stnm - [Nx1] list of element names
    
    Stephen Arrowsmith (sarrowsmith@smu.edu)
    '''
    
    X = np.zeros((len(st), 2))
    stnm = []
    for i in range(0, len(st)):
        E, N = (st[i].stats.coordinates.latitude, st[i].stats.coordinates.longitude)
        X[i,0] = E; X[i,1] = N
        stnm.append(st[i].stats.station)

    # Adjusting to the reference station, and converting to km:
    ref_station_ix = np.where(np.array(stnm) == ref_station)[0][0]    # index of reference station
    X[:,0] = (X[:,0] - X[ref_station_ix,0])
    X[:,1] = (X[:,1] - X[ref_station_ix,1])
    X = X/1000
    
    return X, stnm

def plot_array_coords(X, stnm, units='km'):
    '''
    Plots the array coordinates for a given array with coordinates X and element names stnm
    '''
    
    if units == 'm':
        X = X*1000
    plt.plot(X[:,1], X[:,0], '.')
    plt.grid()
    for i in range(0, len(stnm)):
        plt.text(X[i,1], X[i,0], stnm[i])
    if units == 'km':
        plt.xlabel('X (km)')
        plt.ylabel('Y (km)')
    elif units == 'm':
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
    else:
        print('Unrecognized units (Options are "km" and "m")')


# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: spectrogram.py
#  Purpose: Plotting spectrogram of Seismograms.
#   Author: Christian Sippl, Moritz Beyreuther
#    Email: sippl@geophysik.uni-muenchen.de
#
# Copyright (C) 2008-2012 Christian Sippl
# --------------------------------------------------------------------
"""
Plotting spectrogram of seismograms.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import os
import math

import numpy as np
from matplotlib import mlab
from matplotlib.colors import Normalize

from obspy.imaging.cm import obspy_sequential


def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def spectrogram(data, samp_rate, per_lap=0.9, wlen=None, log=False,
                outfile=None, fmt=None, axes=None, dbscale=False,
                mult=8.0, cmap=obspy_sequential, zorder=None, title=None,
                show=True, clip=[0.0, 1.0]):
    """
    Computes and plots spectrogram of the input data.

    :param data: Input data
    :type samp_rate: float
    :param samp_rate: Samplerate in Hz
    :type per_lap: float
    :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
    :type wlen: int or float
    :param wlen: Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to
        (samp_rate/100.0).
    :type log: bool
    :param log: Logarithmic frequency axis if True, linear frequency axis
        otherwise.
    :type outfile: str
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :type fmt: str
    :param fmt: Format of image to save
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Plot into given axes, this deactivates the fmt and
        outfile option.
    :type dbscale: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
        sqrt is taken.
    :type mult: float
    :param mult: Pad zeros to length mult * wlen. This will make the
        spectrogram smoother.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then the
        default ObsPy sequential colormap is used.
    :type zorder: float
    :param zorder: Specify the zorder of the plot. Only of importance if other
        plots in the same axes are executed.
    :type title: str
    :param title: Set the plot title
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
    :type clip: [float, float]
    :param clip: adjust colormap to clip at lower and/or upper end. The given
        percentages of the amplitude range (linear or logarithmic depending
        on option `dbscale`) are clipped.
    """
    import matplotlib.pyplot as plt
    # enforce float for samp_rate
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = samp_rate / 100.

    npts = len(data)
    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]

    vmin, vmax = clip
    if vmin < 0 or vmax > 1 or vmin >= vmax:
        msg = "Invalid parameters for clip option."
        raise ValueError(msg)
    _range = float(specgram.max() - specgram.min())
    vmin = specgram.min() + vmin * _range
    vmax = specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)

    if not axes:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
    else:
        ax = axes

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = {k: v for k, v in (('cmap', cmap), ('zorder', zorder))
              if v is not None}

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        # Log scaling for frequency values (y-axis)
        ax.set_yscale('log')
        # Plot times
        ax.pcolormesh(time, freq, specgram, norm=norm, **kwargs)
    else:
        # this method is much much faster!
        specgram = np.flipud(specgram)
        # center bin
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        ax.imshow(specgram, interpolation="nearest", extent=extent, **kwargs)

    # set correct way of axis, whitespace before and after with window
    # length
    ax.axis('tight')
    ax.set_xlim(0, end)
    ax.grid(False)

    if axes:
        return ax

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [Hz]')
    if title:
        ax.set_title(title)

    if not os.environ.get('SPHINXBUILD'):
        # ignoring all NumPy warnings during plot
        with np.errstate(all='ignore'):
            plt.draw()
    if outfile:
        if fmt:
            fig.savefig(outfile, format=fmt)
        else:
            fig.savefig(outfile)
    elif show:
        plt.show()
    else:
        return fig


def _pcolormesh_same_dim(ax, x, y, v, **kwargs):
    # x, y, v must have the same dimension
    try:
        return ax.pcolormesh(x, y, v, shading='nearest', **kwargs)
    except TypeError:
        # matplotlib versions < 3.3
        return ax.pcolormesh(x, y, v[:-1, :-1], **kwargs)

   
def plot_tfr(st, tr, dt=0.01, t0=0., fmin=1., fmax=10., nf=100, w0=6, left=0.1,
             bottom=0.1, h_1=0.2, h_2=0.6, w_1=0.2, w_2=0.6, w_cb=0.01,
             d_cb=0.0, show=True, plot_args=['k', 'k'], clim=0.0,
             cmap=obspy_sequential, mode='absolute', fft_zero_pad_fac=0):
    """
    Plot time frequency representation, spectrum and time series of the signal.

    :param st: signal, type numpy.ndarray with shape (number of components,
        number of time samples) or (number of timesamples, ) for single
        component data
    :param dt: time step between two samples in st
    :param t0: starting time for plotting
    :param fmin: minimal frequency to be analyzed
    :param fmax: maximal frequency to be analyzed
    :param nf: number of frequencies (will be chosen with logarithmic spacing)
    :param w0: parameter for the wavelet, tradeoff between time and frequency
        resolution
    :param left: plot distance from the left of the figure
    :param bottom: plot distance from the bottom of the figure
    :param h_1: height of the signal axis
    :param h_2: height of the TFR/spectrum axis
    :param w_1: width of the spectrum axis
    :param w_2: width of the TFR/signal axes
    :param w_cb: width of the colorbar axes
    :param d_cb: distance of the colorbar axes to the other axes
    :param show: show figure or return
    :param plot_args: list of plot arguments passed to the signal and spectrum
        plots
    :param clim: limits of the colorbars
    :param cmap: colormap for TFEM/TFPM, either a string or
        matplotlib.cm.Colormap instance
    :param mode: 'absolute' for absolute value of TFR, 'power' for ``|TFR|^2``
    :param fft_zero_pad_fac: integer, if > 0, the signal is zero padded to
        ``nfft = next_pow_2(len(st)) * fft_zero_pad_fac`` to get smoother
        spectrum in the low frequencies (has no effect on the TFR and might
        make demeaning/tapering necessary to avoid artifacts)

    :return: If show is False, returns a matplotlib.pyplot.figure object
        (single component data) or a list of figure objects (multi component
        data)

    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.ticker import NullFormatter 
    from matplotlib import rc
    start = tr.stats.starttime
    npts = st.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts) + t0

    if fft_zero_pad_fac == 0:
        nfft = npts
    else:
        nfft = util.next_pow_2(npts) * fft_zero_pad_fac

    f_lin = np.linspace(0, 0.5 / dt, nfft // 2 + 1)

    if len(st.shape) == 1:
        _w = np.zeros((1, nf, npts), dtype=complex)
        _w[0] = cwt(st, dt, w0, fmin, fmax, nf)
        ntr = 1

        spec = np.zeros((1, nfft // 2 + 1), dtype=complex)
        spec[0] = np.fft.rfft(st, n=nfft) * dt

        st = st.reshape((1, npts))
    else:
        _w = np.zeros((st.shape[0], nf, npts), dtype=complex)
        spec = np.zeros((st.shape[0], nfft // 2 + 1), dtype=complex)

        for i in np.arange(st.shape[0]):
            _w[i] = cwt(st[i], dt, w0, fmin, fmax, nf)
            spec[i] = np.fft.rfft(st[i], n=nfft) * dt

        ntr = st.shape[0]

    if mode == 'absolute':
        _tfr = np.abs(_w)
        spec = np.abs(spec)
    elif mode == 'power':
        _tfr = np.abs(_w) ** 2
        spec = np.abs(spec) ** 2
    else:
        raise ValueError('mode "' + mode + '" not defined!')

    figs = []

    for itr in np.arange(ntr):
        fig = plt.figure()

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom, w_2, h_1])
        ax_sig.plot(t, st[itr], plot_args[0])

        # plot TFR
        ax_tfr = fig.add_axes([left + w_1, bottom + h_1, w_2, h_2])

        x, y = np.meshgrid(
            t, np.logspace(np.log10(fmin), np.log10(fmax),
                           _tfr[itr].shape[0]))
        img_tfr = _pcolormesh_same_dim(ax_tfr, x, y, _tfr[itr], cmap=cmap)
        img_tfr.set_rasterized(True)
        ax_tfr.set_yscale("log")
        ax_tfr.set_ylim(fmin, fmax)
        ax_tfr.set_xlim(t[0], t[-1])

        # plot spectrum
        ax_spec = fig.add_axes([left, bottom + h_1, w_1, h_2])
        ax_spec.semilogy(spec[itr], f_lin, plot_args[1])

        # add colorbars
        ax_cb_tfr = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom +
                                  h_1, w_cb, h_2])
        fig.colorbar(img_tfr, cax=ax_cb_tfr)

        # set limits
        ax_sig.set_ylim(st.min() * 1.1, st.max() * 1.1)
        ax_sig.set_xlim(t[0], t[-1])

        xlim = spec.max() * 1.1

        ax_spec.set_xlim(xlim, 0.)
        ax_spec.set_ylim(fmin, fmax)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True) 
        formatter.set_powerlimits((-1,1)) 
        ax_spec.xaxis.set_major_formatter(formatter)

        if clim == 0.:
            clim = _tfr.max()

        img_tfr.set_clim(0., clim)

        ax_sig.set_xlabel('time (s) after %s' %(start),)
        ax_spec.set_ylabel('frequency (Hz)')
        ax_tfr.set_title('Station : %s' %(start))
        ax_spec.set_title('Channel : %s' %(tr.stats.channel))
        ax_sig.set_ylabel('velocity (m/s)')

        # set grid
        ax_sig.grid()
        ax_spec.grid()

        # remove axis labels
        ax_tfr.xaxis.set_major_formatter(NullFormatter())
        ax_tfr.yaxis.set_major_formatter(NullFormatter())

        figs.append(fig)

    if show:
        plt.savefig('Signal Spectrum - %s' %(start.strftime('%Y-%m-%d-%H-%M-%S.png')), facecolor='white')
        plt.show()
    else:
        if ntr == 1:
            return figs[0]
        else:
            return figs
    

if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)


def plot_traces(sts,labels,title):

    """
    This function plot streams of seismic signals.
    Several streams can be passed to the function for comparison.

    Input parameters
    sts:     List of N streams (Obspy Stream)
    labels:  List of N labels
    title:   Title of figure
    
    """

    rc('font', size=12.0)
    
    fig, ax = plt.subplots(sts[0].count(), 1, sharex=True, sharey=True,
            figsize=(10,12))
    ax[0].set_title(title)
   
    # Plot each trace of each stream
    for st, label in zip(sts, labels):
        for itr, tr in enumerate(st):
            ilabel = tr.stats.station+'.'+tr.stats.channel[2]+' - '+label
            ax[itr].grid('on',which='both')
            ax[itr].plot(tr.times('matplotlib'), tr.data,label=ilabel)
            ax[itr].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if itr is not 0:
                ax[itr].yaxis.get_offset_text().set_visible(False)
            ax[itr].set(ylabel='$v_Z$ (m/s)')
            ax[itr].legend(loc=1)

    ax[itr].set(xlabel='Time (s)')
    # Bring subplots close to each other.
    fig.subplots_adjust(hspace=0.1)
    # Hide x labels and tick labels for all but bottom plot.
    for axi in ax:
        axi.label_outer()
    # Format time axis
    if st[0].stats.npts*st[0].stats.delta > 80 :
        plt.gca().xaxis.set_minor_locator( 
                SecondLocator(bysecond=range(10,60,10)) )
    else:
        plt.gca().xaxis.set_minor_locator( 
                SecondLocator(bysecond=range( 5,60, 5)) )
    plt.gca().xaxis.set_minor_formatter( DateFormatter("%S''") )
    plt.gca().xaxis.set_major_locator( MinuteLocator(byminute=range( 0,60, 1)) )
    plt.gca().xaxis.set_major_formatter( DateFormatter('%H:%M:%S') )
    
    plt.show()
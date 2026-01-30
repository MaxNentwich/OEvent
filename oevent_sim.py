#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:57:42 2026

@author: max
"""
import sys, copy
import numpy as np
from neurodsp.sim.combined import sim_peak_oscillation
from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim import sim_oscillation

import matplotlib.pyplot as plt

sys.path.append('/media/max/Workspace/Code/python_funcs/OEvent-master')
import oevent

#%% Simulation
n_cyc = 10

sig_cyc = np.empty(n_cyc, dtype=object)

for c in range(n_cyc):
    
    n_sec_osc = 5
    n_sec_ap = 5
    fs = 512
 
    sig_ap = sim_powerlaw(n_sec_ap, fs, exponent=-2.3, f_range=[2, None])

    sig_ap_osc = sim_powerlaw(n_sec_osc, fs, exponent=-2.3, f_range=[2, None])
    sig_osc = sim_oscillation(n_sec_osc, fs, 10)
    
    sig_osc += sig_ap_osc
    
    sig_cyc[c] = np.hstack((sig_ap, sig_osc))
    
sig_sim = np.hstack(sig_cyc)

time = np.arange(len(sig_sim)) / fs

#%% OEvent

# parameters for OEvent
medthresh = 4.0 # median threshold

winsz = len(sig_sim) / fs # 10 second window size
freqmin = 1. # minimum frequency (Hz)
freqmax = 250.  # maximum frequency (Hz)
freqstep = 2 # frequency step (Hz)
overlapth = 0.1 # overlapping bounding box threshold (threshold for merging event bounding boxes)

# Get spectrum with oevent
lms,lnoise,lsidx,leidx = oevent.getmorletwin(sig_sim, int(winsz*fs), fs, 
                                             freqmin=freqmin, freqmax=freqmax, 
                                             freqstep=freqstep,
                                             noiseamp=200.)

plt.figure()
plt.plot(lms[0].f, np.log(np.median(lms[0].TFR, axis=1)))

plt.figure()
plt.imshow(np.log(lms[0].TFR), aspect='auto')

#%%
chan = 0 # which channel to use for event analysis
lchan = [chan]

sig_sim = np.expand_dims(sig_sim, 0)
dout = oevent.getIEIstatsbyBand(sig_sim,winsz,fs,freqmin,freqmax,freqstep,medthresh,lchan,None,overlapth,getphase=True,savespec=True,normop=oevent.one_over_f_norm)
# dout = oevent.getIEIstatsbyBand(sig_sim,winsz,fs,freqmin,freqmax,freqstep,medthresh,lchan,None,overlapth,getphase=True,savespec=True,normop=oevent.mednorm)

df = oevent.GetDFrame(dout,fs, sig_sim, None, haveMUA=False) # convert the oscillation event data into a pandas dataframe

dfs = df[(df.peakF>5) & (df.peakF<15) & (df.filtsigcor>0.5) & (df.Fspan<10) & (df.ncycle>3)] 

dlms={chan:dout[0]['lms'] for chan in lchan};

plt.figure()
plt.imshow(dlms[0][0].TFR, aspect='auto', vmax=20)

for i in range(dfs.shape[0]):
    evv = oevent.eventviewer(df,sig_sim,None,time,fs,winsz,dlms) # create an event viewer to examine the oscillation events
    evv.specrange = (0,10) # spectrogram color range (in median normalized units)
    evv.draw(dfs.index[i],clr='red',drawfilt=True,filtclr='b',ylspec=(0.25,40),lwfilt=3,lwbox=3,verbose=False)
    evv.cbaxes = evv.fig.add_axes([0.925, 0.5+.135, 0.0125, 0.15]); oevent.colorbar(evv.specimg, cax=evv.cbaxes)
    
idx_win = dfs.windowidx.values
T_min = dfs.minT.values
T_max = dfs.maxT.values

# Pick an event
for i in range(len(idx_win)):

    lfp_bg = copy.copy(sig_sim)
    
    sidx = int(fs*winsz*idx_win[i])
    eidx = int(fs*winsz*(idx_win[i]+1))
    
    sidx_ev = sidx + round(T_min[i]/1e3 * fs)
    eidx_ev = sidx + round(T_max[i]/1e3 * fs)
    
    lfp_bg[chan, sidx_ev:eidx_ev] = np.nan
    
    plt.figure()
    plt.plot(time[sidx:eidx], lfp_bg[chan, sidx:eidx], 'k', linewidth=1)
    plt.plot(time[sidx_ev:eidx_ev], sig_sim[chan, sidx_ev:eidx_ev], 'r', linewidth=1)    

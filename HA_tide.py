# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 23:26:55 2018

@author: JiaSi
"""

import numpy as np
import pandas as pd
import lunardate

#60 tide
tide60_name = [
        'Sa',
        'Ssa',
        'Mm',
        'Msf',
        'Mf',
        '2Q1',
        'σ1',
        'Q1',
        'ρ1',
        'O1',
        'MP1',
        'M1',
        'χ1',
        'π1',
        'P1',
        'S1',
        'K1',
        'ψ1',
        'φ1',
        'θ1',
        'J1',
        'SO1',
        'OO1',
        'OQ2',
        'MNS2',
        '2N2',
        'μ2',
        'N2',
        'ν2',
        'OP2',
        'M2',
        'MKS2',
        'λ2',
        'L2',
        'T2',
        'S2',
        'R2',
        'K2',
        'MSN2',
        'KJ2',
        '2SM2',
        'MO3',
        'M3',
        'SO3',
        'MK3',
        'SK3',
        'MN4',
        'M4',
        'SN4',
        'MS4',
        'MK4',
        'S4',
        'SK4',
        '2MN6',
        'M6',
        'MSN6',
        '2MS6',
        '2MK6',
        '2SM6',
        'MSK6', 
        ]
tide60 = pd.DataFrame([0.0410686,
          0.0821373,
          0.5443747,
          1.0158958,
          1.0980331,
          12.8542862,
          12.9271398,
          13.3986609,
          13.4715145,
          13.9430356,
          14.0251729,
          14.4920521,
          14.5695476,
          14.9178647,
          14.9589314,
          15.0000000,
          15.0410686,
          15.0821353,
          15.1232059,
          15.5125897,
          15.5854433,
          16.0569644,
          16.1391017,
          27.3416964,
          27.4238337,
          27.8953548,
          27.9682084,
          28.4397295,
          28.5125831,
          28.9019669,
          28.9841042,
          29.0662415,
          29.4556253,
          29.5284789,
          29.9589333,
          30.0000000,
          30.0410667,
          30.0821373,
          30.5443747,
          30.6265120,
          31.0158958,
          42.9271398,
          43.4761563,
          43.9430356,
          44.0251729,
          45.0410686,
          57.4238337,
          57.9682084,
          58.4397295,
          58.9841042,
          59.0662415,
          60.0000000,
          60.0821373,
          86.4079380,
          86.9523127,
          87.4238337,
          87.9682084,
          88.0503457,
          88.9841042,
          89.0662415])
tide60.index = tide60_name
w = tide60*np.pi/180/3600
#---read raw data---
#rwavedata(Raw File Path,Row of Data Start)
#Raw Data Format:
#*st    yyyymmddhh height(mm)
#格式為中央氣象局資料格式
def rwavedata_cwb(wave_path,start_l):
    raw_wave_f = pd.read_fwf(wave_path, widths = [6,11,7,1], header=None) #切割行
    wave_raw = raw_wave_f.loc[start_l-1:,1:2]
    wave_raw.index = range(len(wave_raw))
    return wave_raw

#---date and wave_height to num---
#Date,Wave Height = data_wave(Data from rwavedata)
def date_wave(wave_raw):
    date_obs = pd.to_datetime(wave_raw.loc[:,1],format='%Y%m%d%H')
    date_obs_int = date_obs.values.astype(np.int64) // 10**9
    wave_height = wave_raw.loc[:,2].astype('float64')
    return date_obs,date_obs_int,wave_height

#----harmonic analyze----
#HA parameter,Amplitude,phase,angular = HA_tide(Date,Wave Height)
def HA_tide(wave_date,wave_height,w):
    #leastsquare
    t = pd.DataFrame(wave_date) #Observation time
    h = wave_height                #Observation wave height
    nan_ind = np.isnan(wave_height) #nan index
    t = t[~nan_ind]
    h = h[~nan_ind]
    A=np.ones((len(t),len(w)*2+1))
    for i in range(1,len(w)+1):
        A[:,2*i-1] = np.cos(w.iloc[i-1]*t).T
        A[:,2*i] = np.sin(w.iloc[i-1]*t).T
    para = np.linalg.lstsq(A,h)[0]
    #--amplitude and angular frequency--
    amp = np.ones((1,len(w)))
    pha_ang = amp
    for ii in range(1,len(w)+1):
      amp[:,ii-1] = np.sqrt(para[2*ii-1]**2 + para[2*ii]**2)/1000
      pha_ang[:,ii-1] = np.arctan(para[2*ii]/para[2*ii-1])
      if para[2*ii]<0:
          pha_ang[:,ii-1]=pha_ang[:,ii-1]+np.pi
      if pha_ang[:,ii-1]<0:
          pha_ang[:,ii-1]=pha_ang[:,ii-1]+2*np.pi
    pha_ang = pha_ang*180/np.pi
    return para,amp,pha_ang

#---calculate HA wave height---
# Caculate Wave = HA_wave(start time,end time,parameter)
#Import time pormat : yyyymmddhh
def HA_wave(start,end,para,w,mat='%Y%m%d%H'):
    #Make a integral time series
    start = pd.to_datetime(start,format=mat)
    end = pd.to_datetime(end,format=mat)
    date_n =  pd.date_range(start,end,freq='H')
    date_n_int = date_n.values.astype(np.int64) //10**9
    t_n = pd.DataFrame(date_n_int)    
    A_n=np.ones((len(t_n),len(w)*2+1))
    for i2 in range(1,len(w)+1):
        A_n[:,2*i2-1] = np.cos(w.iloc[i2-1]*t_n).T
        A_n[:,2*i2] = np.sin(w.iloc[i2-1]*t_n).T
    h_ha = np.dot(A_n,para)
    return h_ha,date_n

#---peak---
#toppeak,buttonpeak = peak(wave height)
def peak(wave_height):
    delt_w1=0
    peak_ind = []
    peak_t_ind = []
    peak_b_ind = []
    #---anypeak---
    for j in range(len(wave_height)-1):    
        delt_w = (wave_height.loc[j+1]-wave_height.loc[j])/2
        if (delt_w*delt_w1) <=0:
            peak_ind+=[j]
            if delt_w1>delt_w:
                peak_t_ind +=[j] #---toppeak---
            else:
                peak_b_ind +=[j] #---buttonpeak---
        delt_w1 = delt_w
    return peak_t_ind,peak_b_ind

#---statistic of wave---
#{MWL,MHWL,MLWL,HWL,LWL} = wave_statistic(wave height)
def wave_statistic(date_obs,wave_height):
    peak_t_ind,peak_b_ind = peak(wave_height)
    #lunar HA
    syzygy_date = [1,2,14,15,16,29,30]
    syzygy_ind =[]
    for i in range(0,len(date_obs)):
        year= date_obs[i].year
        month= date_obs[i].month
        day = date_obs[i].day
        lunar_date_temp = lunardate.LunarDate.fromSolarDate(year,month,day)
        if lunar_date_temp.day in syzygy_date:
            syzygy_ind += [i]
    peak_t_ind,peak_b_ind = peak(wave_height)
    set1 = set(peak_t_ind)
    set2 = set(syzygy_ind)
    set3 = set(peak_b_ind)
    hwost_ind = set1 & set2
    lwost_ind = set2 & set3
    #---WL---
    MWL  = np.mean(wave_height)
    MHWL = np.mean(wave_height[peak_t_ind])
    MLWL = np.mean(wave_height[peak_b_ind])
    HWL = np.mean(wave_height[hwost_ind])
    LWL = np.mean(wave_height[lwost_ind])
    sta_para = {'MWL':MWL,
                'MHWL':MHWL,
                'MLWL':MLWL,
                'HWL':HWL,
                'LWL':LWL}
    return sta_para

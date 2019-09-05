import os
import numpy as np
import time
from matplotlib import pyplot as plt
import subprocess
import math
from scipy.optimize import curve_fit as cf
from scipy.interpolate import interp1d
from matplotlib import style
# style.use('seaborn-whitegrid')
import random as rand
from ..pkgs import gem
from ..pkgs import simPyon as sim
import simPyon as sp

def volt_dict_pm_adj(volt_dict,pos_scale=1,neg_scale=1):
    out_dict = {}
    for elec in volt_dict:
        if volt_dict[elec] < 10000:
            out_dict[elec] = volt_dict[elec]*pos_scale if volt_dict[elec] > 0 else volt_dict[elec]*neg_scale
        else: out_dict[elec] = volt_dict[elec]
    return(out_dict)

def refine_it(pa_nam,gem_nam):
    coms = sim.simion(pa = pa_nam, gemfil = gem_nam)           
    coms.gem2pa(gem_nam,pa_nam+'#')
    coms.refine(pa_nam+'#')
    coms.get_elec_nums_gem(gem_nam)

def volt_adj_dict(elec_dict,pa_nam,gem_nam,scale_fact = 1):
#    volts = np.array([0,-713.6,0,-220.9,-828.8,1603.2,16000,1111.1,0])
    coms = sim.simion(pa = pa_nam, gemfil = gem_nam)           
    coms.get_elec_nums_gem()
    print(coms.elec_num)
    for val in coms.elec_num:
        print(coms.elect_dict[val])
    volts = [elec_dict[coms.elect_dict[val]]*(scale_fact if val < 16 else 1) for val in coms.elec_num]
    print(volts)
    coms.fast_adjust(pa_nam + '0',volts)      

def main_e_steps(volt_base_dict,pa_nam = [],gem_nam = [],n_parts = 10000):
    scale_fact = {1:.034735,
                  2:81.2/1212,
                  3:156/1212,
                  4:307/1212,
                  5:592/1212,
                  6:1,
                  7:1.93775}
    IMAP_base_scale = 1.0679
    coms = sim.simion(pa = pa_nam,gemfil = gem_nam)
    data = []
    for step in range(1,8):
        coms.volt_adj_dict(volt_base_dict,scale_fact = scale_fact[step])
        coms.parts.ke.dist_vals['max'] = 1000*scale_fact[step]
        data.append(coms.fly(n_parts = n_parts).data)
    
    e_peaks = []
    for dat in data:
        print("throughput = %f"%dat.throughput())
        plt.loglog()
        e_peaks.append(sp.data.gauss_fit(dat.good().start().df['ke'],weights = dat.good().start().df['counts'])[0][1])
    return(data,e_peaks)

def fly_steps(volt_base_dict,sim,n_parts = 10000,e_steps = np.arange(1,8)):
    scale_fact = {1:.034735,
                  2:81.2/1212,
                  3:156/1212,
                  4:307/1212,
                  5:592/1212,
                  6:1,
                  7:1.93775}
    # coms = sim.simion(pa = pa_nam,gemfil = gem_nam)
    data = []
    for step in e_steps:
        sim.volt_adj_dict(volt_base_dict,scale_fact = scale_fact[step])
        data.append(sim.fly(n_parts = n_parts).data)
    
    # e_peaks = []
    # for dat in data:
    #     print("throughput = %f"%dat.throughput())
    #     plt.loglog()
    #     e_peaks.append(sim.gauss_fit(dat.good().start().df['ke'],weights = dat.good().start().df['counts'])[0][1])
    return(data)


def main_eng_band_change(volt_dict,pa_nam=[],gem_nam=[],
    scales = np.linspace(.8,1.2,5),n_parts = 10000,E_max = 1000,de_e = 1,e=330):
    data = []
    coms = sim.simion(pa = pa_nam,gemfil = gem_nam)
    # scales = np.linspace(.6,1.4,8)
    for scale in scales:
        coms.volt_adj_dict(volt_dict_pm_adj(volt_dict,
            neg_scale = scale, 
            pos_scale = 1),scale_fact = 1,quiet = True)
        data.append(coms.fly(n_parts = n_parts).data)
    coms.volt_adj_dict(volt_dict)
    E = []
    for dat in data:
        # E.append(sim.gauss_fit(dat.good().start().df['ke'],
        #     weights = dat.good().start().df['counts'],n_tot = n_parts,plot = False)[0][1])
        E.append(np.average(dat.good().start().df['ke'],weights = dat.good().start().df['counts']))
    # plt.plot(np.array(E)/330,1/scales)
    lin_fit_e_params = np.polyfit(scales,E,1)
    print(lin_fit_e_params)
    poly_eng = np.poly1d(lin_fit_e_params)
    norm_dat = []
    for scale,eng in zip(scales,E):
        # print('=========================================')
        # print(volt_dict)
        coms.volt_adj_dict(volt_dict_pm_adj(volt_dict,
            neg_scale = scale, 
            pos_scale = 1),scale_fact = e/eng,quiet = True)
        norm_dat.append(coms.fly(n_parts = n_parts).data)
    coms.volt_adj_dict(volt_dict)
    de = []
    for dat in norm_dat:
        # de.append(sim.gauss_fit(dat.good().start().df['ke'],
        #     weights = dat.good().start().df['counts'],
        #     n_tot = n_parts,plot = True)[1])
        de.append(2*np.sqrt(2*np.log(2))*np.std(dat.good().start().df['ke'])/\
            np.average(dat.good().start().df['ke'],weights = dat.good().start().df['counts']))
    poly_de = np.poly1d(np.polyfit(de,scales,2))
    plt.figure()
    plt.plot(de,scales)
    multi_scales=np.linspace(de[0],de[-1],50)
    plt.plot(multi_scales, poly_de(multi_scales))
    # plt.plot(np.array(E)/330,1/scales)
    scale_out = poly_de(de_e)
    return(norm_dat,volt_dict_pm_adj(volt_dict,
        neg_scale = scale_out*e/poly_eng(scale_out),
        pos_scale = e/poly_eng(scale_out)))


def main_eng_band_change_2(volt_dict,scales = np.linspace(.8,1.2,5),
    n_parts = 10000,E_max = 1000,de_e = 1,e=330):
    data = []
    coms = sim.simion(pa = pa_nam,gemfil = gem_nam)
    # scales = np.linspace(.6,1.4,8)
    for scale in scales:
        coms.volt_adj_dict(volt_dict_pm_adj(volt_dict,
            neg_scale = scale, 
            pos_scale = 1),scale_fact = 1,quiet = True)
        data.append(coms.fly(n_parts = n_parts).data)
    coms.volt_adj_dict(volt_dict)
    E = []
    de = []
    for dat in data:
        # E.append(sim.gauss_fit(dat.good().start().df['ke'],
        #     weights = dat.good().start().df['counts'],n_tot = n_parts,plot = False)[0][1])
        # de.append(2*np.sqrt(2*np.log(2))*np.std(dat.good().start().df['ke'])/\
        #     np.average(dat.good().start().df['ke'],weights = dat.good().start().df['counts']))
        de.append(sim.gauss_fit(dat.good().start().df['ke'],
            weights = dat.good().start().df['counts'],
            n_tot = n_parts,plot = True)[1])
        E.append(np.average(dat.good().start().df['ke'],weights = dat.good().start().df['counts']))
    # plt.plot(np.array(E)/330,1/scales)
    poly_de = np.poly1d(np.polyfit(de,scales,2))
    lin_fit_e_params = np.polyfit(scales,E,1)
    plt.figure(1)
    plt.plot(de,scales)
    plt.figure(2)
    plt.plot(scales,E)
    # print(lin_fit_e_params)
    poly_eng = np.poly1d(lin_fit_e_params)
    # norm_dat = []
    # for scale,eng in zip(scales,E):
    #     # print('=========================================')
    #     # print(volt_dict)
    #     coms.volt_adj_dict(volt_dict_pm_adj(volt_dict,
    #         neg_scale = scale, 
    #         pos_scale = 1),scale_fact = e/eng,quiet = True)
    #     norm_dat.append(coms.fly(n_parts = n_parts).data)
    # coms.volt_adj_dict(volt_dict)
    # de = []
    # for dat in norm_dat:
    #     # de.append(sim.gauss_fit(dat.good().start().df['ke'],
    #     #     weights = dat.good().start().df['counts'],
    #     #     n_tot = n_parts,plot = True)[1])
    #     de.append(2*np.sqrt(2*np.log(2))*np.std(dat.good().start().df['ke'])/\
    #         np.average(dat.good().start().df['ke'],weights = dat.good().start().df['counts']))
    # poly_de = np.poly1d(np.polyfit(de,scales,2))
    # plt.figure()
    # plt.plot(de,scales)
    # multi_scales=np.linspace(de[0],de[-1],50)
    # plt.plot(multi_scales, poly_de(multi_scales))
    # # plt.plot(np.array(E)/330,1/scales)
    scale_out = poly_de(de_e)
    return(data,volt_dict_pm_adj(volt_dict,
        neg_scale = scale_out*e/poly_eng(scale_out),
        pos_scale = e/poly_eng(scale_out)))


def main_single_electrode_adj(volt_base_dict,scale_fact,electrodes,
        pa_nam=[],gem_nam=[],n_parts = 10000):
    coms = sim.simion(pa = pa_nam,gemfil = gem_nam)
    data = []
    volt_shift = dict((name,volt_base_dict[name]) for name in volt_base_dict)
    volts_out = []
    for scale in scale_fact:
        for electrode in electrodes:
            volt_shift[electrode] = volt_base_dict[electrode]*scale
        volts_out.append(dict(volt_shift))
        coms.volt_adj_dict(volt_shift,scale_fact = 1)
        data.append(coms.fly(n_parts = n_parts).data)
    
    th = []
    de_e = []
    for dat in data:
        th.append(dat.throughput())
        de_e.append(np.std(dat.good().start()()['ke'])/np.mean(dat.good().start()()['ke'])*2.634)
    plt.figure(1)
    plt.plot(scale_fact,th)
    plt.figure(2)
    plt.plot(scale_fact,de_e)
    return(data,volts_out[np.argmax(np.array(th))])

def dat_plot(data,comp = [],names = [''],save = False,n_tot = 10000,just_eng = False):    
    textstr = ''
    txt = None
    for out_data,name in zip([data]+comp,names):
        # textstr += name
        # lines = open(out_data).readlines()
        # data = np.genfromtxt(out_data,skip_header = 12, delimiter = ',')
        # headder = lines[10].replace('"','').replace('\n','').lower().split(',')
        # dat = sim_data(headder,data)
        dat = data
        plt.figure(1)
        de_e = sim.gauss_fit(dat.good().start().df['ke'],n_tot = n_tot,plot = True,
            label = name.strip(),weights = dat.good().start().df['counts'])[-1]
        th = len(dat.good().start().df['x'])/len(dat.start().df['x'])
        print(th)
        ax = plt.figure(1).axes[0]
        ax.legend(loc = 1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        textstr += '$\Delta E/E$: %.2f \nThrouput: %.2f'%(de_e,th)
        props = dict(facecolor='grey', alpha=0.2)
        if txt != None:
            txt.remove()
        # txt = ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            # verticalalignment='top', bbox=props)

        print(len(dat.start().df['x']))
        #%
        # for nam in dat.df:print(nam)
        #trash = plt.hist(dat.start().good().df['x'],75)
        #%
        if just_eng == False:
            plots = [(dat.good().start().df['x']-min(dat.start().df['x']))/np.cos(15/180*np.pi),
                    dat.good().start().df['theta'],
                    dat.good().stop().df['theta'],
                    dat.good().stop().df['r']]
            x_lables = ['Position on CS [mm]',
                        'Elevation Angle At CS [$\degree$]',
                        'Elevation angle at ToF Enterance [$\degree$]',
                        'Radial Position at ToF [mm]']
            n = 2
            for param,label in zip(plots,x_lables):
            # if plot == True:
                plt.figure(n)
                plt.hist(param,75,weights = dat.good().start().df['counts'],normed = True,alpha = .4,label = name.strip())
                plt.legend(loc = 1)
                plt.xlabel(label,fontsize = 14)
                plt.ylabel('Normalized Count Rate',fontsize = 14)
                n = n+1
                if label == 'Radial Position at ToF [mm]':
                    plt.axvline(45.1)
                    plt.axvline(35.4)

def main_estep_scan(volt_base_dict,n_parts = 10000,col = None,e_peaks = []):
    e_step_scale = np.array([.034735,81.2/1212,156/1212,307/1212,592/1212,1,1.93775])
    
    if e_peaks == []:
        # e_peaks = np.array([14,27,52,102,197,451,908])
        e_peaks = 330*e_step_scale
    
    # for the substeps
    e_peaks_sub = []
    for peak in e_peaks:
        e_peaks_sub+=[peak/(2**(n/4)) for n in np.linspace(3,0,4)]
    e_peaks_sub +=[peak*(2**(n/4)) for n in np.linspace(1,3,3)]
    print(e_peaks_sub)
    e_step_scale_sub = np.array(e_peaks_sub)/330
    e_peaks_sub = np.array(e_peaks_sub)
    # e_peaks = e_peaks_sub
    # e_step_scale = e_step_scale_sub
    coms = sim.simion(pa = pa_nam,gemfil = gem_nam)
    tot_th = []
    tot_e = []
    plt.figure(1)
    # beams = np.array([14,27,52,102,197,403,780])
    beams = [200]

    e_loss = np.array([.15,.13,.17,.17,.15,.16,.25,.27,.30])
    f_loss = interp1d(np.append(np.append(e_peaks_sub[0],e_peaks),e_peaks_sub[-1]),e_loss)
    volt_tot = []
    for beam_peak in beams:
        # volt_max = []
        data = []
        # e_peaks = np.concatenate([e_peaks_sub[e_peaks_sub<beam_peak][-3:],
        #                         e_peaks_sub[e_peaks_sub>beam_peak][:4]])
        # e_step_scale = np.concatenate([e_step_scale_sub[e_peaks_sub<beam_peak][-3:],
        #                         e_step_scale_sub[e_peaks_sub>beam_peak][:4]])
        coms.parts.ke.dist_type = 'gaussian'
        coms.parts.ke.dist_vals['mean'] = beam_peak*(1-f_loss(beam_peak))
        coms.parts.ke.dist_vals['fwhm'] = beam_peak
        for scale in e_step_scale:    
            coms.volt_adj_dict(volt_base_dict,scale_fact = scale)
            data.append(coms.fly(n_parts = n_parts).data)
            # plt.hist(coms.data.start().df['ke'],50,alpha = .2)
            # print(len(coms.data))
        
        th = []
        # plt.figure(2)
        for dat,e in zip(data,e_peaks):
            th.append(np.sum(dat.good().start()()['counts']))
            t,x = np.histogram(dat.good().start()()['ke'])
            # th.append(max(t))
            # plt.plot(x[:-1],t,color = col)
        #     # e_peaks.append(np.mean(dat.good().start().df['ke']))
        # e_w = []
        # for dat in data:

        # 
        plt.plot(e_peaks,th/max(th),'.-', label = str(beam_peak)+'eV')
        plt.semilogx()
        tot_th.append(th)
        tot_e.append(e_peaks)
        plt.legend()
    return(tot_th,tot_e,beams)

def main_voltage_optimize_th(volt_base_dict,electrodes,de_e = 1,de_e_tolerance = .1,
        generations = 15, siblings = 8, mut_w = .1,n_parts = 10000,e = 330):
    np.set_printoptions(precision = 2)

    coms = sim.simion(pa = pa_nam,gemfil = gem_nam)
    volt_shift = dict((name,volt_base_dict[name]) for name in volt_base_dict)
    volt_hold = dict((name,volt_base_dict[name]) for name in volt_base_dict)
    volts_gens = []
    th_tot = []
    th_gens = np.zeros(generations)
    # de_e_gens = np.zeros(generations)
    for i in range(generations):
        volts_siblings = []
        th_siblings = np.zeros(siblings)
        de_e_siblings = np.zeros(siblings)
        e_siblings = np.zeros(siblings)
        de_siblings = np.zeros(siblings)
        for j in range(siblings):
            muts = mut_w*(np.random.rand(len(electrodes)) - .5) + 1
            for electrode,mut in zip(electrodes,muts):
                volt_shift[electrode] = volt_hold[electrode]*mut
            volts_siblings.append(dict(volt_shift))
            coms.volt_adj_dict(volt_shift,scale_fact = 1,quiet = True)
            coms.fly(n_parts = n_parts,surpress_output = True)
            th_siblings[j] = coms.data.throughput()
            de_e_siblings[j] = np.std(coms.data.good().start()()['ke'])/np.mean(coms.data.good().start()()['ke'])*2.634
            e_siblings[j] = np.mean(coms.data.good().start()()['ke'])
            de_siblings[j] = np.std(coms.data.good().start()()['ke'])
        print('generation: %d'%i)
        print(th_siblings)
        print(de_e_siblings)
        th_siblings = th_siblings/de_siblings*e
        th_tot.append(th_siblings)
        good_th = th_siblings[abs(de_e - de_e_siblings) < de_e_tolerance]
        if good_th.size > 0:
            best_child = np.argwhere(th_siblings == np.max(good_th)).reshape(1)[0]
            if th_siblings[best_child] > max(th_gens):
                # print(best_child)
                th_gens[i] = th_siblings[best_child]
                # volts_gens.append(volt_dict_pm_adj(volts_siblings[best_child],pos_scale = e/e_siblings[best_child],
                #     neg_scale = e/e_siblings[best_child]))
                volts_gens.append(volts_siblings[best_child])
                # volt_hold = volt_dict_pm_adj(volts_siblings[best_child],pos_scale = e/e_siblings[best_child],
                #     neg_scale = e/e_siblings[best_child])
                volt_hold = volts_siblings[best_child]
                print_loc = ' '*np.array2string(th_siblings).find('%.2f'%th_siblings[best_child])+'  |'
                # print_loc=print_loc[:int((best_child+1)*len(np.array2string(th_siblings))/siblings)] + '|'
                print(print_loc)
            else:
                th_gens[i] = max(th_siblings)
                # volt_hold = volts_gens[np.argmax(th_gens)]
                volts_gens.append(volt_hold)
                print('x'*len(np.array2string(th_siblings)))
        else:
            th_gens[i] = max(th_siblings)
            volts_gens.append(volt_hold)
            print('-'*len(np.array2string(th_siblings)))
            # volt_hold = volts_siblings[best_child]
    plt.figure(1)
    plt.plot(np.arange(generations),th_gens,'.-')
    plt.ylabel('Throughput/ $\Delta E/E$')
    plt.xlabel('generation')
    # plt.figure(2)
    # plt.plot(scale_fact,de_e)
    return(th_gens,volts_gens,th_tot)

def main_voltage_optimize(volt_base_dict,electrodes,opt_param_func,
        constraint_func,opt_func = np.max,
        generations = 10, siblings = 8, mut_w = .1,n_parts = 10000,e = 330):
    np.set_printoptions(precision = 2)
    coms = sim.simion(pa = pa_nam,gemfil = gem_nam)
    volt_shift = dict((name,volt_base_dict[name]) for name in volt_base_dict)
    volt_hold = dict((name,volt_base_dict[name]) for name in volt_base_dict)
    volts_gens = []
    param_gens = np.zeros(generations)*np.nan
    coms.volt_adj_dict(volt_base_dict,quiet = True)
    coms.fly(n_parts = n_parts,surpress_output = True)
    param_gens[0] = opt_param_func(coms.data)
    print('generation: 0')
    print('[%.2f]'%param_gens[0])
    for i in range(1,generations):
        volts_siblings = []
        data_siblings = []
        param_siblings = np.zeros(siblings)
        for j in range(siblings):
            muts = mut_w*(np.random.rand(len(electrodes)) - .5) + 1
            for electrode,mut in zip(electrodes,muts):
                volt_shift[electrode] = volt_hold[electrode]*mut
            volts_siblings.append(dict(volt_shift))
            coms.volt_adj_dict(volt_shift,scale_fact = 1,quiet = True)
            coms.fly(n_parts = n_parts,surpress_output = True)
            data_siblings.append(coms.data)
            param_siblings[j] = opt_param_func(coms.data)
        print('generation: %d'%i)
        print(param_siblings)
        good_data_sibs = []
        good_volt_sibs = []
        good_param_sibs = []
        for dat,param in zip(data_siblings,param_siblings):
            if const_func(dat):
                good_param_sibs.append(param)
        # for dat,volt,param in zip(data_siblings,volts_siblings,param_siblings):
        #     if const_func(dat): 
        #         good_data_sibs.append(dat)
        #         good_volt_sibs.append(volt)
        #         good_param_sibs.appen(param)
        if len(good_param_sibs) > 0:
            best_child = np.argwhere(param_siblings == opt_func(good_param_sibs)).reshape(1)[0]
            if opt_func([opt_func(param_gens),param_siblings[best_child]])==\
                param_siblings[best_child]:
                param_gens[i] = param_siblings[best_child]
                volts_gens.append(volts_siblings[best_child])
                volt_hold = volts_siblings[best_child]
                print_loc = ' '*np.array2string(param_siblings).find('%.2f'%param_siblings[best_child])+'  |'
                print(print_loc)
            else:
                param_gens[i] = opt_func(param_siblings)
                volts_gens.append(volt_hold)
                print('x'*len(np.array2string(param_siblings)))
        else:
            param_gens[i] = max(param_siblings)
            volts_gens.append(volt_hold)
            print('-'*len(np.array2string(param_siblings)))
    plt.figure(1)
    plt.plot(np.arange(generations),param_gens)
    plt.xlabel('generation')
    plt.ylabel(opt_func.__name__+' parameter')
    # plt.figure(2)
    # plt.plot(scale_fact,de_e)
    return(param_gens,volts_gens)


def location_chng(shift_gem, out_gem,shift_names,shift_vals):
    with open(shift_gem) as lines:
        shift_lines = lines.readlines()
        for tag,val in zip(shift_names,shift_vals):
            n = 0
            for line in shift_lines:
                if tag in line:
                    shift_lines[n] = line.replace(tag,str(val))
                n+=1
    with open(out_gem,'w') as fil_out:
        for line in shift_lines:
            fil_out.write(line)

def get_sputtered_prod(volts,n = 10000):
    # generate deconvoluted Ibex-lo oxygen efficiency
    #   distribution
    def sputtered(x,a,b,c,d):
        # b = 2
        # c = .01
        # d = .9
        # e = .9
        return(a*x -b*(x-c)**2+ d*np.log(x**(-d)))

    E = np.array([12.2,22.3,43.4,86.5])/105
    dE = .7*E
    eff = np.array([.8,.9,1,.17])

    # n = eff*dE[2]/dE

    E_shif = np.copy(E)
    E_shif[3] = .62
    # eff_shif = eff.copy()
    dE_shif = .7*E_shif
    # dE_shif[3] = 2*(E_shif[3]-(E[3]-dE[3]/2))
    n_shif = eff*dE_shif[2]/dE_shif

    plt.semilogy(E_shif,n_shif,'.-')
    plt.bar(E,n_shif,width = .7*E,alpha = .2)

    x = np.linspace(.001,.85,200)
    params,errs = cf(sputtered,E_shif,n_shif,sigma = dE)
    # print(params)
    plt.plot(x,sputtered(x,params[0],params[1],params[2],params[3]))
    plt.xlabel('$E/E_{ISN}$',fontsize = 16)
    plt.ylabel('$n/n_3$',fontsize = 16)


    # Use devonvoluted distribution to test simulation
    #   response
    E_0 = 105
    E_ISN = E_0
    coms = sim.simion()
    coms.parts.mass = 32
    coms.parts.ke = coms.parts.dist('sputtered',n)
    coms.parts.ke.dist_vals['E_isn'] = E_ISN
    coms.parts.ke.dist_vals['b'] = E_shif[3]
    coms.parts.elv = coms.parts.dist('cos')

    dat = coms.fly_steps(volts,e_steps = np.arange(1,5),n_parts = n)
    th = np.array([t.throughput() for t in dat])
    E = np.array([12.2,22.3,43.4,86.5])
    eff = th#/(.7*E)
    params_2,errs_2 = cf(sputtered,E/E_0,eff/max(eff),[1.3,5.4,.39,.71],
        sigma = 1/np.sqrt(th))
    plt.figure()
    plt.loglog(E/E_0,eff/max(eff),'.-',
        label = 'IBEX-lo Simulated Efficiency')
    plt.plot(x,sputtered(x,params_2[0],params_2[1],params_2[2],params_2[3]),
        label = 'Simulation Fit')
    plt.plot(x,sputtered(x,1.316,5.4252,0.39939,0.70979),
        label = 'Observation Fit')
    # plt.loglog(np.logspace(-2,0,100),ibex_sput_out(np.logspace(-2,0,100)))
    parts_flown,x_flown = np.histogram(coms.parts.ke(),100)
    x_flown = (x_flown[1:]+x_flown[:-1])/2
    w = x_flown[1:]-x_flown[:-1]
    parts_flown = parts_flown/max((parts_flown[1:]+parts_flown[:-1])/2)
    # plt.bar(x_flown/E_0,parts_flown,alpha = .4,width = w[0]/E_0,
    #     label = 'Flown Energy Distribution')
    plt.xlabel('$E_i/E_{ISN}$',fontsize = 16)
    plt.ylabel('$R/R_3$',fontsize = 16)
    plt.ylim(.05,1.5)
    plt.xlim(.1,1)
    plt.legend(loc = 'lower left')

    # plot estimated oxygen distribution at CS
    primary = sp.particles.auto_parts.dist('gaussian')
    primary.dist_vals['mean'] = 425
    primary.dist_vals['fwhm'] = 100
    sputtered = sp.particles.auto_parts.dist('sputtered')

    plt.figure()
    plt.hist(np.concatenate((primary(10000),sputtered(8000))),200)
    # plt.semilogx()
    plt.xlabel('Oxygen Energy (ev)',fontsize = 16)
    plt.ylabel('counts',fontsize = 16)
    # plt.hist(coms.parts.ke(),density = True)
    return(params,params_2)
    # get IMAP and IBEX energy response to Energy distribution

def fly_sputtered_prod_compare(volt_ibex,volt_imap, n_main = 10000,n_sput = 5000):
    import simPyon as sim
    imap_data = []
    ibex_data = []
    coms = sim.simion.simion()
    n=n_main
    for volt,data in zip([volt_ibex,volt_imap],[ibex_data,imap_data]):
        coms.parts.ke = sim.particles.auto_parts.dist('gaussian')
        coms.parts.ke.dist_vals['mean'] = 425
        coms.parts.ke.dist_vals['fwhm'] = 100
        coms.parts.el = coms.parts.dist('gaussian')
        coms.parts.el.dist_vals['mean'] = 150
        coms.parts.el.dist_vals['fwhm'] = 19.2
        data.append(coms.fly_steps(volt,n))
        coms.parts.ke = sim.particles.auto_parts.dist('sputtered')
        coms.parts.el = coms.parts.dist('cos')
        data.append(coms.fly_steps(volt,n))

    imap_th = []
    ibex_th = []
    for th,dat in zip([ibex_th,imap_th],[ibex_data,imap_data]):
        for s_d in dat:
            sub_t = []
            for d in s_d:
                sub_t.append(d.throughput())
            th.append(np.copy(sub_t))

    steps = np.arange(1,8)
    for dat,col,nam in zip([ibex_th,imap_th],['red','blue'],['IBEX-lo','IMAP-lo']):
        plt.plot(center_energy,dat[0]*n_main,'.:',color = col,label = nam+':O Prime' )
        plt.plot(center_energy,dat[1]*n_sput,'+-',color = col,label= nam+':O Sputt' )
        plt.plot(center_energy,dat[0]*n_main+dat[1]*n_sput,'.--',color = col,label = nam+':Combined')
    plt.legend()
    plt.semilogx()
    plt.xlabel('Energy Step [eV]')
    plt.ylabel('counts')
    return(ibex_data,imap_data)
# def main_voltage_optimize(volt_base_dict,n_parts = 10000,de_e = 1):


# imap_loc = r'C:\Users\Jonny Woof\Google Drive\research\IMAP\full_model\base Sims\IMAP\ '
# imap_loc = r'C:\Users\Jonny Woof\Documents\Google Drive Local\IMAP Sims\full_model\base Sims\IMAP\ '
# # os.chdir(imap_loc)
# # shift_gem_nam = 'IMAP_lo_geo_ref_shift.GEM'
# imap_gem_nam = 'IMAP-Lo_CR3_CE4_TOF2_HK3.GEM'
# imap_pa_nam = r'pa\IMAP-Lo_CR3_CE4_TOF2_HK3.pa'
# ibex_loc = r'C:\Users\Jonny Woof\Documents\Google Drive Local\IMAP Sims\full_model\base Sims\IBEX\ '
# # shift_gem_nam = 'IMAP_lo_geo_ref_shift.GEM'
# ibex_gem_nam = 'IBEX-Lo_CR3_CE4_TOF2_HK3.GEM'
# ibex_pa_nam = r'pa\IBEX-Lo_CR3_CE4_TOF2_HK3.pa'
# IMAP_base_scale = 1.0679
# os.chdir(imap_loc)
# gem_nam = imap_gem_nam
# pa_nam = imap_pa_nam
# IMAP_base_scale = 1.0679
# volts_base = np.array([0,-713.6,0,-220,-828.8,1603.2,16000,1111.1,0])
volt_base_IMAP_dict = {'Conversion Surface':-713.6,
                    'P2 Electrode':-828.8,
                    'P3 Electrode': 0,
                    'Inner ESA':1111.1,
                    # 'Outter Esa':0,
                    'P9 Electrode':-220.9,
                    'P10 Electrode':1603.2,
                    'collimator':0,
                    'Rejection electrode inner radius':0,
                    'Rejection electrode outer radius':0,
                    'Optics deck and CS ground can.':0,
                    'TOF_PAC_VOLTAGE':16000,
                    'TOF_PAC_+1kV_VOLTAGE':16000,
                    'MCP Plane':16000}
volt_base_IBEX_dict = {'Conversion Surface':-415,
                    'P2 Electrode':-482,
                    'P3 Electrode': 0,
                    'Inner ESA':840, #814
                    'Outter Esa':0,
                    'P9 Electrode':-167,
                    'P10 Electrode':1212,
                    'collimator':0,
                    'Rejection electrode inner radius':0,
                    'Rejection electrode outer radius':0,
                    'Optics deck and CS ground can.':0,
                    'TOF_PAC_VOLTAGE':16000,
                    'TOF_PAC_+1kV_VOLTAGE':16000,
                    'MCP Plane':16000}

center_energy = np.array([12.2,22.3,43.4,86.5,164.7,338.9,664.8])
# imap = sim.simion()
# volts = np.load('volts_de_e_1_330_3.npy').item()
# volt_num_dict =  dict([num,volt_base_IMAP_dict[imap.elect_dict[num]]] for num in imap.elec_num)
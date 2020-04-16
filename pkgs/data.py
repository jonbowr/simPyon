import numpy as np
from scipy.optimize import curve_fit as cf
from scipy import stats
import math
from matplotlib import pyplot as plt
from ..defaults import *

def log_starts(ion_num):
    start_loc = np.ones(len(ion_num)).astype(bool)
    start_loc[1:] = (ion_num[1:] - ion_num[:-1]).astype(bool)
    return(start_loc)


def log_stops(ion_num):
    start_loc = np.ones(len(ion_num)).astype(bool)
    start_loc[:-1] = (ion_num[:-1] - ion_num[1:]).astype(bool)
    return(start_loc)


class sim_data:

    def __init__(self, data):
        self.data = data
        self.header = ["Ion N","TOF","X","Y","Z",
        "Azm","Elv","Vx","Vy","Vz","KE"]
        self.df = {}
        for head, arr in zip(self.header, np.transpose(data)):
            self.df[head.lower()] = arr
            
        #define the cylindrical symmetry coords
        self.df['r'] = np.sqrt(self.df['z']**2 + self.df['y']**2)
        self.df['omega'] = np.arctan2(self.df['z'],self.df['y'])
        self.df['omega'][self.df['omega']<0] += 360 
        vr = (self.df['z']*self.df['vz']+self.df['y']*self.df['vy'])/\
                np.sqrt(self.df['z']**2 + self.df['y']**2)
        self.df['vr'] = vr
        vtheta = (self.df['z']*self.df['vy']-\
            self.df['y']*self.df['vz'])/\
            np.sqrt(self.df['z']**2 + self.df['y']**2)
        self.df['vtheta'] = vtheta
        self.df['theta'] = np.arctan2(vr,np.sqrt(self.df['vx']**2+self.df['vtheta']**2))*180/np.pi
        self.df['phi'] = np.arctan2(vtheta,self.df['vx'])*180/np.pi

        # fix the count rate for increase in CS counts with radius
        if R_WEIGHT == True:
            self.df['counts'] = np.zeros(len(self.df['x']))
            if len(data)!=0:
                starts = log_starts(self.df['ion n'])
                stops = log_stops(self.df['ion n'])
                starty = min(self.df['r'][starts])
                self.df['counts'][starts] = self.df['r'][starts] / starty
                self.df['counts'][starts] = self.df['counts'][starts] * \
                    len(self.df['x'][starts]) /\
                    np.sum(self.df['counts'][starts])
                self.df['counts'][stops][:] = self.df['counts'][starts]

        # load the detection parameters From defaults so they can be actively updated
        self.obs = {'X_MAX':X_MAX,'X_MIN':X_MIN,
                    'R_MAX':R_MAX,'R_MIN':R_MIN}

    def __call__(self):
        return(self.df)

    def __len__(self):
        return(len(self.data))

    def __iter__(self):
        return(iter(self.df))

    def __getitem__(self,item):
        return(self.df[item])

    def throughput(self):
        return(np.sum(self.good().start().df['counts'])/\
            np.sum(self.start().df['counts']))

    def start(self):
        startz = log_starts(self.df['ion n'])
        return(sim_data(self.data[startz]))

    def stop(self):
        stopz = log_stops(self.df['ion n'])
        return(sim_data(self.data[stopz]))

    def good(self):
        stops = log_stops(self.df['ion n'])
        gx = np.logical_and(self.df['x'][stops] > self.obs['X_MIN'],
                    self.df['x'][stops] < self.obs['X_MAX'])
        gy = np.logical_and(self.df['r'][stops] < self.obs['R_MAX'],
                    self.df['r'][stops] > self.obs['R_MIN']) 
        goot = np.logical_and(gx, gy)

        good_ions = np.in1d(
            self.df['ion n'], self.df['ion n'][
                log_starts(self.df['ion n'])][goot])
        return(sim_data(self.data[good_ions]))

    def not_good(self):
        stops = log_stops(self.df['ion n'])
        gx = np.logical_and(self.df['x'][stops] > self.obs['X_MIN'],
                    self.df['x'][stops] < self.obs['X_MAX'])
        gy = np.logical_and(self.df['r'][stops] < self.obs['R_MAX'],
                    self.df['r'][stops] > self.obs['R_MIN']) 
        goot = np.logical_and(gx, gy)

        good_ions = np.in1d(
            self.df['ion n'], self.df['ion n'][
                log_starts(self.df['ion n'])][goot])
        return(sim_data(self.data[~good_ions]))

    def show(self,hist_bins = 50,variables = ['tof','ke','x','r','theta','phi']):
        fig, axs = plt.subplots(3,2)
        for ax,var in zip(axs.reshape(axs.size,1)[:,0],variables):
            ax.hist(self()[var],hist_bins,weights = self()['counts'],alpha = .4)
            ax.set_xlabel(var)
            ax.set_ylabel('counts')
        plt.tight_layout()

    def e_res(self,norm = None,label = '',plot = True):
        return(gauss_fit(self.good().start()()['ke'],
            weights = self.good().start()()['counts'],norm = norm,
            label = label,plot = plot))

    def de_e(self):
        return(np.std(self.good().start()()['ke'])/\
            np.mean(self.good().start()()['ke'])*2*np.sqrt(2*np.log(2)))

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
        
def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fwhm(ex,gauss_fit,gauss_param):

    lo_ex=ex[np.less(ex,gauss_param[1])]
    lo_fit=gauss_fit[np.less(ex,gauss_param[1])]      
    haf_down=lo_ex[find_nearest(lo_fit,gauss_param[0]/2)]

    hi_ex=ex[np.greater(ex,gauss_param[1])]
    hi_fit=gauss_fit[np.greater(ex,gauss_param[1])]      
    haf_up=hi_ex[find_nearest(-hi_fit,-gauss_param[0]/2)]
    return haf_up - haf_down,[haf_down,haf_up]
    
def gauss_fit(data,title = '', plot = True,n_tot = [],weights = None,label='',norm = None):
    rng = max(data)
    if n_tot == []:
        n_tot = len(data)
    bin_e,bin_edges=np.histogram(data,bins=np.linspace(0,rng,50),density=False,weights = weights)
    err = np.sqrt(bin_e)

    bin_mid=(bin_edges[1:]+bin_edges[:-1])/2
    gauss_param=cf(gauss,bin_mid,bin_e,(max(bin_e),np.average(bin_mid,weights=bin_e),
        np.average(bin_mid,weights=bin_e)),sigma = 1/err)[0]
    
    ex=np.linspace(min(data),rng,400)
    gauss_fit=gauss(ex,gauss_param[0],gauss_param[1],gauss_param[2])    

    width = abs(gauss_param[2]*2*np.sqrt(2*np.log(2)))
    width_locs = gauss_param[1]+np.array((-width/2,width/2))
    
    if norm == 'peak':
        gauss_fit=gauss(ex,gauss_param[0],gauss_param[1],gauss_param[2])/max(bin_e) 
        err = bin_e/max(bin_e)*np.sqrt(1/bin_e+1/np.max(bin_e))
        bin_e = bin_e/max(bin_e) 
    if type(norm) != str and norm != None:
        bin_e = bin_e/norm
        err = err/norm
        gauss_fit=gauss(ex,gauss_param[0],gauss_param[1],gauss_param[2])/norm
    if plot == True:
        print('\nFit parmameters: '+title)
        print('[       a                x0              sigma     ]')
        print(gauss_param)
        print('\nthroughput(%): '+str(len(data)/n_tot))
        print('E: '+str(gauss_param[1]))
        print('FWHM: '+str(width))
        print('dE/E: '+str(width/gauss_param[1]))
        print('skewness: '+str(stats.skew(data)))
        line = plt.plot(ex,gauss_fit,alpha=.5,label=label)[0]
        plt.errorbar(bin_mid,bin_e,err,fmt = '.',color = line.get_color())
        plt.axvline(width_locs[0],color = line.get_color())
        plt.axvline(width_locs[1],color = line.get_color())
        plt.ylabel('normalized count rate',fontsize = 14)
        plt.xlabel('Accepted Enterance Energy(ev)',fontsize = 14)
        plt.title(title)
        plt.show()
    return gauss_param,width/gauss_param[1]
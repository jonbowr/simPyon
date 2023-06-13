import numpy as np
from scipy.optimize import curve_fit as cf
from scipy import stats
import math
from matplotlib import pyplot as plt
from ..defaults import *
from pandas import DataFrame

def log_starts(ion_num):
    start_loc = np.ones(len(ion_num)).astype(bool)
    start_loc[1:] = (ion_num[1:] - ion_num[:-1]).astype(bool)
    return(start_loc)

def log_stops(ion_num):
    return(~log_starts(ion_num))

class sim_data:

    def __init__(self, data,
                 headder = ["Ion N","TOF","X","Y","Z",
                            "Azm","Elv","Vx","Vy","Vz","KE"],
                 symmetry='cylindrical',
                 mirroring='y',
                 obs = {'X_MAX':X_MAX,'X_MIN':X_MIN,
                        'R_MAX':R_MAX,'R_MIN':R_MIN,
                        'TOF_MEASURE':TOF_MEASURE,
                        'R_WEIGHT':R_WEIGHT}):

        if type(data) is np.ndarray:
            self.header = headder
            self.df = {}
            for head, arr in zip(self.header, np.transpose(data)):
                self.df[head.lower()] = arr
            
            base = {'x':'y','y':'x'}
            mirroring = mirroring.lower()
            symmetry = symmetry.lower()
            self.symmetry = symmetry
            self.mirror_ax = mirroring
            self.base_ax = base[mirroring]

            # load the detection parameters From defaults so they can be actively updated
            self.obs = obs

            if symmetry == 'cylindrical'\
                     or symmetry =='cyl':
                ax_mir = self.df[mirroring]
                vmir = self.df['v'+mirroring]
                ax_base = self.df[base[mirroring]]
                vbase = self.df['v'+base[mirroring]]
                #define the cylindrical symmetry coords
                self.df['r'] = np.sqrt(self.df['z']**2 + ax_mir**2)
                self.df['omega'] = np.arctan2(self.df['z'],ax_mir)
                # self.df['omega'][self.df['omega']<0] += 360 
                vr = (self.df['z']*self.df['vz'] +ax_mir*vmir)/\
                        np.sqrt(self.df['z']**2 +ax_mir**2)
                self.df['vr'] = vr
                vtheta = (self.df['z']*vmir-\
                   ax_mir*self.df['vz'])/\
                    np.sqrt(self.df['z']**2 +ax_mir**2)
                self.df['vtheta'] = vtheta
                self.df['theta'] = np.arctan2(vr,np.sqrt(vbase**2+self.df['vtheta']**2))*180/np.pi
                self.df['phi'] = np.arctan2(vtheta,vbase)*180/np.pi

                # fix the count rate for increase in CS counts with radius
                if obs['R_WEIGHT'] == True:
                    self.df['counts'] = np.zeros(len(ax_base))
                    if len(data)!=0:
                        starts = log_starts(self.df['ion n'])
                        stops = log_stops(self.df['ion n'])
                        starty = min(self.df['r'][starts])
                        cts = self.df['r'][starts] / starty
                        self.df['counts'][starts] =  cts* len(ax_base[starts]) /\
                            np.sum(cts)
                        self.df['counts'][stops] = self.df['counts'][starts]
                self.df = DataFrame(self.df)

        elif str(type(data)) == str(type(self)):
            self.header = list(data.df.keys())
            if type(data)==dict:
                self.df = {}
                for head, arr in data.df.items():
                    self.df[head.lower()] = arr
            else:
                self.df = data.df.copy()
            self.symmetry = data.symmetry
            self.mirror_ax = data.mirror_ax
            self.base_ax = data.base_ax
            self.obs = data.obs

    def __call__(self):
        return(self.df)

    def __iter__(self):
        return(iter(self.df.keys()))

    def __getitem__(self,item):
        if type(self.df)==DataFrame:
            return(self.df[item].values)
        elif type(self.df)==dict:
            return(self.df[item])

    def __setitem__(self,item,value):
        self.df[item] = value

    def __len__(self):
        return(len(self.df))

    def copy(self):
        return(sim_data(self))

    def data(self):
        dat = []
        head = []
        for nam,vals in self.df.items():
            head.append(nam)
            dat.append(vals)
        dat = np.stack(dat).T
        self.header = head
        return(dat)

    def mask(self,mask):
        cop = sim_data(self)
        cop.df = cop.df.iloc[mask]
        # for nam in cop:
            # cop[nam] = cop[nam][mask]
        return(cop)

    def drop(self,nam,axis = 1):
        cop = sim_data(self)
        cop.df = cop.df.drop(nam,axis = axis)
        return(cop)

    def append_col(self,data,name='',stage = 'both',
                            inplace = True,fill_val = np.nan):
        # if type(data) == pd.series:
        #     name = data.name
        if inplace:
            self.df[name] = np.ones(len(self.df))*fill_val
            starts = log_starts(self['ion n'])
            stops = log_stops(self['ion n'])
            if stage == 'start':
                self[name][starts] = data
            elif stage == 'stop':
                self[name][stops] = data
            elif stage == 'both':
                self[name][starts] = data
                self[name][stops] = data
        return(self)

    def throughput(self):
        return(np.sum(self.good().start().df['counts'])/\
            np.sum(self.start().df['counts']))

    def start(self):
        startz = log_starts(self['ion n'])
        return(self.mask(startz))

    def stop(self):
        stopz = log_stops(self['ion n'])
        return(self.mask(stopz))

    def log_good(self):
        stops = log_stops(self['ion n'])
        goot = np.logical_and.reduce([
                    self[self.base_ax][stops] > self.obs['X_MIN'],
                    self[self.base_ax][stops] < self.obs['X_MAX'],
                    self['r'][stops] < self.obs['R_MAX'],
                    self['r'][stops] > self.obs['R_MIN']
                    ])

        if self.obs['TOF_MEASURE'] == True:
            # need to remove to generalize class
            r_max = 49
            r_min = 28
            L = 50.2
            
            yf = self['y'][stops]+\
                L*self['vy'][stops]/self['vx'][stops]

            zf = self['z'][stops]+\
                L*self['vz'][stops]/self['vx'][stops]
            rf = np.sqrt(zf**2+yf**2)
            g_theta = np.logical_and(rf < r_max,
                                     rf > r_min)
            goot = np.logical_and(goot,g_theta)
            self.rf = rf[goot]
        good_ions = np.in1d(
            self['ion n'], self['ion n'][
                log_starts(self['ion n'])][goot])
        return(good_ions)

    def good(self):
        return(self.mask(self.log_good()))

    def not_good(self):
        return(self.mask(~self.log_good()))

    def show(self,hist_bins = 50,variables = ['tof','ke','x','r','theta','phi']):
        fig, axs = plt.subplots(int(np.ceil(len(variables)/2)),2)
        for ax,var in zip(axs.reshape(axs.size,1)[:,0],variables):
            ax.hist(self()[var],hist_bins,weights = self()['counts'],alpha = .4)
            ax.set_xlabel(var)
            ax.set_ylabel('counts')
        plt.tight_layout()

    def de_e(self):
        return(np.std(self.good().start()['ke'])/\
            np.mean(self.good().start()['ke'])*2*np.sqrt(2*np.log(2)))

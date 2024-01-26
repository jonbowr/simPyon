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
                 obs = {'X_MAX':float(X_MAX),
                        'X_MIN':float(X_MIN),
                        'R_MAX':float(R_MAX),
                        'R_MIN':float(R_MIN),
                        'TOF_MEASURE':bool(TOF_MEASURE),
                        'R_WEIGHT':float(R_WEIGHT)}):

        if str(type(data)) == str(type(self)):
            self.df = data.df.copy()
            self.header = list(data.df.keys())
            self.symmetry = str(data.symmetry)
            self.mirror_ax = str(data.mirror_ax)
            self.base_ax = str(data.base_ax)
            self.obs = dict(data.obs)
        else:
            self.header = [h.lower() for h in headder]
            if type(data) is np.ndarray:
                self.df = DataFrame(data,columns = [h.lower() for h in headder])
            elif type(data)==dict:
                self.df = DataFrame(data)[self.header]
            else:
                self.df = data.copy()[self.header]

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
                ax_mir = self.df[mirroring].values
                vmir = self.df['v'+mirroring].values
                ax_base = self.df[base[mirroring]].values
                vbase = self.df['v'+base[mirroring]].values
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

                self.df['counts'] = np.ones(len(ax_base))
                # fix the count rate for increase in CS counts with radius
                if obs['R_WEIGHT'] == True:
                    if len(data)!=0:
                        starts = log_starts(self.df['ion n'].values)
                        stops = log_stops(self.df['ion n'].values)
                        starty = min(self.df['r'][starts])
                        cts = self.df['r'][starts].values / starty
                        all_cts = np.ones(len(ax_base))
                        all_cts[starts] =  cts* len(ax_base[starts]) /\
                            np.sum(cts)
                        if np.sum(stops)>0:
                            all_cts[stops] = all_cts[starts]
                        self.df['counts'] = all_cts
                self.df['is_start'] = log_starts(self['ion n'])

    def __call__(self):
        return(self.df)

    def __iter__(self):
        return(iter(self.df.keys()))

    def __getitem__(self,item):
        if type(self.df)==DataFrame:
            return(self.df[item].values)
        elif type(self.df)==dict:
            return(self.df[item])

    def __str__(self):
        return(repr(self))

    def __repr__(self):
        return(str(type(self))+
               '\n Size:%s'%str(self.df.shape)+
               '\n Obs Region %s'%str(self.obs))

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
            starts = self['is_start']
            stops = ~self['is_start']
            if stage == 'start':
                self[name][starts] = data
            elif stage == 'stop':
                self[name][stops] = data
            elif stage == 'both':
                self[name][starts] = data
                self[name][stops] = data
        return(self)

    def append(self,data):
        dat = data.copy()
        dat['ion n'] = dat['ion n']+self['ion n'].max()
        self.df =self.df.append(dat)
        return(self)

    def throughput(self,weights = True):
        if weights:
            return(np.sum(self.good().start().df['counts'])/\
                np.sum(self.start().df['counts']))
        else:
            return(len(self.good().start())/\
                    len(self.start()))

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
        return(fig,axs)

    def de_e(self):
        return(np.std(self.good().start()['ke'])/\
            np.mean(self.good().start()['ke'])*2*np.sqrt(2*np.log(2)))

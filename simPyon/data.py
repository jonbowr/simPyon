import numpy as np
from scipy.optimize import curve_fit as cf
from scipy import stats
import math
from matplotlib import pyplot as plt
from ..defaults import *
import pandas as pd

def log_starts(ion_num):
    start_loc = np.ones(len(ion_num)).astype(bool)
    start_loc[1:] = (ion_num[1:] - ion_num[:-1]).astype(bool)
    return(start_loc)


def log_stops(ion_num):
    start_loc = np.ones(len(ion_num)).astype(bool)
    start_loc[:-1] = (ion_num[:-1] - ion_num[1:]).astype(bool)
    return(start_loc)


class sim_data:

    def __init__(self, data,
                 headder = ["Ion N","TOF","X","Y","Z",
                            "Azm","Elv","Vx","Vy","Vz","KE"],
                 symmetry='cylindrical',
                 mirroring='y',
                 obs = {'X_MAX':X_MAX,'X_MIN':X_MIN,
                        'R_MAX':R_MAX,'R_MIN':R_MIN,
                        'TOF_MEASURE':TOF_MEASURE}):

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
                if R_WEIGHT == True:
                    self.df['counts'] = np.zeros(len(ax_base))
                    if len(data)!=0:
                        starts = log_starts(self.df['ion n'])
                        stops = log_stops(self.df['ion n'])
                        # print('stop_no: %d'%np.sum(stops))

                        # print('start_no: %d'%np.sum(starts))
                        # print('start_no: %d'%np.sum(starts[stops]))
                        starty = min(self.df['r'][starts])

                        # self.df['counts'][starts] = self.df['r'][starts] / starty
                        # self.df['counts'][starts] = self.df['counts'][starts] * \
                        #     len(ax_base[starts]) /\
                        #     np.sum(self.df['counts'][starts])
                        cts = self.df['r'][starts] / starty
                        self.df['counts'][starts] =  cts* len(ax_base[starts]) /\
                            np.sum(cts)
                        self.df['counts'][stops] = self.df['counts'][starts]

        elif str(type(data)) == str(type(self)):
            self.header = list(data.df.keys())
            self.df = {}
            for head, arr in data.df.items():
                self.df[head.lower()] = arr
            self.symmetry = data.symmetry
            self.mirror_ax = data.mirror_ax
            self.base_ax = data.base_ax
            self.obs = data.obs

    def __call__(self):
        return(self.df)

    # def __len__(self):
    #     return(len(list(self.df.values())[0]))

    def __iter__(self):
        return(iter(self.df.keys()))

    def __getitem__(self,item):
        return(self.df[item])

    def __setitem__(self,item,value):
        self.df[item] = value

    # def __str__(self):
    #     return(str(self.df))

    # def __repr__(self):
    #     return(pd.DataFrame(self.df))

    # def _repr_pretty_(self, p, cycle):
    #     return(pd.DataFrame(self.df)._repr_pretty_(p, cycle))

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
        for nam in cop:
            cop[nam] = cop[nam][mask]
        return(cop)

    def throughput(self):
        return(np.sum(self.good().start().df['counts'])/\
            np.sum(self.start().df['counts']))

    def start(self):
        startz = log_starts(self.df['ion n'])
        return(self.mask(startz))

    def stop(self):
        stopz = log_stops(self.df['ion n'])
        return(self.mask(stopz))

    def good(self):
        stops = log_stops(self.df['ion n'])
        gx = np.logical_and(self.df[self.base_ax][stops] > self.obs['X_MIN'],
                    self.df[self.base_ax][stops] < self.obs['X_MAX'])
        gy = np.logical_and(self.df['r'][stops] < self.obs['R_MAX'],
                    self.df['r'][stops] > self.obs['R_MIN']) 
        goot = np.logical_and(gx, gy)

        if self.obs['TOF_MEASURE'] == True:
            r_max = 49
            r_min = 28
            L = 50.2
            # r_max = 47
            # r_min = 31.5
            # L=27.5
            
            yf = self.df['y'][stops]+\
                L*self.df['vy'][stops]/self.df['vx'][stops]

            zf = self.df['z'][stops]+\
                L*self.df['vz'][stops]/self.df['vx'][stops]
            rf = np.sqrt(zf**2+yf**2)
            # r_fin = self.df['r'][stops] +\
            #          L*np.sin(self.df['theta'][stops]*np.pi/180)*\
            #          np.cos(self.df['phi'][stops]*np.pi/180)
            # plt.figure()
            # plt.plot()
            # theta_max = np.arctan2(r_max - self.df['r'][stops],L)*180/np.pi
            # theta_min = np.arctan2(r_min - self.df['r'][stops],L)*180/np.pi
            # g_theta = np.logical_and(self.df['theta'][stops] < theta_max,
            #                          self.df['theta'][stops] > theta_min)

            g_theta = np.logical_and(rf < r_max,
                                     rf > r_min)
            goot = np.logical_and(goot,g_theta)
            self.rf = rf[goot]
        good_ions = np.in1d(
            self.df['ion n'], self.df['ion n'][
                log_starts(self.df['ion n'])][goot])
        return(self.mask(good_ions))

    def not_good(self):
        stops = log_stops(self.df['ion n'])
        gx = np.logical_and(self.df[self.base_ax][stops] > self.obs['X_MIN'],
                    self.df[self.base_ax][stops] < self.obs['X_MAX'])
        gy = np.logical_and(self.df['r'][stops] < self.obs['R_MAX'],
                    self.df['r'][stops] > self.obs['R_MIN']) 
        goot = np.logical_and(gx, gy)

        good_ions = np.in1d(
            self.df['ion n'], self.df['ion n'][
                log_starts(self.df['ion n'])][goot])
        return(self.mask(~good_ions))

    def show(self,hist_bins = 50,variables = ['tof','ke','x','r','theta','phi']):
        fig, axs = plt.subplots(int(np.ceil(len(variables)/2)),2)
        for ax,var in zip(axs.reshape(axs.size,1)[:,0],variables):
            ax.hist(self()[var],hist_bins,weights = self()['counts'],alpha = .4)
            ax.set_xlabel(var)
            ax.set_ylabel('counts')
        plt.tight_layout()

    def e_res(self,norm = 'bin',label = '',plot = True,fig= None, ax = None):
        if plot == True:
            print('\nthroughput(%): '+str(self.throughput()))
        return(gauss_fit(self.good().start()['ke'],
            weights = self.good().start()['counts'],norm = norm,
            label = label,plot = plot,fig = fig,ax = ax))

    def de_e(self):
        return(np.std(self.good().start()['ke'])/\
            np.mean(self.good().start()['ke'])*2*np.sqrt(2*np.log(2)))


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
        
def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

# def fwhm(ex,gauss_fit,gauss_param):

#     lo_ex=ex[np.less(ex,gauss_param[1])]
#     lo_fit=gauss_fit[np.less(ex,gauss_param[1])]      
#     haf_down=lo_ex[find_nearest(lo_fit,gauss_param[0]/2)]

#     hi_ex=ex[np.greater(ex,gauss_param[1])]
#     hi_fit=gauss_fit[np.greater(ex,gauss_param[1])]      
#     haf_up=hi_ex[find_nearest(-hi_fit,-gauss_param[0]/2)]
#     return haf_up - haf_down,[haf_down,haf_up]
def fwhm(ex,fit,param=[]):
    y = fit(ex,*param)
    peak = np.argmax(y)

    lo_ex=ex[ex<ex[peak]]
    lo_fit=y[ex<ex[peak]]      
    haf_down=lo_ex[find_nearest(lo_fit,y[peak]/2)]


    hi_ex=ex[ex>ex[peak]]
    hi_fit=y[ex>ex[peak]]      
    haf_up=hi_ex[find_nearest(-hi_fit,-y[peak]/2)]

    # hi_ex=ex[np.greater(ex,gauss_param[1])]
    # hi_fit=gauss_fit[np.greater(ex,gauss_param[1])]      
    # haf_up=hi_ex[find_nearest(-hi_fit,-gauss_param[0]/2)]

    # print((haf_up-haf_down)/ex[peak])
    return haf_up - haf_down,[haf_down,haf_up,ex[peak]]
    
def gauss_fit(data,title = '', plot = True,weights = None,label='',
              norm = 'bin',fig = None,ax = None,bino = 50):
    rng = max(data)
    w = np.std(data)
    binz = np.linspace(min(data)-w,rng+w,50)
    bin_e,bin_edges=np.histogram(data,bins=binz,density=False,weights = weights)
    if norm == 'bin':
        bin_e = bin_e/np.diff(bin_edges)
    err = np.sqrt(np.histogram(data,bins=binz,density=False)[0])

    bin_mid=(bin_edges[1:]+bin_edges[:-1])/2
    gauss_param=cf(gauss,bin_mid,bin_e,(max(bin_e),np.average(bin_mid,weights=bin_e),
        np.average(bin_mid,weights=bin_e)),sigma = 1/err)[0]
    
    ex=np.linspace(min(data),rng,400)
    gauss_fit=gauss(ex,gauss_param[0],gauss_param[1],gauss_param[2])    

    width = abs(gauss_param[2]*2*np.sqrt(2*np.log(2)))
    width_locs = gauss_param[1]+np.array((-width/2,width/2))
    
    if norm == 'peak':
        gauss_fit=gauss(ex,gauss_param[0],gauss_param[1],gauss_param[2])/max(bin_e) 
        # err = bin_e/max(bin_e)*np.sqrt(1/bin_e+1/np.max(bin_e))
        bin_e = bin_e/max(bin_e) 
    if type(norm) != str and norm != None:
        bin_e = bin_e/norm
        err = err/norm
        gauss_fit=gauss(ex,gauss_param[0],gauss_param[1],gauss_param[2])/norm
    if plot == True:
        print('\nFit parmameters: '+title)
        print('[       a                x0              sigma     ]')
        print(gauss_param)
        print('E: '+str(gauss_param[1]))
        print('FWHM: '+str(width))
        print('dE/E: '+str(width/gauss_param[1]))
        print('skewness: '+str(stats.skew(data)))

        if not ax:
            fig,ax = plt.subplots()

        # else:
        line = ax.plot(ex,gauss_fit,alpha=.5,label=label)[0]
        ax.errorbar(bin_mid,bin_e,bin_e/err,fmt = '.',color = line.get_color())
        ax.axvline(width_locs[0],color = line.get_color())
        ax.axvline(width_locs[1],color = line.get_color())
        ax.set_ylabel('normalized count rate',fontsize = 14)
        ax.set_xlabel('Accepted Recoil Ion Energy(ev)',fontsize = 14)
        ax.set_title(title)
            # plt.show()
    return gauss_param,width/gauss_param[1]


def bin_fit(data,weights = None,fittype = 'gauss',bino= 50,norm_data = None,norm_weights = None):

    def gauss(x, a=1, x0=1, sigma=.5):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def log_gauss(x,a=1,sigma=1,mu=0):
        return(a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))

    def tanh(xl,a0 = 1,loc =1,scale =10):
        x = (xl- loc)/scale
        return(a0*(np.tanh(x)+1)/2)

    def log_gauss_flip(xi,a=1,sigma=1,mu=0,x0 = 2):
        x = abs(1-xi/x0)
        return(a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2))*np.heaviside(loc-xi,1))
    
    # def tanh_trunk_gauss(xi,a1=1,sigma=1,mu=0,loc1 = 2,loc2= .5,scale2= 10):
    #     x2 = (xi- loc2)/scale2
    #     x1 = abs(1-xi/loc1)
    #     return((((np.tanh(x2)+1)/2))*(a1*(x1*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x1) - mu)**2/(2*sigma**2))*np.heaviside(loc1-xi,1)))

    def tanh_trunk_gauss(xi,a1=1,sigma=1,mu=0,x01 = 2,x02= .5,lamb= 10):
        x1 = abs(1-xi/x01)
        x2 = (xi- x02)/lamb
        return((((np.tanh(x2)+1)/2))*(a1*(x1*sigma*np.sqrt(2*np.pi))**-1*\
                    np.exp(-(np.log(x1) - mu)**2/(2*sigma**2))*np.heaviside(x01-xi,1)))

    def skew_gauss(x, a=1, loc=.25, scale = .3):
        return(stats.skewnorm.pdf(x,a, loc, scale))

    def skew_trunk_gauss(xi,a=1,loc = .25,scale= .3,x02= .5,zig = 10):
        return(skew_gauss(xi,a,loc,scale)*tanh(xi,x02,zig))

    def exp_gauss(x,k=1,loc=.25,scale=.3):
        return(stats.exponnorm.pdf(x,k,loc,scale))

    e_funcs = {
              'gauss': gauss,
              'log_gauss':log_gauss,
              'tanh':tanh,
              'log_gauss_flip': log_gauss_flip,
              'log_trunk_gauss':tanh_trunk_gauss,
              'skew_gauss':skew_gauss,
              'skew_trunk_gauss':skew_trunk_gauss
              # 'exp_gauss':exp_gauss
              }


    # rng = max(data)
    mean = np.average(data,weights = weights)
    std = np.std(data)

    mino = np.min(data)-std
    maxo = np.max(data)+std
    y,bin_edges=np.histogram(data,bins=np.linspace(mino,maxo,bino),
                                                density=False,weights = weights)
    # if type(norm_data)==np.array:

    y = np.nan_to_num(y/np.histogram(norm_data,bins=np.linspace(mino,maxo,bino),
                                                density=False,weights = norm_weights)[0])
    x=(bin_edges[1:]+bin_edges[:-1])/2
    cnts = np.histogram(data,bins=np.linspace(min(data),max(data),bino))[0]
    err = 1/np.sqrt(cnts)

    
    if fittype is not 'spline':

        from scipy.stats import lognorm,skewnorm
        e_guess = {
              'gauss': [np.max(y),mean,std],
              'log_gauss':lognorm.fit(data),
              'tanh':None,
              'log_gauss_flip': [np.max(y),mean,mino,maxo],
              'log_trunk_gauss':[np.max(y),mean,mino,maxo,mean-2*std,std],
              'skew_gauss':skewnorm.fit(data),
              'skew_trunk_gauss':list(skewnorm.fit(data))+[mean-2*std,std]
              # 'exp_gauss':exponnorm.fit(data)
              }  


        parms=cf(e_funcs[fittype],x,y,e_guess[fittype],sigma = err)
        
        # def out_func(x):
        #     return(e_funcs[fittype](x,*parms[0]))
        return(func(e_funcs[fittype],*parms),[x,y,y*err])
    else:
        from scipy.interpolate import interp1d as up
        from scipy.ndimage import gaussian_filter1d as gf
        return(func(up(x,gf(gf(y,1),1),kind = 'cubic'),[]),[x,y,y*err])

class func:

    def __init__(self,func,params,cov = None):
        self.f = func
        self.params = params
        self.cov = cov

    def __call__(self,x):
        if type(self.params) == dict:
            return(self.f(x,**self.params))
        else:
            return(self.f(x,*self.params))
            

from scipy.stats import rv_continuous
class rv_func:


    def __init__(self,func,data,p0 = None):
        
        def gauss(x, a=1, x0=1, sigma=.5):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        def log_gauss(x,a=1,sigma=1,mu=0):
            return(a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2)))

        def tanh(xl,x0 =1,sigma =10):
            # x = (xl/np.nanmax(xl)- x0/np.nanmax(xl))*2*sigma
            x = (xl- x0)*2*sigma/np.nanmax(xl)
            y = (np.tanh(x)+1)/2
            return(y)

        def log_gauss_flip(xi,a=1,sigma=1,mu=0,x0 = 2):
            x = abs(1-xi/x0)
            y = a*(x*sigma*np.sqrt(2*np.pi))**-1*np.exp(-(np.log(x) - mu)**2/(2*sigma**2))*np.heaviside(x0-xi,.5)
            # y[xi>x0]= 0
            return(y)    
        
        def tanh_trunk_gauss(xi,a=1,sigma=1,mu=0,x0 = 2,x02= .5,zig = 10):
            return(log_gauss_flip(xi,a,sigma,mu,x0)*tanh(xi,x02,zig))

        def skew_gauss(x, a=1, loc=.25, scale = .3):
            return(stats.skewnorm.pdf(x,a, loc, scale))

        def skew_trunk_gauss(xi,a=1,loc = .25,scale= .3,x02= .5,zig = 10):
            return(skew_gauss(xi,a,loc,scale)*tanh(xi,x02,zig))

        def exp_gauss(x,k=1,loc=.25,scale=.3):
            return(stats.exponnorm.pdf(x,k,loc,scale))

        def poisson(x1,b=1,c=0,k=1):
            a=1
            x = (x1-c)/b
            norm = k**k*np.exp(-k)/np.math.factorial(k)
            return(a*(x*k**2)**k*np.exp(-x*k**2)/np.math.factorial(k)/norm)

        e_funcs = {
              'gauss': gauss,
              'log_gauss':log_gauss,
              'tanh':tanh,
              'log_gauss_flip': log_gauss_flip,
              'log_trunk_gauss':tanh_trunk_gauss,
              'skew_gauss':skew_gauss,
              'skew_trunk_gauss':skew_trunk_gauss
              # 'exp_gauss':exp_gauss
              }
        self.data = data
        self.pdf = rv_continuous()
        self.pdf._pdf = e_funcs[func]
        self.p0 = (p0 if p0 else self.pdf.fit(data))
        # self.pdf.fit(data)
        # self.f = e_funcs[func_nam]

    #     from scipy.stats import lognorm,skewnorm
    #     e_guess = {
    #       'gauss': [np.max(y),mean,std],
    #       'log_gauss':lognorm.fit(data),
    #       'tanh':None,
    #       'log_gauss_flip': [np.max(y),mean,mino,maxo],
    #       'log_trunk_gauss':[np.max(y),mean,mino,maxo,mean*.1,mean*.1],
    #       'skew_gauss':skewnorm.fit(data),
    #       'skew_trunk_gauss':list(skewnorm.fit(data))+[mean*.1]*2
    #       # 'exp_gauss':exponnorm.fit(data)
    #       }  
    #     self.f = func
    #     self.params = params
    #     self.cov = cov

    def __call__(self,x):
        return(self.pdf._pdf(x,*self.p0))


def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """

    from scipy.interpolate import interpn
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , 
                np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    return ax
import numpy as np
from scipy import stats
import math
from ..defaults import *


def rand_line(v1,v2,size):
    d = v2 - v1

    if d[0] !=0:
        dx_rand = np.random.uniform(0,d[0],size)
        m = d[1]/d[0]
        return(np.transpose(np.stack((dx_rand+v1[0],m*dx_rand+v1[1]))))
    else:
        dy_rand = np.random.uniform(v1[1],v2[1],size)
        return(np.transpose(np.stack([np.ones(size)*v1[0],dy_rand])))

class pdf:

    def sputtered(x,a,b,c,d):
        # a = -2.06741184,b = 12.01113288,c =  0.76598885,d = 1.94674493
        # a,b,c,d = (-2.12621554, 12.28266708,  0.762046  ,  1.95374093)
        return(a*x - b*(x-c)**2+ d*np.log(x**(-d)))

    def cos(x,x_min):
        return((np.cos(x*np.pi-np.pi/2)*(1-x_min)+x_min))

    # def poisson(x,k):
    #     # k = kwargs['k']
    #     norm = k**k*np.exp(-k)/np.math.factorial(k)
    #     return((x*k**2)**k*np.exp(-x*k**2)/np.math.factorial(k)/norm)
    
    def log(x):
        return(1/x)

    def secondary_elec(x,a0,a1):
        # a0 = kwargs['a0']
        # a1 = kwargs['a1']
        k = (a0-a1)**3*4**4/3**3
        return(k*(x-a0)/(x-a1)**4)

    def poisson(x1,b,c,k):
        a=1
        x = (x1-c)/b
        norm = k**k*np.exp(-k)/np.math.factorial(k)
        return(a*(x*k**2)**k*np.exp(-x*k**2)/np.math.factorial(k)/norm)

    func_dict = {
                'sputtered':sputtered,
                'cos':cos,
                # 'poinsson': poisson,
                'log': log,
                'secondary_elec':secondary_elec,
                'poisson':poisson
                }

    def_values = {
                'sputtered':{
                            'a': -2.12621554,
                             'b': 12.28266708,
                             'c': 0.762046,
                             'd':1.95374093
                             },
                'cos':{'x_min':0},
                # 'poinsson': {'k':3},
                'log': {},
                'secondary_elec':{'a0':.4286,
                                  'a1': -4.544},
                'poisson':{'b':.2, 
                            'c':.05,
                            'k':1}
                }
    def __init__(self,f='sputtered',kwargs = {}):
        self.f =  self.func_dict[f]
        self.kwargs = {}
        for key,val in self.def_values[f].items():
            self.kwargs[key] = val
        # self.kwargs = self.def_values[f]
        for t in kwargs:
            self.kwargs[t] = kwargs[t]

    def __call__(self,x):
        return(self.f(x,**self.kwargs))

    def __getitem__(self,item):
        return(self.kwargs[item])

    def __setitem__(self,item,value):
        self.kwargs[item] = value

    def sample(self,n,a=0,b=1):
        # if log_space == True:
        #     x = np.logspace(np.log(a),np.log(b),n)
        # else:
        # x = np.linspace(a,b,n)
        x = np.random.rand(n)*(b-a)+a
        y = self(x)
        select = np.repeat(x,abs(y/(max(y)-min(y))*np.log(n)**3).astype(int))
        return(np.random.choice(select, n))  


class source:

    def gaussian(self,n):
        return(np.random.normal(self.dist_vals['mean'],
                abs(self.dist_vals['fwhm']/(2*np.sqrt(2*np.log(2)))), n))

    def uniform(self,n):
        return(np.random.uniform(self.dist_vals['min'],self.dist_vals['max'],n))

    def line(self,n):
        d = self.dist_vals['last'] - self.dist_vals['first']
        dx_rand = np.random.uniform(0,d[0],n)
        m = d[1]/d[0]
        return(np.transpose(np.stack((dx_rand+self.dist_vals['first'][0],
            m*dx_rand+self.dist_vals['first'][1]))))

    def single(self,n):
        return(np.ones(n)*self.dist_vals['value'])

    def sputtered(self,n):
        return(pdf('sputtered').sample(n,
            self.dist_vals['a'],self.dist_vals['b'])*self.dist_vals['E_beam'])

    def cos(self,n):
        fnc = pdf('cos')
        fnc.kwargs['x_min'] = self.dist_vals['x_min']
        return((fnc.sample(n,self['a']/self['range'],
                                  self['b']/self['range'])*self.dist_vals['range']\
            -self.dist_vals['range']/2)+self.dist_vals['mean'])

    def sample_vector(self,n):
        self.dist_vals['index'] = np.random.choice(len(self.dist_vals['vector']),n)
        return(self.dist_vals['vector'][self.dist_vals['index']])

    def coupled_vector(self,n):
        self.dist_vals['parent'].dist_vals['type'] = 'parent'
        return(self.dist_vals['vector'][self.dist_vals['parent'].dist_vals['index']])

    def cos_coupled_vector(self,n):
        self.dist_vals['parent'].dist_vals['type'] = 'parent'
        return((pdf('cos').sample(len(self.dist_vals['parent'].dist_vals['index']))*180-180/2)+\
               self.dist_vals['vector'][self.dist_vals['parent'].dist_vals['index']])
        # return((pdf('cos').sample(len(self.dist_vals['parent'].dist_vals['index']))*180-180/2))

    def beam_2_poisson(self,n):
        return(self.dist_vals['E_beam']*abs(self.dist_vals['b']-pdf('poisson').sample(n,
            self.dist_vals['a'],self.dist_vals['b'])))

    def fixed_vector(self,n):
        return(self.dist_vals['vector'][:n])

    def log_uniform(self,n):
        return(pdf('log').sample(n,
            self.dist_vals['min'],self.dist_vals['max']))

    def coupled_func(self,n):
        if self.dist_vals['type'] == 'parent':
            # not going to reset to none type after initilaized
            # if type(self.dist_vals['sample']) == 'NoneType':
            self.dist_vals['sample'] = self.dist_vals['sample_func'](n)
            return(self.dist_vals['f'](self.dist_vals['sample']))
        elif self.dist_vals['type'] == 'child':
            if self.dist_vals['parent'].dist_vals['sample'] is None:
                self.dist_vals['parent'].dist_vals['sample'] = \
                    self.dist_vals['sample_func'](n)
            return(self.dist_vals['f'](self.dist_vals['parent'].dist_vals['sample']))

    def secondary_elec(self,n):
        return(pdf('secondary_elec').sample(n,self.dist_vals['a'],self.dist_vals['b']))

    def circle_pos(self,n):
        angs = np.random.rand(n)*(self.dist_vals['max_ang']-\
                                  self.dist_vals['min_ang'])+\
                                    self.dist_vals['min_ang']
        dx = np.cos(angs*np.pi/180)*self.dist_vals['r']
        dy = np.sin(angs*np.pi/180)*self.dist_vals['r']
        return(np.stack([dx+self.dist_vals['origin'][0],
                        dy+self.dist_vals['origin'][1]]).T)

    def dependent_func(self,n):
            if self.dist_vals['parent'].dist_out is None or len(self.dist_vals['parent'].dist_out)!=n:
                self.dist_vals['parent'](n)
            return(self.dist_vals['f'](self.dist_vals['parent'].dist_out))

    def pdf(self,n):
        return(self.dist_vals['f'].sample(n,a = self.dist_vals['a'],b = self.dist_vals['b']))
    
    def __init__(self,dist_type='',n=1,dist_vals = {}):

        func_dict = {'gaussian':self.gaussian,
                'uniform':self.uniform,
                'line':self.line,
                'single':self.single,
                'sputtered':self.sputtered,
                'cos':self.cos,
                'sample_vector':self.sample_vector,
                'coupled_vector':self.coupled_vector,
                'cos_coupled_vector':self.cos_coupled_vector,
                'fixed_vector':self.fixed_vector,
                'log_uniform':self.log_uniform,
                'coupled_func':self.coupled_func,
                'secondary_elec':self.secondary_elec,
                'circle_pos':self.circle_pos,
                'dependent_func':self.dependent_func,
                'pdf':self.pdf,
                'new':None}

        func_defaults  = {'gaussian':{'mean':0,'fwhm':1},
                'uniform':{'min':0,'max':1},
                'line':{'first':np.array([0,0,0]),'last':np.array([1,1,0])},
                'single':{'value':0},
                'sputtered':{'E_beam':105,'a':.01,'b':.65},
                'cos':{'mean':75,'range':180,'a':0,'b':180,'x_min':0},
                'sample_vector':{'vector':np.linspace(0,1,1000)},
                'coupled_vector':{'vector':np.linspace(0,1,1000),
                                'parent':None,
                                'type':'child'},
                'cos_coupled_vector':{'vector':np.linspace(0,1,1000),
                                'parent':None,
                                'type':'child'},
                'fixed_vector':{'vector':np.linspace(0,1,1000)},
                'log_uniform':{'min':0.01,'max':1},
                'coupled_func':{'f':[],
                                'sample_func':np.random.rand,
                                'type':'child',
                                'sample':None,
                                'parent':None},
                'secondary_elec':{'a':0,'b':50},
                'circle_pos':{'origin':np.array([0,0]),
                                'r':10,
                                'min_ang':0,
                                'max_ang':90},
                'dependent_func':{'f':[],
                                'type':'child',
                                'parent':None},
                'pdf':{'f':[],'a':0,'b':1},
                'new':{}
                }

        self.defaults = func_defaults
        # if type(dist_type) == str:
        if dist_type == '':
            dist_type = 'uniform'

        self.dist_type = dist_type.lower()
        self.f = func_dict[self.dist_type]
        self.dist_out =  None
        if dist_vals == {}:
            self.dist_vals = func_defaults[self.dist_type]
        elif all(list(name in func_defaults[self.dist_type] for name in dist_vals)) or dist_type == 'new':
            self.dist_vals = dist_vals
        else:
            print('WARNING: dist_vals provided not supported')

        self.n = n

    def __call__(self,n = []):
        if n != []:
            self.n = n
        self.dist_out = self.f(self.n)
        # self.dist_out[np.isnan(self.dist_out)] = -999999999
        return(self.dist_out)

    def __str__(self):
        return(str([self.dist_type,self.dist_vals]))

    def __getitem__(self,item):
        return(self.dist_vals[item])

    def __setitem__(self,item,value):
        self.dist_vals[item] = value

class auto_parts:
    
    def __init__(self, fil='auto_ion.ion', n=10000):
        import pandas as pd
        self.fil = fil
        # distribution defaults
        self.df = pd.Series({'n':n,
                              'mass':MASS,
                              'charge':CHARGE,
                              'ke':source(str(KE_DIST_TYPE),n,dist_vals =KE_DIST_VALS.copy()),
                              'az':source(str(AZ_DIST_TYPE),n,dist_vals = AZ_DIST_VALS.copy()),
                              'el':source(str(EL_DIST_TYPE),n,dist_vals = EL_DIST_VALS.copy()),
                              'pos':source(str(POS_DIST_TYPE),n,dist_vals = POS_DIST_VALS.copy()),
                              'tof':source('single',n,dist_vals = {'value':0})})

    def __getitem__(self,item):
        return(self.df[item])

    def __setitem__(self,item,value):
        self.df[item] = value

    def __str__(self):
        return(str(self.df))

    def sample(self):
        for samp_dist in self[['ke','az','el','pos','tof']].values:
            if 'type' in samp_dist.dist_vals:
                if samp_dist.dist_vals['type'] == 'parent':
                    distor = samp_dist(n = self['n'])
                    
        for samp_dist in self[['ke','az','el','pos','tof']].values:
            if 'type' in samp_dist.dist_vals:
                pass
            else:
                samp_dist(n = self['n'])
        
        for samp_dist in self[['ke','az','el','pos','tof']].values:
            if 'type' in samp_dist.dist_vals:
                if samp_dist.dist_vals['type'] == 'child':
                    distor = samp_dist(n = self['n'])

    # Ion File output headder
    #time of birth,mass,charge,x0,y0,z0,azimuth, elevation, energy, cfw, color
    def ion_print(self):
        self.sample()
        ke = abs(self['ke'].dist_out)
        az = self['az'].dist_out
        el = self['el'].dist_out
        pos = self['pos'].dist_out
        charge = self['charge']
        mass = self['mass']
        tof = self['tof'].dist_out
        x = pos[:,0]
        y = pos[:,1]

        if type(self.fil) == str:
            fils = [self.fil]
        else: fils = self.fil

        sub_num = int(self.df['n']/len(fils))
        f_count = 0
        for f in fils:
            with open(f, 'w') as fil:
                for n in range(f_count*sub_num,(f_count+1)*sub_num):
                    if np.sum(np.isnan(np.array([0,mass,charge,x[n],y[n],0,
                         az[n],el[n],ke[n],1,1])))==0:
                        fil.write("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f \n"%\
                            (tof[n],mass,charge,pos[n,0],pos[n,1],0,
                             az[n],el[n],ke[n],1,1))
            f_count +=1

    def splat_to_source(self,splat):
        # function which takes the simPyon.data.sim_data.df frame and input df components to source 
        # components, assumes cilidrical coords to r,phi,theta to y,az,el
        self.df['ke'] = source('fixed_vector',dist_vals = {'vector':splat['ke']})
        self.df['az'] = source('fixed_vector',dist_vals = {'vector':splat['phi']})
        self.df['el'] = source('fixed_vector',dist_vals = {'vector':splat['theta']})
        self.df['pos'] = source('fixed_vector',dist_vals = {'vector':np.stack([splat['x'],splat['r']]).T})
        self.df['n'] = len(splat['ke'])
        self.df['tof'] = source('fixed_vector',dist_vals = {'vector':splat['tof']})


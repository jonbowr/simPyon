import numpy as np
from scipy import stats
import math
from ..defaults import *

class sim_parts:

    class dist:
        def __init__(self, dist_type):
            self.dist_type = dist_type
            self.dist_vals = {}
            if self.dist_type.lower() == 'gaussian':
                self.dist_vals['mean'] = 0
                self.dist_vals['fwhm'] = 1
            if self.dist_type.lower() == 'uniform':
                self.dist_vals['min'] = 0
                self.dist_vals['max'] = 1
            if self.dist_type.lower() == 'line':
                self.dist_vals['first'] = np.array([0, 1, 0])
                self.dist_vals['last'] = np.array([0, 1, 0])
        
        def dist_print(self):
            dists = list(self.dist_vals.keys())
            if self.dist_type is not 'line':
                return('%s_distribution {%s = %f,%s = %f}'%(
                        self.dist_type,
                        dists[0],self.dist_vals[dists[0]],
                        dists[1],self.dist_vals[dists[1]]))
            else:
                return('%s_distribution {'%self.dist_type+\
                '%s = vector'%dists[0]+\
                np.array2string(self.dist_vals[dists[0]],separator = ',').\
                replace('[','(').replace(']',')')+','+\
                '%s = vector'%dists[1]+\
                np.array2string(self.dist_vals[dists[1]],separator = ',').\
                replace('[','(').replace(']',')')+'}')

    def __init__(self, fil='auto_fly_fil.fly2', n=10000, mass=1, charge=-1,
                 ke_type='uniform', az_type='gaussian', el_type='gaussian',
                 pos_type='line'):
        self.n = n
        self.mass = mass
        self.charge = -1
        self.fil = fil

        # distribution defaults

        self.ke = self.dist(ke_type)
        self.ke.dist_vals['min'] = 0
        self.ke.dist_vals['max'] = 1000

        self.az = self.dist(az_type)
        self.az.dist_vals['mean'] = 0
        self.az.dist_vals['fwhm'] = 24
        
        self.el = self.dist(el_type)
        self.el.dist_vals['mean'] = 150
        self.el.dist_vals['fwhm' ] = 19.2

        self.pos = self.dist(pos_type)
        self.pos.dist_vals['first'] = np.array([99.4,133])
        self.pos.dist_vals['last'] = np.array([158.9,116.8])

    def fly_print(self):
        lines = []
        lines.append('particles {')
        lines.append('coordinates = 0,')
        lines.append('standard_beam {')
        lines.append('n = %d,'%self.n)
        lines.append('tob = 0,')
        lines.append('mass = %d,'%self.mass)
        lines.append('charge = %d,'%(self.charge))
        lines.append('ke = %s,'%self.ke.dist_print())
        lines.append('az = %s,'%self.az.dist_print())
        lines.append('el = %s,'%self.el.dist_print())
        lines.append('position = ' +self.pos.dist_print())
        lines.append('}\n}')
        with open(self.fil, 'w') as fil:
             for line in lines: fil.write(line+'\n')


def rand_line(v1,v2,size):
    d = v2 - v1
    dx_rand = np.random.uniform(0,d[0],size)
    m = d[1]/d[0]
    return(np.transpose(np.stack((dx_rand+v1[0],m*dx_rand+v1[1]))))

class pdf:

    def sputtered(x,a = -2.06741184,b = 12.01113288,c =  0.76598885,
        d = 1.94674493):
        a,b,c,d = (-2.12621554, 12.28266708,  0.762046  ,  1.95374093)
        return(a*x - b*(x-c)**2+ d*np.log(x**(-d)))

    def cos(x):
        return(np.cos(x*np.pi-np.pi/2))
    
    func_dict = {
                'sputtered':sputtered,
                'cos':cos
                }

    def __init__(self,f='sputtered'):
        self.f =  self.func_dict[f]

    def __call__(self,x):
        return(self.f(x))

    def sample(self,n,a=0,b=1,log_space = False):
        if log_space == True:
            x = np.logspace(ln(a),ln(b),n)
        else:
            x = np.linspace(a,b,n)
        y = self.f(x)
        select = np.repeat(x,abs(y/(max(y)-min(y))*n).astype(int))
        return(np.random.choice(select,n))  


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
        return((pdf('cos').sample(n)*self.dist_vals['range']\
            -self.dist_vals['range']/2)+self.dist_vals['mean'])

    def __init__(self,dist_type,n=1):

        func_dict = {'gaussian':self.gaussian,
                'uniform':self.uniform,
                'line':self.line,
                'single':self.single,
                'sputtered':self.sputtered,
                'cos':self.cos
                }
        func_defaults  = {'gaussian':{'mean':0,'fwhm':1},
                'uniform':{'min':0,'max':1},
                'line':{'first':np.array([0,0,0]),'last':np.array([1,1,0])},
                'single':{'value':1},
                'sputtered':{'E_beam':105,'a':.01,'b':.65},
                'cos':{'mean':75,'range':180}
                }
        self.defaults = func_defaults
        self.dist_type = dist_type.lower()
        self.f = func_dict[self.dist_type]
        self.dist_vals = func_defaults[self.dist_type]
        self.n = n

    def __call__(self,n = []):
        if n != []:
            self.n = n
        self.dist_out = self.f(self.n)
        return(self.dist_out)

    def __str__(self):
        return(str([self.dist_type,self.dist_vals]))


class auto_parts:
    
    def __init__(self, fil='auto_ion.ion', n=10000):
        # from ..defaults import KE_DIST_TYPE,KE_DIST_VALS,\
        #         AZ_DIST_TYPE,AZ_DIST_VALS,EL_DIST_TYPE,\
        #         EL_DIST_VALS,POS_DIST_TYPE,POS_DIST_VALS,\
        #         MASS,CHARGE

        self.n = n
        self.mass = MASS
        self.charge = CHARGE
        self.fil = fil

        # distribution defaults
        self.ke = source(str(KE_DIST_TYPE),n)
        self.ke.dist_vals = KE_DIST_VALS.copy()

        self.az = source(str(AZ_DIST_TYPE),n)
        self.az.dist_vals = AZ_DIST_VALS.copy()
        
        self.el = source(str(EL_DIST_TYPE),n)
        self.el.dist_vals= EL_DIST_VALS.copy()

        self.pos = source(str(POS_DIST_TYPE),n)
        self.pos.dist_vals = POS_DIST_VALS.copy()

    def __call__(self):
        return({'n':self.n,
                  'mass':self.mass,
                  'charge':self.charge,
                  'ke':str(self.ke),
                  'az':str(self.az),
                  'el':str(self.el),
                  'pos':str(self.pos)})

    def __str__(self):
        return(str({'n':self.n,
                  'mass':self.mass,
                  'charge':self.charge,
                  'ke':str(self.ke),
                  'az':str(self.az),
                  'el':str(self.el),
                  'pos':str(self.pos)}))
    # Ion File output headder
    #time of birth,mass,charge,x0,y0,z0,azimuth, elevation, energy, cfw, color
    def ion_print(self):
        ke = abs(self.ke(n = self.n))
        az = self.az(n = self.n)
        el = self.el(n = self.n)
        pos = self.pos(n = self.n)
        with open(self.fil, 'w') as fil:
            for n in range(len(ke)):
                fil.write("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f \n"%\
                    (0,self.mass,self.charge,pos[n,0],pos[n,1],0,az[n],el[n],ke[n],1,1))

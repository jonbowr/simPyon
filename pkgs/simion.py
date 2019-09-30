import os
import numpy as np
import subprocess
import multiprocessing
import time
import tempfile as tmp
from scipy.optimize import curve_fit as cf
from scipy import stats
import math
from matplotlib import pyplot as plt
from . import gem
from .particles import auto_parts
from .data import sim_data
from ..defaults import *

class simion:
    def __init__(self,volt_dict = {},home = './'):
        self.commands = []
        self.home = home
        self.sim = r'simion.exe --nogui --noprompt --default-num-particles=100000 '
        self.elec_num = []
        self.pa = []
        self.gemfil = []
        self.elect_dict = {}
        self.volt_dict = volt_dict
        self.data = []
        self.parts = auto_parts()
        for file in os.listdir(home):
                if file.endswith(".iob"):
                    self.bench = os.path.join(home,file)
        for root,dirs,files in os.walk(home):
            for file in files:
                if file.endswith(".pa0"):
                    self.pa0 = os.path.join(root,file)
                elif file.endswith(".pa#"):
                    self.pa = os.path.join(root,file).strip('#')
        for file in os.listdir(home):
            if file.lower().endswith(".gem"):
                self.gemfil = os.path.join(home,file)
        self.get_elec_nums_gem()

    def gem2pa(self,pa):
        self.commands = r"gem2pa %s %s#" % (self.gemfil, pa)
        self.run()

    def refine(self):
        self.commands = r"refine %s#" % self.pa
        self.run()

    def volt_adjust(self,voltages, quiet = False):
        fast_adj_str = ''
        for volt, num in zip(voltages, self.elec_num):
            if num != 0:
                fast_adj_str += '%d=%f,' % (num, volt)
        self.commands = "fastadj %s %s" % (self.pa0, fast_adj_str[:-1])
        self.run(quiet = quiet)

    def fast_adjust(self,volt_dict = [],scale_fact = 1,quiet = False):

        if volt_dict == []:
            volt_dict = self.volt_dict
        elif type(volt_dict) == dict: 
                self.volt_dict = volt_dict
        dict_out = dict([(elec,volt_dict[elec]) for elec in volt_dict])
        # print(dict_out)
        volts = [volt_dict[self.elect_dict[val]]*\
        (scale_fact if val < 16 else 1) for val in self.elec_num]
        # self.volt_dict = dict(volt_dict[self.elect_dict[val]]*\
        # (scale_fact if val < 16 else 1) for val in self.elec_num)
        # print(volts)
        if quiet == False:
            for val in self.elec_num:
                print(self.elect_dict[val])
            print(volts)
        for num,volt in zip(self.elec_num,volts):
            dict_out[self.elect_dict[num]] = volt
        self.volt_adjust(volts,quiet = quiet)
        return(self)  

    def fly(self,n_parts = 1000,
        cores = multiprocessing.cpu_count(),surpress_output = False,
        fast_adj = True):
        if fast_adj == True and self.volt_dict!={}:
            self.fast_adjust(quiet = True)
        start_time = time.time()
        checks = []
        fly_fils = []
        self.parts.n = int(n_parts/cores)
        for i in range(int(cores)):
            fly_fil = self.home+'auto_fly_%i.ion'%i
            fly_fils.append(fly_fil)
            self.parts.fil = fly_fil
            self.parts.ion_print()

        for ion_fil in fly_fils:
            loc_com = r"fly  "+\
            r" --retain-trajectories=0 --restore-potentials=0"
            loc_com += r" --particles=" + ion_fil
            loc_com += r" %s"%self.bench
            self.commands = loc_com
            f = tmp.TemporaryFile()
            check = subprocess.Popen(self.sim+' '+self.commands,
                stdout = f)
            checks.append((check,f))

        outs = []
        for check,f in checks:
            check.wait()

        for check,f in checks:
            f.seek(0)
            outs.append(f.read())
            check.kill()

        for nam in fly_fils:
            if os.path.isfile(nam)==True:
                os.remove(nam)

        tot_lines = []
        j=0
        b = 0
        for out in outs:
            st_out = str(out).split('\\r\\n')
            out_line = []
            start = False
            for line in st_out:
                if start == True:
                    if line[0].isdigit(): 
                        out_line.append(np.fromstring(line,sep = ',')) 
                elif j == 0 and surpress_output == False:print(line)
                if "------ Begin Next Fly'm ------" in line:
                    start = True
            if start == False:
                print('============= Fly Failed on Core %d ============='%j)
                for line in st_out: print(line)
                print('============= Fly Failed on Core %d ============='%j)
                return()
            for i in range(len(out_line)):
                out_line[i][0] += int(n_parts/cores)*j
            tot_lines+=out_line
            j+=1
        if surpress_output == False:
            print(time.time() - start_time)
        data = np.stack(tot_lines)
        self.data = sim_data(data)
        return(self)

    def old_fly(self,outfile, bench, particles = [],remove = True):
        loc_com = r"fly --recording-output="+outfile+\
        r" --retain-trajectories=0 --restore-potentials=0"
        if particles != []:
            loc_com += r" --particles=" + particles
        loc_com += r" %s"%bench
            
        if remove == True:
            if os.path.isfile(outfile)==True:
                os.remove(outfile)
        self.commands = loc_com
        start_time = time.time()
        self.run()
        print(time.time()-start_time)

    def get_elec_nums_gem(self, gem_fil=[]):
        if gem_fil == []:
            gem_fil = self.gemfil
        lines = open(gem_fil).readlines()
        for line in lines:
            if line != '':
                if line.lower()[:line.find(';')].find('electrode') != -1:
                    num = int(line[line.find('(') + 1:line.find(')')])
                    if num not in self.elec_num:
                        self.elec_num += [num]
                        self.elect_dict[num] = line[line.find(
                            ';'):].strip(';').strip()

    def electrode_nums(self, nums):
        self.elec_num = nums

    def run(self,quiet = False):
        if quiet == False:
            print(self.sim)
            print(self.commands)
        if quiet == True:
            check = subprocess.Popen(self.sim + ' ' + self.commands,
                stdout = subprocess.PIPE)
        else:
            check = subprocess.Popen(self.sim + ' ' + self.commands)
        check.wait()
        check.kill()
        return check
    
    def show(self,measure = False,annotate = False):
        fig,ax1 = gem.gem_draw_poly(self.gemfil,measure = measure,
                annotate = annotate,elec_names = self.elect_dict)
        ax1.set_ylabel('$r=\sqrt{z^2 + y^2}$ [mm]')
        ax1.set_xlabel('x [mm]')
        if self.data != []:
            ax1.plot(self.data()['x'],self.data()['r'],'.')
            all_h,all_x,all_y = np.histogram2d(self.data.stop()()['x'],
                self.data.stop()()['r'], bins =int(400),
                weights =self.data.stop()()['counts'])
            cs = plt.contour((all_x[1:]+all_x[:-1])/2,
                (all_y[1:]+all_y[:-1])/2,all_h.T)

            fig,ax2 = plt.subplots(1)
            h,xbins,ybins = np.histogram2d(self.data.good().stop()()['z'],
                self.data.good().stop()()['y'], bins =int(30/10000*len(self.data)/2),
                weights =self.data.good().stop()()['counts'])
            if TOF_MEASURE == True:
                ax2.scatter(self.data.good_tof().stop()()['z'],
                self.data.good_tof().stop()()['y'],color = 'red')
            ax2.scatter(self.data.good().stop()()['z'],
            self.data.good().stop()()['y'], color = 'blue')
            ax2.set_xlabel('z [mm]')
            ax2.set_ylabel('y [mm]')
            cs = plt.contour((xbins[1:]+xbins[:-1])/2,
                (ybins[1:]+ybins[:-1])/2,h.T)
            ax2.set_aspect('equal')

            def circ(r,x_vec):
                return(np.sqrt(r**2 - x_vec**2))
            
            r_max = 45.1
            r_min = 35.4

            for r in [r_min,r_max]:
                x_min =circ(r,min(ybins)) 
                x = np.linspace(-x_min,x_min,100)
                ax2.plot(x,circ(r,x), color = 'r')

    def fly_steps(self,n_parts = 10000,volt_base_dict={},
        e_steps = np.arange(1,8),e_max = 1000,
        volt_scale_factors = {1:.034735,
                      2:81.2/1212,
                      3:156/1212,
                      4:307/1212,
                      5:592/1212,
                      6:1,
                      7:1.93775}):
        data = []
        # upper_eng = np.copy(self.parts.ke.dist_vals['max'])
        for step in e_steps:
            if volt_base_dict != {}:
                self.fast_adjust(volt_base_dict,
                        scale_fact = volt_scale_factors[step])
            else:
                self.fast_adjust(scale_fact = volt_scale_factors[step])
            self.parts.ke.dist_vals['max'] = \
                    e_max*volt_scale_factors[step]
            data.append(self.fly(n_parts = n_parts,fast_adj = False).data)
        return(data)

    def define_volts(self, save = False):
        volts = {}
        for elect in self.elect_dict.values():
            volts[elect] = float(input(elect+': '))
        if save == True:
            np.save(input('File Name to Save to: '),volts)
        self.volt_dict = volts
        return(volts)

    def get_master_volts(self,volt_dict = []):
        if volt_dict ==[]:
            volt_dict = self.volt_dict
        self.master_volts = {}
        for elec_num in self.elect_dict:
            self.master_volts[elec_num] = volt_dict[self.elect_dict[elec_num]]
        for elec_name in volt_dict:
            self.master_volts[elec_name] = volt_dict[elec_name]
        return(self.master_volts)

    def scale_volts(self,volt_dict,scale_fact):
        m_volts = self.get_master_volts(volt_dict)
        s_volts = {}
        for num,nam in self.elect_dict.items():
            s_volts[nam]=m_volts[num]*(scale_fact if num < 16 else 1)
        return(s_volts)

    # def check_volts()
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
from shutil import copy

class simion:
    '''
    simion ion simulation environment controler. Utilizes .GEM geometry files to visualize 
    electrode geometry, generate and refine potential arrays. Allows for voltage fast 
    adjustment, and particle flying. particle fly_em is parallelized by spawning multiple 
    simion instances, particle data is output to buffer and imported to numpy array and assigned
    simPyon.data.simdata data structure rather than saved to disk. 

    performing multiple fly commands overwrite previous data in buffer. 

    Parameters
    ----------
    home : string
        environment location on disk, automatically set to current directory
    volt_dict : dict
        dict of floats: {'electrode_name':VOLTAGE} relating the electrode names to desired voltages

    Attributes
    ----------
    commands : list of strings 
        multiple simion commands for cmd to execute
    home : string
        environment location on disk, automatically set to current directory
    sim : string
        baseline simion execution command, disables simion GUI, surpresses user lua input 
        (automatically says yes when prompted), and increases the number of particles simion 
        allows to import from .ion files (from 1000 to 10000)
    elec_num : list
        order of electrode numbers as they appear in .GEM
    pa : string
        .pa0 file used to fast adjust. simion.__init__ automatically grabs .pa0 file in current directory
    gemfil : string
        .GEM file defining geometry. simion.__init__ automatically grabs .pa0 file in current directory
    elect_dict : dict
        dict of strings: {ELEC_NUM:'electrode_name'} relating the electrode number to the electrode name. 
        Electrode name taken from commented string following the electrode declaration in gemfile. 
            Syntax in .GEM: electrode(1); name_of_electrode_1  
    volt_dict : dict
        dict of floats: {'electrode_name':VOLTAGE} relating the electrode names to desired voltages
    data : simPyon.data.sim_data object
        ion data output from fly command in simPyon.data.sim_data data structure
    parts : simPyon.particles.sim_parts object
        distribution of source particles 
    '''
    def __init__(self,volt_dict = {},home = './',gemfil = '',recfil = ''):
        self.commands = []
        self.home = home
        self.sim = r'simion.exe --nogui --noprompt --default-num-particles=1000000'
        self.elec_num = []
        self.pa = []
        self.gemfil = []
        self.elect_dict = {}
        self.volt_dict = volt_dict
        self.data = []
        self.parts = auto_parts()

        # copy rec file to home directory if none already exists
        if recfil == '':
            self.recfil = self.home+'simPyon_base.rec'
            copy("%s/rec/simPyon_base.rec"%\
                            os.path.dirname(os.path.dirname(__file__)+'..'),
                            self.recfil)

        # Grab workbench, potential arrays and gemfiles from current directory
        for file in os.listdir(home):
                if file.endswith(".iob"):
                    self.bench = os.path.join(home,file)
        for root,dirs,files in os.walk(home):
            for file in files:
                if file.endswith(".pa0"):
                    self.pa0 = os.path.join(root,file)
                elif file.endswith(".pa#"):
                    self.pa = os.path.join(root,file).strip('#')
        if gemfil =='':
            for file in os.listdir(home):
                if file.lower().endswith(".gem"):
                    self.gemfil = os.path.join(home,file)
        else:
            self.gemfil = gemfil

        #scrape the gemfile for numbers
        self.get_elec_nums_gem()


    def gem2pa(self,pa):
        '''
        Converts .GEM gemfile defined in self.gemfil to potential array (.pa#) with
        file name pa. Assigns pa to self.pa

        Parameters
        ----------
        pa: string
            name of newly generated .pa# potential array file
        '''
        self.commands = r"gem2pa %s %s#" % (self.gemfil, pa)
        self.pa = pa
        self.run()

    def refine(self):
        '''
        Refines potential array self.pa from .pa# to simions potential array structure

        '''
        self.commands = r"refine %s#" % self.pa
        self.run()

    def __volt_adjust__(self,voltages, quiet = False):
        '''
        executes simion command to actually execute fast adjust

        Parameters
        ----------
        voltages: array
            array of voltages connecting desired electrode voltages to electrode number through
            position in array
        quiet: bool
            surpresses lua output to cmd if quiet == True

        ** Needs to be integrated with self.fast_adjust 
        *** user should use self.fast_adjust to perform fast adjust *** 
        '''
        fast_adj_str = ''
        for volt, num in zip(voltages, self.elec_num):
            if num != 0:
                fast_adj_str += '%d=%f,' % (num, volt)
        self.commands = "fastadj %s %s" % (self.pa0, fast_adj_str[:-1])
        self.run(quiet = quiet)

    def fast_adjust(self,volt_dict = [],scale_fact = 1,
                    quiet = False,keep = False):
        '''
        Perform simion fast adjust of potential array .pa0. 

        Allows for relative scaling of all voltages.

        Parameters
        ----------
        volt_dict: dict
            optional input of voltage dict to be used in fast adjust. If no new voltage 
            is provided fast adjust is performed using voltages in self.volt_dict
        scale_fact: float
            optional value to scale all of the electrode voltages to excludes electrodes 
            with electrode number >=16 from fast adjust. 

            eg: FAST_ADJUST_VOLTAGE = self.volt_dict['electrode_name']*scale_fact
        quiet: bool
            surpresses lua output to cmd if quiet == True
        keep: bool
            saves new voltages to self. elec_dict if keep == True
        '''

        if volt_dict == []:
            volt_dict = self.volt_dict
        elif type(volt_dict) == dict: 
                self.volt_dict = volt_dict
        dict_out = dict([(elec,volt_dict[elec]) for elec in volt_dict])
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
        self.__volt_adjust__(volts,quiet = quiet)
        if keep:
            self.volt_dict = dict_out
        return(self)  

    def fly(self,n_parts = 1000,cores = multiprocessing.cpu_count(),
            surpress_output = False):
        '''
        Fly n_parts particles using the particle probability distributions defined in self.parts. 
        Parallelizes the fly processes by spawing a number of instances associated with the
        number of cores of the processing computer. Resulting particle data is stored in 
        self.data as a simPyon.data.sim_data object. 

        Parameters
        ----------
        n_parts: int
            number of particles to be flown. n_parts/cores number of particles is flown in each 
            instance of simion on each core. 
        cores: int
            number of cores to use to process the fly particles request. Default initializes 
            a simion instance to run on each core. 
        surpress_output: bool
            With surpress_output == True, the preparation statement from one of the simion 
            instances is printed to cmd. 
        '''

        start_time = time.time()


        # Fly the particles in parallel and scrape the resulting data from the shell
        outs = core_fly(self,n_parts,cores,surpress_output)
        data = str_data_scrape(outs,n_parts,cores,surpress_output)
        self.data = sim_data(data)

        if surpress_output == False:
            print(time.time() - start_time)

        return(self)

    def old_fly(self,outfile, bench, particles = [],remove = True):
        '''
        Original version of the fly command that uses simion declared particle 
        distributions to fly
        '''
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

    def particle_traj(self,n_parts = 1000,cores = multiprocessing.cpu_count(),
                      surpress_output = False, dat_step = 30,show= True):
        '''
        Fly n_parts particles using the particle probability distributions defined in self.parts. 
        Parallelizes the fly processes by spawing a number of instances associated with the
        number of cores of the processing computer. Resulting particle data is stored in 
        self.data as a simPyon.data.sim_data object. 

        Parameters
        ----------
        n_parts: int
            number of particles to be flown. n_parts/cores number of particles is flown in each 
            instance of simion on each core. 
        cores: int
            number of cores to use to process the fly particles request. Default initializes 
            a simion instance to run on each core. 
        surpress_output: bool
            With surpress_output == True, the preparation statement from one of the simion 
            instances is printed to cmd. 
        '''

        # Copy Traj record file to working home directory
        if os.path.isfile(self.home+'simPyon_traj.rec')==False:
            copy("%s/rec/simPyon_traj.rec"%\
                            os.path.dirname(os.path.dirname(__file__)+'..'),
                            self.home)

        # Fly the particles in parallel and scrape the resulting data from the shell
        outs = core_fly(self,n_parts,cores,surpress_output)
        data = str_data_scrape(outs,n_parts,cores,surpress_output)
        
        if surpress_output == False:
            print(time.time() - start_time)
        
        # Parse the trajectory data into list of dictionaries
        head = ["Ion N","X","Y","Z","KE"]
        self.traj_data = []
        for n in np.unique(data[:,0]):
            self.traj_data.append(dict([h.lower(),arr[data[:,0]==n]] for h,arr in zip(head,np.transpose(data))))

        # calculate the r position vec
        for dat in self.traj_data:
            dat['r'] = np.sqrt(dat['z']**2+dat['y']**2)

        # Plot the trajectories
        if show == True:
            fig,ax = self.show()
            for traj in self.traj_data:
                ax.plot(traj['x'],traj['r'])
        return(self)

    def get_elec_nums_gem(self, gem_fil=[]):
        '''
        Get the electrode numbers and names from a gemfile and store them in 
        elec_nums and elec_dict. 
        
        Parameters
        ----------
        gem_fil: string
            gemfile to take the names and values from. Default uses the GEM file stored
            in self.gemfil
        '''

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

    def run(self,quiet = False):
        '''
        Executes all of the initialized commands in self.commands
        '''
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
    
    def show(self,measure = False,mark=False,
             annotate = False, origin = [0,0],collision_locs = False):
        '''
        Plots the geometry stored geometry file by scraping the gemfile for 
        polygon shape and renders them in pyplot using a collection of patches. 

        Parameters
        ----------
        measure: bool
            With measure == True, displays a dragable ruler, defined by the fig_measure.measure 
            function, overlaid on the plotted geometry. 
        mark: bool
            With mark == True, adds draggable points displaying points the symmetry plane using
            the fig_measure.mark function. 
        annotate: bool 
            With annotate == True, names or numbers of each of the electrodes are overlaid
            on the plotted geometry using the figmeasure.annote function
        origin: [0x2] list
            Point in the simmetry plane to shift the origin to for the displayed geometry. 

        '''
        fig,ax1 = gem.gem_draw_poly(self.gemfil,
                                    measure = measure,
                                    mark=mark,
                                    annotate = annotate,
                                    elec_names = self.elect_dict,
                                    origin = origin)
        ax1.set_ylabel('$r=\sqrt{z^2 + y^2}$ [mm]')
        ax1.set_xlabel('x [mm]')
        if self.data != [] and collision_locs ==True:

            
            from scipy.interpolate import interpn

            def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
                """
                Scatter plot colored by 2d histogram
                """
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

            # density_scatter(self.data.stop()['x'],self.data.stop()['r'],ax1,bins = 100)
            ax1.plot(self.data.stop()['x'],self.data.stop()['r'],'.')
            ax1.plot(self.data.good().stop()['x'],self.data.good().stop()['r'],'.',color = 'blue')
            # for n in range(len(self.data.rf)):
            #     ax1.plot([self.data.stop().good()['x'][n],128.2],
            #     [self.data.stop().good()['r'][n],self.data.rf[n]])
            # ax1.plot(self.data().stop()['x'],self.data().stop()()['r'],'.')
            # Calculate the point density
            # xy = np.vstack([self.data.good().start()()['x'],self.data.good().start()()['r']])
            # z = gaussian_kde(xy)(xy)
            # ax1.scatter(self.data.good().start()()['x'], self.data.good().start()()['r'],
            #             c=z, s=100, edgecolor='')

            # from scipy.stats import gaussian_kde
            # xy = np.vstack([self.data.stop()['x'],self.data.stop()['r']])
            # z = gaussian_kde(xy)(xy)
            # ax1.scatter(self.data.stop()['x'], self.data.stop()['r'],
            #             c=z, s=100, edgecolor='')


            # all_h,all_x,all_y = np.histogram2d(self.data.stop()()['x'],
            #     self.data.stop()()['r'], bins =int(400),
            #     weights =self.data.stop()()['counts'])
            # cs = plt.contour((all_x[1:]+all_x[:-1])/2,
            #     (all_y[1:]+all_y[:-1])/2,all_h.T)

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
                ax2.plot(x,circ(r,x), color = 'black')
        return(fig,ax1)

    def fly_steps(self,voltage_scale_factors,n_parts = 10000,e_max = 1000,
                  volt_base_dict={}):
        '''
        Fly particles for a set of voltages generated by scaling stored voltages 
        to the assigned voltage_scale_factors. Output for each run is stored in data
        as a list of simPyon.data.sim_data objects. 

        Currently only designed to work with flat energy distributions

        Parameters
        ----------
        voltage_scale_factors: array or list
            List of values to use in the voltage scaling. The default voltages for 
            each of the electrodes are multiplied by these values before flying
        n_parts: int
            number of particles to be flown per fly
        e_max: float
            energy values also mutiplied by scale factor to shift distribution in 
            energy while scaling. 
        volt_base_dict: dict
            volt_dict assigning the base voltage setting prior to scaling. 
        e_steps: 
        '''
        data = []
        # upper_eng = np.copy(self.parts.ke.dist_vals['max'])
        # volt_scale_factors = {1:.034735,
        #               2:81.2/1212,
        #               3:156/1212,
        #               4:307/1212,
        #               5:592/1212,
        #               6:1,
        #               7:1.93775}
        for scale in voltage_scale_factors:
            if volt_base_dict != {}:
                self.fast_adjust(volt_base_dict,
                        scale_fact = scale)
            else:
                self.fast_adjust(scale_fact = scale)
            self.parts.ke.dist_vals['max'] = \
                    e_max*scale
            data.append(self.fly(n_parts = n_parts).data)
        return(data)

    def define_volts(self, save = False):
        '''
        Prompts user input for electrode voltages

        Parameters
        ----------
        save: bool
            if save == true electrodes are saved to npz file to filename 
            prompted after volages are input. 
        '''
        volts = {}
        for elect in self.elect_dict.values():
            volts[elect] = float(input(elect+': '))
        if save == True:
            np.save(input('File Name to Save to: '),volts)
        self.volt_dict = volts
        return(volts)

    def get_master_volts(self,volt_dict = []):
        '''
        Generates master volts dict that links electrode name and number to the 
        same voltage value. 
        '''
        if volt_dict ==[]:
            volt_dict = self.volt_dict
        self.master_volts = {}
        for elec_num in self.elect_dict:
            self.master_volts[elec_num] = volt_dict[self.elect_dict[elec_num]]
        for elec_name in volt_dict:
            self.master_volts[elec_name] = volt_dict[elec_name]
        return(self.master_volts)

    def scale_volts(self,volt_dict,scale_fact):
        '''
        Scales voltages in volt_dict by float value in scale_fact
        '''
        m_volts = self.get_master_volts(volt_dict)
        s_volts = {}
        for num,nam in self.elect_dict.items():
            s_volts[nam]=m_volts[num]*(scale_fact if num < 16 else 1)
        return(s_volts)

def str_data_scrape(outs,n_parts,cores,surpress_output):
    tot_lines = []
    j=0
    b = 0
    for out in outs:
        st_out = str(out).split('\\r\\n')
        out_line = []
        start = False
        for line in st_out:
            try:
                if start == True:
                    if line[0].isdigit(): 
                        out_line.append(np.fromstring(line,sep = ',')) 
                elif j == 0 and surpress_output == False:print(line)
            except(IndexError):
                pass
            if "------ Begin Next Fly'm ------" in line:
                start = True
            # elif "Ion N" in line:
            #     head = line
        if start == False:
            print('============= Fly Failed on Core %d ============='%j)
            for line in st_out: print(line)
            print('============= Fly Failed on Core %d ============='%j)
            return()
        for i in range(len(out_line)):
            out_line[i][0] += int(n_parts/cores)*j
        tot_lines+=out_line
        j+=1
    data = np.stack(tot_lines)
    return(data)

def core_fly(sim,n_parts,cores,surpress_output):
    checks = []
    fly_fils = []
    sim.parts.n = int(n_parts/cores)
    for i in range(int(cores)):
        fly_fil = sim.home+'auto_fly_%i.ion'%i
        fly_fils.append(fly_fil)
        sim.parts.fil = fly_fil
        sim.parts.ion_print()

    for ion_fil in fly_fils:
        loc_com = r"fly  "
        loc_com+=r" --retain-trajectories=0 --restore-potentials=0"
        loc_com+=r" --recording=%s"%sim.recfil
        loc_com += r" --particles=" + ion_fil
        loc_com += r" %s"%sim.bench
        sim.commands = loc_com
        f = tmp.TemporaryFile()
        check = subprocess.Popen(sim.sim+' '+sim.commands,
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
    return(outs)
            
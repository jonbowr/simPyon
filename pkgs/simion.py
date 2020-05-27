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
    def __init__(self,volt_dict = {},home = './',
                 gemfil = '',recfil = '',
                 traj_recfil = '',
                 bench = ''):
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
        self.traj_refil = traj_recfil
        self.trajectory_quality = 3
        # copy rec file to home directory if none already exists
        if recfil == '':
            self.recfil = self.home+'simPyon_base.rec'
            copy("%s/rec/simPyon_base.rec"%\
                            os.path.dirname(os.path.dirname(__file__)+'..'),
                            self.recfil)

        # Grab workbench, potential arrays and gemfiles from current directory
        if bench == '':
            for file in os.listdir(home):
                if file.endswith(".iob"):
                    self.bench = os.path.join(home,file)
        else:
            self.bench = os.path.join(home,bench)

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
            self.gemfil = home + gemfil

        self.name = gemfil.upper().strip('.GEM')
        #scrape the gemfile for numbers
        self.get_elec_nums_gem()
        self.pa_info = gem.get_pa_info(self.gemfil)


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
        volts = [(volt_dict[self.elect_dict[val]] if self.elect_dict[val] in volt_dict else 0)*\
        (scale_fact if val < 16 else 1) for val in self.elec_num]

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
            quiet = False):
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
        quiet: bool
            With quiet == True, the preparation statement from one of the simion 
            instances is printed to cmd. 
        '''

        start_time = time.time()


        # Fly the particles in parallel and scrape the resulting data from the shell
        outs = core_fly(self,n_parts,cores,quiet,
                        trajectory_quality =self.trajectory_quality)
        data = str_data_scrape(outs,n_parts,cores,surpress_output)
        self.data = sim_data(data,symmetry = self.pa_info['symmetry'],
                                    mirroring = self.pa_info['mirroring'])

        if quiet == False:
            print(time.time() - start_time)

        return(self)


    def fly_trajectory(self,n_parts = 100,cores = multiprocessing.cpu_count(),
                      surpress_output = False, dat_step = 30,show= True,geo_3d = False):
        '''
        Fly n_parts particles, and plot their trajectories. Uses the particle probability 
        distributions defined in self.parts, but tracks the particle movement. 
        Parallelizes the fly processes by spawing a number of instances associated with the
        number of cores of the processing computer. Resulting particle data is stored in 
        self.traj_data as a list of dictionaries. 

        Parameters
        ----------
        n_parts: int
            number of particles to be flown. n_parts/cores number of particles is flown in each 
            instance of simion on each core. 
        cores: int
            number of cores to use to process the fly particles request. Default initializes 
            a simion instance to run on each core. 
        quiet: bool
            With quiet == True, the preparation statement from one of the simion 
            instances is printed to cmd. 
        '''

        # Copy Traj record file to working home directory
        # if os.path.isfile(self.home+'simPyon_traj.rec')==False:

        if self.traj_refil == '':
            copy("%s/rec/simPyon_traj.rec"%\
                            os.path.dirname(os.path.dirname(__file__)+'..'),
                            self.home)
            # os.remove(self.home+'simPyon_traj.rec')

        start_time = time.time()
        # Fly the particles in parallel and scrape the resulting data from the shell
        outs = core_fly(self,n_parts,cores,quiet,
                        rec_fil = self.home + 'simPyon_traj.rec',
                        markers = 20,trajectory_quality = 0)
        data = str_data_scrape(outs,n_parts,cores,quiet)
        
        if quiet == False:
            print(time.time() - start_time)
        
        # Parse the trajectory data into list of dictionaries
        head = ["X","Y","Z","V","Grad V","ke"]
        self.traj_data = []
        for n in np.unique(data[:,0]):
            self.traj_data.append(dict([h.lower(),arr[data[:,0]==n]] for h,arr in zip(head,np.transpose(data[:,1:]))))

        # calculate the r position vec
        for dat in self.traj_data:
            dat['r'] = np.sqrt(dat['z']**2+dat[self.pa_info['mirroring']]**2)

        # Plot the trajectories
        if show == True:
            if geo_3d == False:
                fig,ax = self.show()
                for traj in self.traj_data:
                    ax.plot(traj[self.pa_info['base']],traj['r'])
            if geo_3d == True:
                from mpl_toolkits.mplot3d import Axes3D,art3d
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                elec_verts,exclude_verts = gem.get_verts(self.gemfil)
                for el in elec_verts.values():
                    for vt in el:
                        ax.plot(vt[:,0],np.zeros(len(vt)),vt[:,1],
                                color = 'grey')
                        ax.plot(vt[:,0],np.zeros(len(vt)),-vt[:,1],
                                color = 'grey')
                        ax.plot(vt[:,0],-vt[:,1],np.zeros(len(vt)),
                                color = 'grey')
                        
                for traj in self.traj_data:
                    ax.plot(traj['x'],-traj['z'],traj['y'])
                ax.view_init(30, -70)

                # ax.set_aspect('equal')
                return(fig,ax)
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
    
    def show(self,measure = False,annotate = False, origin = [0,0],
             collision_locs = False,fig = [],ax = []):
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
        from .poly_gem import draw
        fig,ax1 = draw(self.gemfil,canvas = [self.pa_info['Lx'],
                                    self.pa_info['Ly']],
                                    fig = fig, ax = ax,
                                    mirror_ax = self.pa_info['mirroring'],
                                    origin = origin)
        if measure == True:
            from . import fig_measure as meat
            fig,ax1 = meat.measure_buttons(fig,ax1)

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
            ax1.plot(self.data.stop()['x'],self.data.stop()['r'],'.',color = 'r')
            ax1.plot(self.data.good().stop()['x'],self.data.good().stop()['r'],'.',color = 'blue')

            fig,ax2 = plt.subplots(1)
            h,xbins,ybins = np.histogram2d(self.data.good().stop()()['z'],
                self.data.good().stop()()['y'], bins =int(30/10000*len(self.data)/2),
                weights =self.data.good().stop()()['counts'])

            ax2.scatter(self.data.good().stop()()['z'],
            self.data.good().stop()()['y'], color = 'blue')
            ax2.set_xlabel('z [mm]')
            ax2.set_ylabel('y [mm]')
            cs = plt.contour((xbins[1:]+xbins[:-1])/2,
                (ybins[1:]+ybins[:-1])/2,h.T)
            ax2.set_aspect('equal')

            def circ(r,x_vec):
                return(np.sqrt(r**2 - x_vec**2))

            for r in [self.data.obs['R_MIN'],self.data.obs['R_MAX']]:
                x_min =circ(r,min(ybins)) 
                x = np.linspace(-x_min,x_min,100)
                ax2.plot(x,circ(r,x), color = 'black')
        return(fig,ax1)

    def fly_steps(self,voltage_scale_factors,n_parts = 10000,particle_scale = '',part_value = 1000,
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
        for scale in voltage_scale_factors:
            if volt_base_dict != {}:
                self.fast_adjust(volt_base_dict,
                        scale_fact = scale)
            else:
                self.fast_adjust(scale_fact = scale)
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

def str_data_scrape(outs,n_parts,cores,quiet):
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
                elif j == 0 and quiet == False:print(line)
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

def core_fly(sim,n_parts,cores,quiet,rec_fil = '',markers = 0,trajectory_quality = 3):
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
        loc_com+=r" --trajectory-quality=%d"%trajectory_quality
        if markers !=0:
            loc_com+=r" --markers=%d"%markers
        loc_com+=r" --recording=%s"%(sim.recfil if rec_fil == '' else rec_fil)
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
            

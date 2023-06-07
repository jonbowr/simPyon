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
from .geo import geo
from .particles import auto_parts,source
from .data import sim_data
from ..defaults import *
from shutil import copy
from matplotlib import cm

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
    def __init__(self,volt_dict = {},
                 home = './',
                 gemfil = '',recfil = '',
                 traj_recfil = '',
                 bench = '',
                 pa = '',
                 obs_region = {'X_MAX':X_MAX,'X_MIN':X_MIN,
                        'R_MAX':R_MAX,'R_MIN':R_MIN,
                        'TOF_MEASURE':TOF_MEASURE}
                 ):
        self.commands = []
        self.home = home
        self.sim = r'simion.exe --nogui --noprompt --default-num-particles=1000000'
        self.elec_num = []
        self.pa = []
        self.usr_prgm = 'simion.workbench_program() \n function segment.other_actions() \n end'
        self.elect_dict = {}
        self.volt_dict = volt_dict
        self.recfil = recfil
        self.data = []
        self.v_data = []
        self.source = auto_parts()
        self.traj_refil = traj_recfil
        self.trajectory_quality = 3
        self.scale_exclude = []
        self.obs_region = obs_region

        if gemfil =='':
            self.gemfil = []
            for root,dirs,files in os.walk(home):
                for file in files:
                    if file.lower().endswith(".gem"):
                        self.gemfil.append(os.path.join(root,file))
        elif  type(gemfil) == list: 
            self.gemfil = []
            for gm in gemfil:
                self.gemfil.append(os.path.join(home,gm))
        elif type(gemfil) == str:
            self.gemfil = [os.path.join(home,gemfil)]


        # Grab workbench, potential arrays and gemfiles from current directory
        # self.bench = ''
        if bench == '':
            self.bench = os.path.join(home,'simPyon_bench.iob')
            copy("%s/bench/simPyon_bench_%d.iob"%(\
                            os.path.dirname(os.path.dirname(__file__)+'..'),
                            len(self.gemfil)),
                            self.bench)
        else:
            Warning("SimPyon updated to work best with default workbench file")
            self.bench = os.path.join(home,bench)

        self.pa = [os.path.join(home,'simPyon%d.pa'%pan) for pan in range(len(self.gemfil))]
        # if not pa:
        #     self.pa = []
        #     for root,dirs,files in os.walk(home):
        #         for file in files:
        #             if file.endswith(".pa0"):
        #                 self.pa.append(os.path.join(root,file).strip('0'))
        # elif type(pa)== str:
        #     self.pa = [os.path.join(home,pa)]
        # elif type(pa) == list:
        #     self.pa = [os.path.join(home,p) for p in pa]

        self.name = self.gemfil[0].upper().strip('.GEM')
        #scrape the gemfile for numbers
        self.pa_info = []
        self.gem_nums = []
        for gm in self.gemfil:
            self.get_elec_nums_gem(gm)
            self.gem_nums.append(gem.get_elec_nums_gem(gm)[0])
            self.pa_info.append(gem.get_pa_info(gm))

        pa = 1
        for pai in self.pa_info:
            if 'pa_offset_position' in pai:
                self.usr_prgm+='\nfunction segment.initialize_run() \n'
                for d,v in zip(['x','y','z'],pai['pa_offset_position']):
                    self.usr_prgm+='simion.wb.instances[%d].%s = %f\n'%(pa,d,v)
                pa +=1
                self.usr_prgm+='end \n'
                
        self.geo = geo(self.gemfil)


    def __repr__(self):
        return('%s \n'%str(type(self))+
                'Workbench:%s \n'%self.bench +
                'Gemfile: %s \n'%self.gemfil+
                'Pa: %s \n'%self.pa)
        

    def gem2pa(self,gemfil = [], pa = None,pa_tag = '#'):
        '''
        Converts .GEM gemfile defined in self.gemfil to potential array (.pa#) with
        file name pa. Assigns pa to self.pa

        Parameters
        ----------
        pa: string
            name of newly generated .pa# potential array file
        '''
        if not gemfil:
            gemfil = self.gemfil
        elif type(gemfil) == str:
            gemfil = [gemfil]

        if not pa:
            pa = self.pa

        if pa == 'split':
            pa = []
            for i in range(len(gemfil)):
                if i >0:
                    pa.append(self.pa[0].replace('.pa','_%d.pa'%i))
                else:
                    pa.append(self.pa[0])
        elif type(pa) != list:
            pa = [pa]

        self.pa = pa
        for gm,pm in zip(gemfil,pa):
            self.commands = r"gem2pa %s %s%s" % (gm, pm,pa_tag)
            self.pa = pa
            self.run()

    def refine(self,pa = []):
        '''
        Refines potential array self.pa from .pa# to simions potential array structure

        '''
        for pa in (pa if pa else self.pa):
            self.commands = r"refine %s#" %pa
            self.run()

    def fast_refine(self,pa = []):
        m_gems = []
        for gm in self.gemfil:
            temp_gem = gm.replace('.'+gm.split('.')[-1],'_temp.'+gm.split('.')[-1])
            print(temp_gem)
            with open(temp_gem,'w') as w:
                for l in open(gm,'r').readlines():
                    lc = l.split(';')[0]
                    if 'electrode' in lc:
                        for elec in self.elec_num:
                            if '(%d)'%elec in lc.replace(' ',''):
                                w.write(l.replace('(%d)'%elec,'(%.2f)'%self.volt_dict[self.elect_dict[elec]]))
                    else:
                        w.write(l)
            m_gems.append(temp_gem)
        self.gem2pa(m_gems,pa_tag = '')
        for g in m_gems: os.remove(g)
        for pa in (pa if pa else self.pa):
            self.commands = r"refine %s" %pa
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
        for pa,pa_nums in zip(self.pa,self.gem_nums):
            fast_adj_str = ''
            for num in pa_nums:
                if num != 0:
                    fast_adj_str += '%d=%f,' % (num, voltages[num])


            self.commands = "fastadj %s %s" % (pa+'0', fast_adj_str[:-1])
            self.run(quiet = False)

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
            print(' ===============================================')
            print('| Fast Adjusting with Voltage Settings:')
            print( '===============================================')
            # print(self.volt_dict)
            for elec_n,volt in zip(self.elec_num,volts):
                print('| %s: %.2f '%(self.elect_dict[elec_n],volt))
            # for val in self.elec_num:
            #     print(self.elect_dict[val])
            # print(volts)
        num_volts = {}
        for num,volt in zip(self.elec_num,volts):
            dict_out[self.elect_dict[num]] = volt
            num_volts[num] = volt
        self.__volt_adjust__(num_volts,quiet = quiet)
        if keep:
            self.volt_dict = dict_out
        return(self)  

    def fly(self,parts = 1000,cores = multiprocessing.cpu_count(),
            quiet = True):
        '''
        Fly n_parts particles using the particle probability distributions defined in self.source. 
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


        # copy rec file to home directory if none already exists
        if self.recfil == '':
            self.recfil = os.path.join(self.home,'simPyon_base.rec')
            copy("%s/rec/simPyon_base.rec"%\
                            os.path.dirname(os.path.dirname(__file__)+'..'),
                            self.recfil)

        # Write the workbench program in 'usr_prgm'
        if self.bench:
            with open(self.bench.replace('iob','lua'),'w') as fil:
                fil.write(self.usr_prgm)

        # Parse particle input type
        if type(parts) == int:
            n_parts = parts
        elif type(parts) == auto_parts:
            self.source = auto_parts()
            self.source.df = parts.df.copy()
            n_parts = self.source['n']
            print('Flying Distribution:\n%s'%str(parts))
        else:
            self.source.splat_to_source(parts)
            n_parts = self.source['n']
            print('Flying vector:\n%s'%str(parts))

        start_time = time.time()

        if quiet == False:
            print(' ===============================================')
            print('| Begining Next Fly\'em:')
            print('| %d Particles on %d Cores'%(n_parts,cores))
            print(' ===============================================')
        # Fly the particles in parallel and scrape the resulting data from the shell
        outs = core_fly(self,n_parts,cores,quiet,
                        trajectory_quality =self.trajectory_quality)
        data = str_data_scrape(outs,n_parts,cores,quiet)
        self.data = sim_data(data,symmetry = self.pa_info[0]['symmetry'],
                                    mirroring = self.pa_info[0]['mirroring'],
                                    obs = self.obs_region)
        if quiet == False:
            print(time.time() - start_time)

        return(self.data)

    def fly_trajectory(self,n_parts = 100,cores = multiprocessing.cpu_count(),
                      quiet = True, dat_step = 30,show= True,
                      fig = [],ax = [],cmap = 'eng',eng_cmap = cm.plasma,plt_kwargs = {},
                      show_cbar = True,label = '',xlim = [-np.inf,np.inf]):
        '''
        Fly n_parts particles, and plot their trajectories. Uses the particle probability 
        distributions defined in self.source, but tracks the particle movement. 
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

        # Write the workbench program in 'usr_prgm'
        if self.bench:
            with open(self.bench.replace('iob','lua'),'w') as fil:
                fil.write(self.usr_prgm)

        start_time = time.time()
        # Fly the particles in parallel and scrape the resulting data from the shell
        outs = core_fly(self,n_parts,cores,quiet,
                        rec_fil = os.path.join(self.home,'simPyon_traj.rec'),
                        markers = 20,trajectory_quality = 0)
        data = str_data_scrape(outs,n_parts,cores,quiet)
        
        if quiet == False:
            print(time.time() - start_time)
        
        # Parse the trajectory data into list of dictionaries
        head = ["n","x","y","z","v","grad V","ke"]

        from pandas import DataFrame
        self.traj_data = DataFrame(data,columns = head)
        self.traj_data['r'] = np.sqrt(self.traj_data['z']**2+self.traj_data[self.pa_info[0]['mirroring']]**2)

        if show == True:
            if cmap == 'eng':
                from matplotlib import cm
                from mpl_toolkits.axes_grid1 import make_axes_locatable
            if not ax:
                fig,ax = self.show()
            else:
                self.show(fig = fig,ax = ax)
            # for traj in self.traj_data:
            def traj_pltr(traj):
                if cmap == 'eng':
                    plt_kwargs['color'] = eng_cmap(traj['ke'].values[0]/np.max(self.source['ke'].dist_out))
                ax.plot(traj[self.pa_info[0]['base']],traj['r'],**plt_kwargs)
            self.traj_data.groupby('n').apply(traj_pltr)
            # ax.plot(traj[self.pa_info[0]['base']],traj['r'],label = label)
            if cmap == 'eng' and show_cbar == True:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=eng_cmap, 
                                                norm=plt.Normalize(vmin=np.nanmin(self.source['ke'].dist_out), 
                                                                   vmax=np.nanmax(self.source['ke'].dist_out))),
                                        ax = ax,label = 'Ke [eV]',cax = cax)


    def fly_trajectory_3d(self,n_parts = 100,cores = multiprocessing.cpu_count(),
                      quiet = True, dat_step = 30,show= True,geo_3d = False,
                      fig = [],ax = [],cmap = 'eng',eng_cmap = cm.plasma,plt_kwargs = {},
                      show_cbar = True,label = '',xlim = [-np.inf,np.inf]):
        '''
        Fly n_parts particles, and plot their trajectories. Uses the particle probability 
        distributions defined in self.source, but tracks the particle movement. 
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

        # Write the workbench program in 'usr_prgm'
        if self.bench:
            with open(self.bench.replace('iob','lua'),'w') as fil:
                fil.write(self.usr_prgm)

        start_time = time.time()
        # Fly the particles in parallel and scrape the resulting data from the shell
        outs = core_fly(self,n_parts,cores,quiet,
                        rec_fil = os.path.join(self.home,'simPyon_traj.rec'),
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
            dat['r'] = np.sqrt(dat['z']**2+dat[self.pa_info[0]['mirroring']]**2)

        # Plot the trajectories
        if show == True:
            if geo_3d == False:
                if cmap == 'eng':
                    from matplotlib import cm
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                if not ax:
                    fig,ax = self.show()
                else:
                    self.show(fig = fig,ax = ax)
                for traj in self.traj_data:
                    if cmap == 'eng':
                        plt_kwargs['color'] = eng_cmap(traj['ke'][0]/np.max(self.source['ke'].dist_out))
                    ax.plot(traj[self.pa_info[0]['base']],traj['r'],**plt_kwargs)
                ax.plot(traj[self.pa_info[0]['base']],traj['r'],label = label)
                if cmap == 'eng' and show_cbar == True:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=eng_cmap, 
                                                    norm=plt.Normalize(vmin=np.nanmin(self.source['ke'].dist_out), 
                                                                       vmax=np.nanmax(self.source['ke'].dist_out))),
                                            ax = ax,label = 'Ke [eV]',cax = cax)
            if geo_3d == True:
                from mpl_toolkits.mplot3d import Axes3D,art3d
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # elec_verts = self.geo.get_xy()
                for el in self.geo.get_xy():
                    # for vt in el:
                    vt = el[np.logical_and(el[:,0]>xlim[0],el[:,0]<xlim[1])]
                    ax.plot(vt[:,0],np.zeros(len(vt)),vt[:,1],
                            color = 'grey')
                        
                for traj in self.traj_data:
                    ax.plot(traj['x'],-traj['z'],traj['y'])
                ax.view_init(30, -70)
                ax.set_xlim(0,200)
                ax.set_xlim(0,200)
                ax.set_ylim(-100,100)
                
                return(fig,ax)
        return(fig,ax)

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
            print(' ===============================================')
            print('| Executing Simion Command:')
            print(' ===============================================')
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
    
    def show(self,measure = False,label = False, origin = [0,0],
             collision_locs = False,fig = [],ax = [],cmap = cm.viridis,
             show_verts = False,show_mirror  = False):
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
        if not ax:
            fig,ax = plt.subplots()
        for gm,pa_inf in zip(self.gemfil,self.pa_info):
            fig,ax1 = draw(gm,canvas = [pa_inf['Lx'],
                                        pa_inf['Ly']],
                                        fig = fig, ax = ax,
                                        mirror_ax = pa_inf['mirroring'],
                                        origin = origin,cmap = cmap,show_verts = show_verts, show_mirror = show_mirror)

        if show_verts == True:
            ax1.vpts = ax1.plot(np.concatenate(self.geo.get_x())-origin[0],
                     np.concatenate(self.geo.get_y())-origin[1],'.')[0]

        if measure == True:
            from . import fig_measure as meat
            fig,ax1 = meat.measure_buttons(fig,ax1,
                        verts = np.concatenate(self.geo.get_xy(),axis = 0) - origin if show_verts else [])

        ax1.set_ylabel('$r=\sqrt{z^2 + y^2}$ [mm]')
        ax1.set_xlabel('x [mm]')

        # if label ==True:


        if self.data != [] and collision_locs ==True:

            
            from scipy.interpolate import interpn

            def density_scatter( x , y, ax = None, sort = True, bins = 20,weights = None, **kwargs )   :
                """
                Scatter plot colored by 2d histogram
                """
                if ax is None :
                    fig , ax = plt.subplots()
                data , x_e, y_e = np.histogram2d( x, y, bins = bins,weights = weights)
                z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , 
                            np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

                # Sort the points by density, so that the densest points are plotted last
                if sort :
                    idx = z.argsort()
                    x, y, z = x[idx], y[idx], z[idx]

                ax.scatter( x, y, c=z, **kwargs )
                return ax

            ax1.plot(self.data.stop()['x'],self.data.stop()['r'],'.',color = 'r')
            ax1.plot(self.data.good().stop()['x'],self.data.good().stop()['r'],'.',color = 'blue')

            fig2,ax2 = plt.subplots(1)
            h,xbins,ybins = np.histogram2d(self.data.good().stop()()['z'],
                self.data.good().stop()()['y'], bins =int(30/10000*len(self.data['x'])/2),
                weights =self.data.good().stop()()['counts'])

            density_scatter(self.data.good().stop()['z'],self.data.good().stop()['y'],ax2,bins = 100,weights = self.data.good().stop()['ke'])
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
        for num,elec in self.elect_dict.items():
            if elec in self.volt_dict:
                voltin = input('Update %s Voltage [%fV] (enter to skip):'%(elec,self.volt_dict[elec]))
                if voltin == '':
                    volts[elec] = self.volt_dict[elec]
                else:
                    volts[elec] = float(voltin)    
            else:
                volts[elec] = float(input('Input %s Voltage [V]:'%elec))
        self.volt_dict = volts
        if save == True:
            np.save(input('File Name to Save to: '),volts)
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
            os.system('cls' if os.name == 'nt' else 'clear')
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

    def calc_pe(self,mm_pt=10,x_rng = None,y_rng = None,param = 'v',
                        cores = multiprocessing.cpu_count()):
        from .particles import source

        with open(self.bench.replace('iob','lua'),'w') as fil:
            fil.write('simion.workbench_program()\n adjustable max_time = 0   -- microseconds\n'+
                      'function segment.other_actions()\n if ion_time_of_flight >= max_time then\n'+
                        'ion_splat = -1 \n end\n end')

        if not x_rng:
            verts = np.concatenate(self.geo.get_x())
            x_rng = [min(verts),max(verts)]

        if not y_rng:
            verts = np.concatenate(self.geo.get_y())
            y_rng = [min(verts),max(verts)]
        
        x = np.linspace(x_rng[0],x_rng[1],int((x_rng[1]-x_rng[0])/mm_pt))
        y = np.linspace(y_rng[0],y_rng[1],int((y_rng[1]-y_rng[0])/mm_pt))
        xx,yy = np.meshgrid(x,y)
        xy = np.stack([xx.flatten(),yy.flatten()]).T

        p_source = auto_parts()
        p_source.n = len(xy)

        p_source.pos = source('fixed_vector',len(xy))
        p_source.pos['vector'] = xy

        store_parts = self.source

        self.source = p_source


        copy("%s/rec/simPyon_pe.rec"%\
                        os.path.dirname(os.path.dirname(__file__)+'..'),
                        self.home+'simPyon_pe.rec')

        dat = str_data_scrape(core_fly(self,len(xy),cores,quiet = False,
                                        rec_fil = self.home+'simPyon_pe.rec',
                                        trajectory_quality=0),len(xy),cores,quiet = False)
        # v_full = np.zeros(len(xy))
        # v_full[dat[:,0].astype(int)-1] = dat[:,5]

        v_full = np.zeros((len(xy),3))*np.nan
        v_full[dat[:,0].astype(int)-1] = dat[:,4:]

        v_dat = {}
        v_dat['x'] = x
        v_dat['y'] = y
        v_dat['xy'] = xy
        v_dat['v'] = v_full[:,0]
        v_dat['grad v'] = v_full[:,1]
        v_dat['B'] = v_full[:,2]

        self.v_data = v_dat
        self.source = store_parts
        # return(v_dat)

    def show_pe(self,param = 'v',cmap = cm.jet,vmax = None,
                vmin = None,imtype = 'both',levels = 10,
                colorbar_name = '',
                gaussian_sigma = 1,log = True,thresh = 200):

        def sym_logspace(start,stop,num,thresh=1):
            rng = abs(stop-start)
            # if abs(start)>abs(stop):

            sfrac = int(np.log(abs(start))/np.log(rng)*num)
            stfrac = int(np.log(abs(stop))/np.log(rng)*num)

            # print(start/abs(start)*np.geomspace(thresh,abs(start),sfrac))
            if start <0 and stop >0:
                return(np.sort(np.concatenate([
                                start/abs(start)*np.geomspace(thresh,abs(start),sfrac),
                                np.insert(stop/abs(stop)*np.geomspace(thresh,abs(stop),stfrac),0,0)],
                                )))

        from matplotlib import cm,ticker
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from scipy.ndimage import gaussian_filter as gf
        import matplotlib.colors as colors
        fig,ax = plt.subplots()
        im = []
        xy = self.v_data[param].reshape(-1,len(self.v_data['x'])).copy()
        mino = (np.nanmin(xy) if not vmin else vmin)
        maxo = (np.nanmax(xy) if not vmax else vmax)
        if imtype =='vmap' or imtype == 'both':
            dx = self.v_data['x'][1]-self.v_data['x'][0]
            dy = self.v_data['y'][1]-self.v_data['y'][0]
            x = np.zeros(len(self.v_data['x'])+1)
            y = np.zeros(len(self.v_data['y'])+1)
            x[:-1] = self.v_data['x'] - dx/2
            x[-1] = self.v_data['x'][-1]+dx/2
            y[:-1] = self.v_data['y'] - dy/2
            y[-1] = self.v_data['y'][-1]+dy/2
            xy = self.v_data[param].reshape(-1,len(self.v_data['x'])).copy()
            def gauss_filt_nan(U,sigma,truncate = 4):
                from scipy.ndimage import gaussian_filter
                V=U.copy()
                V[np.isnan(U)]=0
                VV=gaussian_filter(V,sigma=sigma,truncate=truncate)

                W=0*U.copy()+1
                W[np.isnan(U)]=0
                WW=gaussian_filter(W,sigma=sigma,truncate=truncate)

                return(VV/WW)
            
            if gaussian_sigma >0:
                xy = gauss_filt_nan(xy,gaussian_sigma)
            xy[np.isnan(xy)] = gf(np.nan_to_num(xy),sigma = 2)[np.isnan(xy)]

            im = plt.pcolormesh(x,y,xy, cmap = cmap,
                                vmax = vmax,vmin = vmin,
                                antialiased = True,shading = 'auto',
                                norm=(colors.SymLogNorm(linthresh=thresh,vmin=mino, vmax=maxo) if log == True else None))
        
        cont = []
        if imtype == 'contour' or imtype == 'both':
            xy = self.v_data[param].reshape(-1,len(self.v_data['x'])).copy()
            if type(levels)==int and log == True:
                for thing in [mino,maxo,levels,thresh]:print(thing)
                plt_levels = sym_logspace(mino,maxo,levels,thresh)
            elif type (levels) == int:
                plt_levels = np.linspace(mino,maxo,levels)
            else: plt_levels = levels
            cont = plt.contour(self.v_data['x'],self.v_data['y'],xy,
                               norm=(colors.SymLogNorm(linthresh=thresh,vmin=mino, vmax=maxo) if log == True else None),
                             levels = plt_levels,)
        
        self.show(fig = fig,ax = ax,show_mirror = True)
        # return(cont)
        if im or cont:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            if im:
                cbar = plt.colorbar(im,ax = ax,label = colorbar_name,cax = cax)
                if cont:
                    cbar.set_ticks(cont.levels)
                    cbar.set_ticklabels(cont.levels)
            if cont:
                tickmax = 15
                skipo = int(len(cont.levels)/tickmax)+1
                cbar = plt.colorbar(cont,ax = ax,label = colorbar_name,cax = cax)
                cbar.set_ticks(cont.levels.round(0)[::skipo])
                cbar.set_ticklabels(cont.levels.round(0)[::skipo])
                # cbar.set_ticklabels(cbar.get_ticklabels()[::2])
        plt.tight_layout()
        ax.set_xlim(min(self.v_data['x']),max(self.v_data['x']))
        ax.set_ylim(min(self.v_data['y']),max(self.v_data['y']))
        return(fig,ax)

    def interp_pe(self,param = 'v'):
        from scipy.interpolate import interp2d
        return(interp2d(self.v_data['x'],self.v_data['y'],
                        self.v_data[param].reshape(-1,len(self.v_data['x'])).copy()))


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
    for i in range(int(cores)):
        pt = os.path.join(sim.home,'auto_fly_%i.ion'%i)
        fly_fils.append(pt)

    sim.source['n'] = n_parts
    sim.source.fil = fly_fils
    sim.source.ion_print()

    for ion_fil in fly_fils:
        loc_com = r"fly  "
        loc_com+=r" --retain-trajectories=0 --restore-potentials=0"
        loc_com+=r" --trajectory-quality=%d"%trajectory_quality
        if markers !=0:
            loc_com+=r" --markers=%d"%markers
        loc_com+=r" --recording=%s"%(sim.recfil if rec_fil == '' else rec_fil)
        loc_com += r" --particles=%s"%ion_fil
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
            

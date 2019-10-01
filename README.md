# simPyon
Python wrapper for [SIMION](https://simion.com/) ion simulation software. This package was developed at the University of New Hampshire for the purpose of automating a number of simulation processes surrounding SIMION. SIMION has been shown to be an effective and efficient tool to simulate simple electrostatics, that has a low barrier of entry for new scientist looking to try thier hand at instrument development. The purpose of this package is to expand on that goal, and make some of the more advanced simulation processes accessable at a lower level. This package parallelizes ION flight calculations, extracts particle data to numpy data structures, and can be easily used in itterative voltage and geometry refinement and optimization.

## Getting started
Once this repository is downloaded, there are a couple steps that need to be taken to get this bad boy up and running. For one, I have not yet set this up as a Python installable, so this repo should be cloned to a folder python tracks, or you will just have to import the package using the file location:
```python
import sys
sys.path.append(r"C:\YOUR\simPyon\FOLDER\LOCATION\")
```
Then you should be able to just import simPyon the normal way:
```python
from simPyon import * 
```
### Prerequisites
- SIMION 8.1.1.32 or higher
- Python 3
- NumPy
- SciPy
- Matplotlib
- multiprocessing
- subpprocess

### Adding SIMION to PATH
Aside from having an updated version of Python 3 and SIMION installed, SIMION needs to be added to PATH so that it is accessable to  windows CMD. 

The process to add folders to path is described [here](https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/)
You will have to identify the SIMION install location, and add that folder location to the path variable. A likely location is:
```
C:\Program Files\SIMION-8.1
```
With the SIMION folder added to path you should be able to boot SIMION straight from the CMD by just calling:
```
> simion
``` 
Make sure not to reassign that system variable, it will break this package. 

## Running SIMION

### Initializing simPyon Environment
simPyon functions best from quasi-3D geometry builds constructed using the SIMION gem language, described [here](https://simion.com/info/gem_geometry_file.html). This package can function fine with any of the other geometry creation methods, but the geomerty visualization and fast-adjusting tools rely on the .gem file, so most of the functionality here requires gem usage. 

The simplist way to get the simPyon environment up and running is generally to do the first gem to Pa conversion, refinement and fast adjusting through the SIMION gui. Once those steps have been taken, save the workbench to the same folder containing the gemfile, and potential arrays, so that within the path containing your workspace there is one workbench .iob file, and its associated .gem and .pa0 files. With those three files in root, simPyon can be initialized by simply importing the package and calling:
```python 
In [1]:sim = simPyon.simion()
```
This will pull together your workbench (.iob), gem file(.gem) and potential array(.pa0) together automatically through their file type, so it is important that there is only one of each in the working directory.

### Generate Pa from Gemfile
You can generate new potentail arrays or update potential arrays using simPyon. This functionality is helpful when doing itterative geometry optimization. 
```python 
In [1]:sim.gem2pa('name_of_new_pafile')
```
This simply generates a .pa0# file, which is refined to the full potential array using:
```python
In [1]:sim.refine()
```
### Fast Adjust Voltages
The voltages can be fast adjusted from the python commandline which is easily exploited for looping: fast adjust-->fly-->fast adjust again. If you don't fast adjust, the voltages will just be taken from the last fast adjust, which is stored in the .pa0 file. 

Fast adjusting is completed using names pulled from the .gemfile, the electrode name is taken as the commented string following a electrode declaration in the gemfile:
```Lua
electrode(1); collimator
{
````
Here the name "collimator" would be assigned to electrode #1, It should be noted, any electrodes that share numbers will be assigned the same voltage, and any electrodes numbered 0 will automatically be set to ground, and will not be fast adjustable. 

You can mannually input voltages through the commandline using:
```python
In [1]:sim.define_volts()
collimator:
``` 
Which will print the electrode names to the commandline and prompt an electrode voltage input. You can also define the electode voltages using a dictionary of electrode names and voltages and assign it to the .volt_dict variable:
```python
In [1]:sim.volt_dict = {'collimator':100}
``` 
With the voltages defined you still need to actually fast adjust the potential arrays:
```python
In [1]:sim.fast_adjust()
``` 
The Lua output will be printed to the commandline, unless you set quiet=True. It is a good idea to make sure from the output text that the fast adjust succeeded, you can run into problems if the potential array is open in another program, so make sure no other instances of SIMION are running. 
### Flying particles
Probably the most important SIMION command supported by this package is the fly functionality. This package parrallelized the fly function by opening an SIMION instances on each core and flying a subgroup of the total number of particles in each instance, the output from each core is collected once all cores have returned. Unfortunatly, the particle distributions must be set from inside this package, SIMION's handleing of random numbers is such that repeated fly commands whith the same particle sources will simply repeat the exact same calculation on each core, effectively making the paralleization pointless. So the default particle sources from inside SIMION cannot be used. 

A discussion reguarding setting the particles sources is takes place below. With the particle distributions set, to fly 10000 particles, you simply use:
```python
In [1]:sim.fly(n_parts = 10000)
``` 
**It should be noted that this defaults to use the total number of cores available and will grab 100% of the available processing power, so keep an eye on system temp if your system doesn't control that well. You can manually set the number of cores in the fly command by just using ```sim.fly(cores = NUMBER_OF_DESIRED_CORES)```.

### Setting the particle distribution
The particle distributions are handled using the ```simPyon.particles.source()``` class, which defines the randomly selected values for the particle's:
- mass: Mass in amu (int)
- charge: elementry charge (int)
- ke: Kinetic energy [ev] (float)
- az: Velocity vector elevation angle [deg]. Defined from the x-z axis perpendicular from the axis of rotation
- el: Velocity vector elevation angle [deg]. Defined from the x axis in the axis of rotation
- pos: Position [mm]

The souce distribution class is constructed using a nested dict, containing the function name and input parameters. So the particle distribution kinetic energy (ke), might be defined by a gaussian distribution, with a mean value of 100eV and FWHM of 50eV ie. ```{'gaussian':{'mean':100,'fwhm':50}}``` 

The source distributions can be changed in two ways; 
1) editing the default values defined in ```simPyon/defaults.py```. Whenever a simPyon environment is initialized, these values will automatically assigned to the particles distributions. 
2) Actively changeing the simPyon particle distribution:
  The distribution type for each of the particle parameters can be updated by setting it to a new source: 
  ```python
  In [1]:sim.parts.pos = simPyon.particles.source('gaussian')
  ``` 
  Which changes the distribution type from the default distribution to a gaussian, and assigns new distribution parameters ```sim.parts.pos.dist_vals```. The distribution parameters can be updated by assignment:
  ```python
  In [1]:sim.parts.pos.dist_vals['fwhm'] = 100
  ``` 
  Which makes the full width half max of the position distribution 100mm. You can print the current settings of the source distribution by calling:
  ```python
  In [1]:print(sim.parts())
  ```
## Supported Particle Source Distributions
Most, but not all of the SIMION supported source distributions are available through simPyon, the modular structure of the source class makes adding new distributions quite simple. The currently supported functions and input parameters can be viewed in the terminal:
```shell
In [1]:simPyon.particles.source().defaults
Out[4]:
{'gaussian': {'mean': 0, 'fwhm': 1},
 'uniform': {'min': 0, 'max': 1},
 'line': {'first': array([0, 0, 0]), 'last': array([1, 1, 0])},
 'single': {'value': 1},
 'sputtered': {'E_beam': 105, 'a': 0.01, 'b': 0.65},
 'cos': {'mean': 75, 'range': 180}}
``` 
The source class handles random sampling the probability distributions through a combination of ```numpy.random```, for numpy supported distributions, and an in house sampling tecnique in defined by the  ```simPyon.particles.pdf``` class. The Currently supported distribution are: 
### Gaussian:
```
dist_vals={'mean': 0, 'fwhm': 1}
```
Draw Random samples from a gaussian (normal) distribution. Utilizes [numpy.random.normal]{https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.normal.html} for sampling. 
  The probability density for the gaussian is given by:
  
  ```p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }} e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} } + dist_vals[mean]```
  
  with ```\sigma = abs(dist_vals[fwhm]/(2*\sqrt{2*ln(2)}))```

### Uniform: 
```
dist_vals={'min': 0, 'max': 1}
```
Draw random samples from a flat, uniform distribution, using [numpy.random.uniform]{https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.uniform.html}. Samples between upper and lower bounds of ```[dist_vals[min],dist_vals[max]]```

### Line: 
```
dist_vals={'first': array([0, 0, 0]), 'last': array([1, 1, 0])}
```
Draw random (x,y) position values along a line defined by points ```(dist_vals[first],dist_vals[last])```, where each of the ```first(x,y)``` and ```last(x,y)``` distribution values is defined using their x and y position. The output x_rand values are selected using ```np.random.uniform```, defined between ```[first(x),last(x)]``` and the associated y_rand values is selected using:
  ```
  y_rand(x_rand) = \frac{last(y)-first(y)}{last(x)-first(x)}*x_rand+first(y)
  ```
  
### Single:
```
dist_vals={'value': 1}
```
Returns a single value array of length n, comprised of ```p(x) = dist_vals[value]```
 
### Sputtered: 
```
dist_vals={'E_beam': 105, 'a': 0.01, 'b': 0.65}
```
Draws random samples from a probability distribution defined in ```simPyon.particles.pdf.sputtered```, over the range ```[dist_vals[a],dist_vals[b]]```. The distribution shape generated using the approximate shape of the energy distribution of sputtered particles from a 15degree incident angle particle beam of energy ```dist_vales[E_beam]```. The functional approximation for the sputtered particle energy distribution is defined by:
  ```
  p(x) = (a*x - b*(x-c)**2+ d*\ln(x**(-d)))*dist_vals[E_beam]
  with a,b,c,d = (-2.12621554, 12.28266708,  0.762046  ,  1.95374093)
  ```
  
### Cos:
```
dist_vals={'mean': 75, 'range': 180}
```
Draws random samples from cosine probability distribution defined in ```simPyon.particles.pdf.cos```, over the range ```[dist_vals[mean]-dist_vals[range]/2,dist_vals[mean]+dist_vals[range]/2]```. The probability density for the cos distribution is given by:
```
p(x) = cos(x)*dist_vals[range]-dist_vals[range]/2+dist_vals[mean]
```
Where x is defined between ```[-\pi/2,\pi/2]```

## Output Data structure

##  Examples
Example Python script showing the functionality can bee seen in ```simPyon/examples/sim_init.py```. This script needs to be copied to a folder containg a simion workplace to function. 

## Versioning 

## Authors
- Jonathan Bower, Research Scientist II, EOS Space Science Center, University of New Hamphire, Durham NH

## License

## Acknowledgements


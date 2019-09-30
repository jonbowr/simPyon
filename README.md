# simPyon
Python wrapper for [SIMION](https://simion.com/) ion simulation software. This package was developed at the University of New Hampshire for the purpose of automating a number of simulation processes surrounding SIMION. SIMION has been shown to be an effective and efficient tool to simulate simple electrostatics, that has a low barrier of entry for new scientist looking to try thier hand at instrument development. The purpose of this package is to expand on that goal, and make some of the more advanced simulation processes accessable at a lower level. This package parallelizes ION flight calculations, extracts particle data to numpy data structures, and can be easily used in itterative voltage and geometry refinement and optimization.

## Getting started
Once this repository is downloaded, there are a couple steps that need to be taken to get this bad boy up and running. For one, I have not yet set this up as a Python installable, so this repo should be cloned to a folder python tracks, or you will just have to import the file using the file location:
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

## Supported Particle Source Distributions

##  Examples
```Python
Check This Example
```

## Versioning 

## Authors
- Jonathan Bower

## License

## Acknowledgements


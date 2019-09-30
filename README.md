# simPyon
Python wrapper for [SIMION](https://simion.com/) ion simulation software. This package was developed at the University of New Hampshire for the purpose of automating a number of simulation processes surrounding SIMION. SIMION has been shown to be an effective and efficient tool to simulate simple electrostatics, that has a low barrier of entry for new scientist looking to try thier hand at instrument development. The purpose of this package is to expand on that goal, and make some of the more advanced simulation processes accessable at a lower level. This package parallelizes ION flight calculations, extracts particle data to numpy data structures, and can be easily used in itterative voltage and geometry refinement and optimization.

## Getting started

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

## Supported Particle Source Distributions

## GEM 

##  Examples
```Python
Check This Example
```

## Versioning 

## Authors
- Jonathan Bower

## License

## Acknowledgements


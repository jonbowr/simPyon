from setuptools import setup

setup(name='simPyon',
      version='0.1',
      description='Simion Ion Simulation Wrapper',
      url='http://https://github.com/jonbowr/simPyon',
      author='J. S. Bower',
      author_email='jonathan.bower@unh.edu',
      license='NOSA',
      packages=['data','gem','particles','simion'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'multiprocessing',
          'subprocess',
          'mpl_toolkits',
          'descartes',
          'shapely',
          'tempfile',
          'shutil',
      ],
      zip_safe=False)
import numpy as np

#==========================================================
# Declare default values for the source distribution
# To see a list of supported distributions and there input 
# values, input the following to python cmd:
# simPyon.particles.source().defaults
#==========================================================
MASS = 1
CHARGE = -1

# Energy defaults
KE_DIST_TYPE = 'uniform'
KE_DIST_VALS = {'min':0,
				'max':1000} #energy in ev

# Angle defaults
# Azimuthal
AZ_DIST_TYPE = 'gaussian'
AZ_DIST_VALS = {'mean':0,
				'fwhm':24}
# Elevation
EL_DIST_TYPE = 'gaussian'
EL_DIST_VALS = {'mean':150,
				'fwhm':19.2}

# Starting Position
POS_DIST_TYPE = 'line'
POS_DIST_VALS = {'first':np.array([99.4,133,0]),
				'last':np.array([158.9,116.8,0])}

#==========================================================
# Define the measurement region as a box, using the 
#	cylindrical r and x coordinates. These values are used 
#	in the data package for processing and visualization.
#==========================================================
X_MAX = 81
X_MIN = 72

R_MAX = 45.1
R_MIN = 35.4

# Use the impact angle to help more accurately determine
#	observability at ToF
TOF_MEASURE = True

#==========================================================
# Weight the particle counts by their distance from the 
# rotation axis to account for increasing diferential 
# surface area. 
#==========================================================
R_WEIGHT = True
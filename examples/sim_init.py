import simPyon as sim
import numpy as np
# To run this file must be in the same folder as 

def main():
	# Load up some voltages
	volts = np.load('volts_de_e_1_330_3.npy').item()

	# Load Simion command environment
	imap = sim.simion()

	# mannually input voltages
	imap.define_volts()

	# Fast Adjust voltages 
	imap.fast_adjust(elec_dict = volts) # if volts was loaded
	imap.fast_adjust() #if voltages were input mannually

	# Define Line distributions for source
	line_1 = [np.array([260,156,0]),np.array([270,106,0])]
	line_2 = [np.array([99.4,133,0])+10,np.array([158.9,116.8,0])+10]

	# Print particle source description
	print(imap.parts())

	# Change distribution type
	# imap.parts.pos = sim.particles.source('gaussian')

	# Change Distribution inputs
	imap.parts.pos.dist_vals['first'] = line_1[0]
	imap.parts.pos.dist_vals['last'] = line_1[1]

	# Fly Particles with souce line_1 and store in data_line_1
	data_line_1 = imap.fly(10000).data

	# Chance source location to line_2 and fly
	imap.parts.pos.dist_vals['first'] = line_2[0]
	imap.parts.pos.dist_vals['last'] = line_2[1]
	data_line_2 = imap.fly(10000).data

	# Show Simion Geometry and last flown particles
	imap.show()
	# enable measurement mode
	imap.show(measure = True)

	# Plot distributions of flow data
	data_line_2.show()
	data_line_1.show()

if __name__ =='__main__':
	main()
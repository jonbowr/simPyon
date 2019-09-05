from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

class img_measure:
	def __init__(self,img):
		self.click_num = 0
		self.fig,self.ax = plt.subplots()
		self.ax.imshow(img,origin = 'lower')
		self.img = img
		self.pointer_loc = np.zeros(2)
		self.lines = []


	def connect(self):
		print('Fig Measure connected:\nDouble Click outside Plot to Disconnect')
		self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		input('')
	
	def onclick(self,event):
		if plt.get_current_fig_manager().toolbar.mode != '': return
		if event.dblclick == True:
			self.click_num = 2
		if event.ydata == None and event.dblclick == True:
			# return([self.og_parts,self.group_parts])
			self.disconnect()
		elif event.ydata != None and event.xdata != None:
			if self.click_num % 2 == 0:
				for line in self.lines:
					line.remove()
				self.lines = []
				self.fig.canvas.draw()
				self.fig.canvas.draw
				self.part_num = self.img[int(event.ydata), int(event.xdata)]
				self.pointer_loc = [event.xdata,event.ydata]
				self.lines += [self.ax.plot(self.pointer_loc[0],self.pointer_loc[1],'+',color = 'k')[0]]
				self.click_num += 1
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				print('Pointer Position:')
				print('p1 = ( %.2f , %.2f )' %(event.xdata,event.ydata))
			else:
				self.part_num = self.img[int(event.ydata), int(event.xdata)]
				self.pointer_loc = np.stack((self.pointer_loc,[event.xdata,event.ydata]))
				self.lines += [self.ax.plot(event.xdata,event.ydata,'+',color = 'k')[0]]
				self.lines+= [self.ax.plot(self.pointer_loc[:,0],self.pointer_loc[:,1],color = 'k')[0]] 
				org = [self.pointer_loc[1,0],self.pointer_loc[0,1]]
				L = np.sqrt(np.sum((self.pointer_loc - org)**2))
				dx = self.pointer_loc[1,0]-self.pointer_loc[0,0]
				dy = self.pointer_loc[1,1]-self.pointer_loc[0,1]
				ang = 180/np.pi*np.arctan2(dy,dx)
				sml_ang = 180/np.pi*np.arctan2(dy,np.abs(dx))
				guides = np.stack((self.pointer_loc[0,:],org,self.pointer_loc[1,:]))
				self.lines += [self.ax.plot(guides[:,0],guides[:,1],color = 'k',linestyle = 'dashed')[0]] 


				self.lines += [self.ax.annotate('%.2f'%dx,(np.mean(self.pointer_loc[:,0]),
								org[1]),xytext = (0,-dy/abs(dy)*14 if dy>0 else 4),textcoords = 'offset pixels')]
				self.lines += [self.ax.annotate('%.2f'%dy,(org[0],np.mean(self.pointer_loc[:,1])),
								xytext = (dx/abs(dx)*14 if dx <0 else 4,0),textcoords = 'offset pixels',
								rotation = 'vertical')]
				self.lines += [self.ax.annotate('%.1f$\degree$'%sml_ang,self.pointer_loc[0,:],
								xytext = (-dx/abs(dx)*40 if dx > 0 else 4,
									-dy/abs(dy)*14 if dy>0 else 4),textcoords = 'offset pixels')]


				arc_angs = np.sort([ang,ang-sml_ang*dx/abs(dx)])
				ang_draw = patches.Arc((self.pointer_loc[0,0],self.pointer_loc[0,1]),
					width = abs(dx)**.8,height = abs(dx)**.8,theta1=arc_angs[0],theta2=arc_angs[1])
				self.lines += [self.ax.add_patch(ang_draw)]
				self.click_num += 1
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				print('p2 = ( %.2f , %.2f )' %(event.xdata,event.ydata))
				print('dx = %.2f'%(dx))
				print('dy = %.2f'%(dy))
				print('dr = %.2f'%L)
				print('Elev: %.2f'%(sml_ang))

	def disconnect(self):
		self.fig.canvas.mpl_disconnect(self.cid)
		plt.close(self.fig)
		print('We are Disconnected: Press Any key to continue')

class fig_measure:
	def __init__(self,fig,ax):
		self.click_num = 0
		self.fig = fig
		self.ax = ax
		self.pointer_loc = np.zeros(2)
		self.lines = []


	def connect(self):
		print('Fig Measure connected:\nDouble Click outside Plot to Disconnect')
		self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		input('')
	
	def onclick(self,event):
		if plt.get_current_fig_manager().toolbar.mode != '': return
		if event.dblclick == True:
			self.click_num = 2
		if event.ydata == None and event.dblclick == True:
			# return([self.og_parts,self.group_parts])
			self.disconnect()
		elif event.ydata != None and event.xdata != None:
			if self.click_num % 2 == 0:
				for line in self.lines:
					line.remove()
				self.lines = []
				self.fig.canvas.draw()
				self.fig.canvas.draw
				self.pointer_loc = [event.xdata,event.ydata]
				self.lines += [self.ax.plot(self.pointer_loc[0],self.pointer_loc[1],'+',color = 'k')[0]]
				self.click_num += 1
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				print('Pointer Position:')
				print('p1 = ( %.2f , %.2f )' %(event.xdata,event.ydata))
			else:
				self.pointer_loc = np.stack((self.pointer_loc,[event.xdata,event.ydata]))
				self.lines += [self.ax.plot(event.xdata,event.ydata,'+',color = 'k')[0]]
				self.lines+= [self.ax.plot(self.pointer_loc[:,0],self.pointer_loc[:,1],color = 'k')[0]] 
				org = [self.pointer_loc[1,0],self.pointer_loc[0,1]]
				L = np.sqrt(np.sum((self.pointer_loc - org)**2))
				dx = self.pointer_loc[1,0]-self.pointer_loc[0,0]
				dy = self.pointer_loc[1,1]-self.pointer_loc[0,1]
				ang = 180/np.pi*np.arctan2(dy,dx)
				sml_ang = 180/np.pi*np.arctan2(dy,np.abs(dx))
				guides = np.stack((self.pointer_loc[0,:],org,self.pointer_loc[1,:]))
				self.lines += [self.ax.plot(guides[:,0],guides[:,1],color = 'k',linestyle = 'dashed')[0]] 


				self.lines += [self.ax.annotate('%.2f'%dx,(np.mean(self.pointer_loc[:,0]),
								org[1]),xytext = (0,-dy/abs(dy)*14 if dy>0 else 4),textcoords = 'offset pixels')]
				self.lines += [self.ax.annotate('%.2f'%dy,(org[0],np.mean(self.pointer_loc[:,1])),
								xytext = (dx/abs(dx)*14 if dx <0 else 4,0),textcoords = 'offset pixels',
								rotation = 'vertical')]
				self.lines += [self.ax.annotate('%.1f$\degree$'%sml_ang,self.pointer_loc[0,:],
								xytext = (-dx/abs(dx)*40 if dx > 0 else 4,
									-dy/abs(dy)*14 if dy>0 else 4),textcoords = 'offset pixels')]


				arc_angs = np.sort([ang,ang-sml_ang*dx/abs(dx)])
				ang_draw = patches.Arc((self.pointer_loc[0,0],self.pointer_loc[0,1]),
					width = abs(dx)**.8,height = abs(dx)**.8,theta1=arc_angs[0],theta2=arc_angs[1])
				self.lines += [self.ax.add_patch(ang_draw)]
				self.click_num += 1
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				print('p2 = ( %.2f , %.2f )' %(event.xdata,event.ydata))
				print('dx = %.2f'%(dx))
				print('dy = %.2f'%(dy))
				print('dr = %.2f'%L)
				print('Elev: %.2f'%(sml_ang))

	def disconnect(self):
		self.fig.canvas.mpl_disconnect(self.cid)
		plt.close(self.fig)
		print('We are Disconnected: Press Any key to continue')

class measure:
	lock = None #  only one can be animated at a time
	def __init__(self,fig,ax):
		self.click_num = 0
		self.fig = fig
		self.ax = ax
		self.pointer_loc = np.zeros((2,2))
		self.lines = []
		self.tot_lines = []
		# self.pts = []
		self.grab = [False,False]
		self.pts = [self.ax.plot(0,0,'+',color = 'k')[0],
					self.ax.plot(0,0,'+',color = 'k')[0]]
		self.line = self.ax.plot([0,0],[0,0])[0]

	def connect(self):
		print('Fig Measure connected:\nDouble Click outside Plot to Disconnect')
		self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
		self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
		input('')
	
	def onclick(self,event):
		# print(self.click_num)
		# measure.lock = self
		if plt.get_current_fig_manager().toolbar.mode != '': return
		if event.ydata != None and event.dblclick == True:
			# self.clear()
			self.tot_lines += self.lines
			self.lines = []
			self.click_num = 1
			self.grab = [False,False]
			for pt in self.pts:
				pt.set_xdata(event.xdata)
				pt.set_ydata(event.ydata)

			self.pointer_loc = np.array([[pt.get_xdata(),pt.get_ydata()] for pt in self.pts])
			self.line.set_xdata(self.pointer_loc[:,0])
			self.line.set_ydata(self.pointer_loc[:,1])
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
		elif event.ydata == None and event.dblclick == True:
			self.tot_clear()
			self.click_num = 0
			self.grab = [False,False]
			for pt in self.pts:
				pt.set_xdata(0)
				pt.set_ydata(0)
			self.line.set_xdata([0,0])
			self.line.set_ydata([0,0])
			self.pointer_loc = np.array([[pt.get_xdata(),pt.get_ydata()] for pt in self.pts])
			self.line.set_xdata(self.pointer_loc[:,0])
			self.line.set_ydata(self.pointer_loc[:,1])
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
		elif event.dblclick == False and event.ydata != None and event.xdata != None:
			if self.click_num == 0:
				self.lines = []
				self.fig.canvas.draw()
				self.fig.canvas.draw
				self.pointer_loc[0,:] = np.array([event.xdata,event.ydata])
				for pt in self.pts:
					pt.set_xdata(event.xdata)
					pt.set_ydata(event.ydata)
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()
				# print('Pointer Position:')
				# print('p1 = ( %.2f , %.2f )' %(event.xdata,event.ydata))
			elif self.click_num == 1:
				self.pointer_loc[1,:] = np.array([event.xdata,event.ydata])
				self.pts[1].set_xdata(event.xdata)
				self.pts[1].set_ydata(event.ydata)
				self.line.set_xdata(self.pointer_loc[:,0])
				self.line.set_ydata(self.pointer_loc[:,1])
			# # print(self.pointer_loc)

				# self.pts += [self.ax.plot(self.pointer_loc[-1,0],self.pointer_loc[-1,1],'+',color = 'k')[0]]
				self.set_pts()
			self.click_num += 1
			self.grab = [pt.contains(event)[0] for pt in self.pts]
			# if self.click_num > 2 and any(self.grab):
			# 		self.clear()
			# print(self.grab)
			# print('======================')
	def on_motion(self,event):
		if any(self.grab) and self.click_num > 2:
			# print(np.argwhere(self.grab)[0][0])
			self.pts[np.argwhere(self.grab)[0][0]].set_xdata(event.xdata)
			self.pts[np.argwhere(self.grab)[0][0]].set_ydata(event.ydata)
			self.pointer_loc = np.array([[pt.get_xdata(),pt.get_ydata()] for pt in self.pts])
			self.line.set_xdata(self.pointer_loc[:,0])
			self.line.set_ydata(self.pointer_loc[:,1])
			# self.pointer_loc = np.array([[pt.get_xdata(),pt.get_ydata()] for pt in self.pts])
			# self.line.set_xdata(self.pointer_loc[:,0])
			# self.line.set_ydata(self.pointer_loc[:,1])
			# # print(self.pointer_loc)
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()

			# self.set_pts()
		# else:print('missed it')

	def on_release(self,event):
		# print('let it gooooo')
		if any(self.grab) and self.click_num > 2:
			self.clear()
			self.set_pts()
		self.grab = [False,False]
		

	def clear(self):
		# self.click_num = 0
		for lineer in self.lines:
			lineer.remove()
		self.lines = []
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
	def tot_clear(self):
		# self.click_num = 0
		for lineer in self.lines:
			lineer.remove()
		for lin in self.tot_lines:
			lin.remove()
		self.tot_lines = []
		self.lines = []
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

	def set_pts(self):
		# print(self.pointer_loc)
		self.lines += [self.ax.plot(self.pointer_loc[:,0],self.pointer_loc[:,1],color = 'k')[0]] 
		org = [self.pointer_loc[1,0],self.pointer_loc[0,1]]
		L = np.sqrt(np.sum((self.pointer_loc - org)**2))
		dx = self.pointer_loc[1,0]-self.pointer_loc[0,0]
		dy = self.pointer_loc[1,1]-self.pointer_loc[0,1]
		ang = 180/np.pi*np.arctan2(dy,dx)
		sml_ang = 180/np.pi*np.arctan2(dy,np.abs(dx))
		guides = np.stack((self.pointer_loc[0,:],org,self.pointer_loc[1,:]))
		self.lines += [self.ax.plot(guides[:,0],guides[:,1],color = 'k',linestyle = 'dashed')[0]] 
		self.lines += [self.ax.annotate('%.2f'%dx,(np.mean(self.pointer_loc[:,0]),
						org[1]),xytext = (0,-dy/abs(dy)*14 if dy>0 else 4),
						textcoords = 'offset pixels',horizontalalignment = 'center')]
		self.lines += [self.ax.annotate('%.2f'%dy,(org[0],np.mean(self.pointer_loc[:,1])),
						xytext = (dx/abs(dx)*14 if dx <0 else 4,0),textcoords = 'offset pixels',
						rotation = 'vertical',verticalalignment = 'center')]
		self.lines += [self.ax.annotate('(%.1f,%.1f)'%(self.pointer_loc[0,0],self.pointer_loc[0,1]),
						self.pointer_loc[0,:],
						xytext = (-dx/abs(dx)*40 if dx > 0 else 4,
							-dy/abs(dy)*14 if dy>0 else 4),textcoords = 'offset pixels')]
		self.lines += [self.ax.annotate('(%.1f,%.1f)'%(self.pointer_loc[1,0],self.pointer_loc[1,1]),
						self.pointer_loc[1,:],
						xytext = (dx/abs(dx)*40 if dx < 0 else 4,
							dy/abs(dy)*14 if dy<0 else 4),textcoords = 'offset pixels')]
		self.lines += [self.ax.annotate('%.1f$\degree$'%sml_ang,
						(np.mean(self.pointer_loc[:,0]),self.pointer_loc[0,1]),
						xytext = (0,dy/abs(dy)*12),
						textcoords = 'offset pixels',horizontalalignment = 'center',
						verticalalignment = 'center')]
		self.lines += [self.ax.annotate('%.2f'%L,np.mean(self.pointer_loc,axis = 0),
						rotation = (ang if dx>0 else ang-180),
						rotation_mode = 'anchor',
						verticalalignment = 'center',
						horizontalalignment = 'center',
						# xytext = (-(6*np.sin(np.degrees(ang))),(6*np.cos(np.degrees(ang)))),
						xytext = (-dx/abs(dx)*abs(10*dy/L),
							    dy/abs(dy)*abs(10*dx/L)),
						textcoords = 'offset pixels')]
		arc_angs = np.sort([ang,ang-sml_ang*dx/abs(dx)])
		ang_draw = patches.Arc((self.pointer_loc[0,0],self.pointer_loc[0,1]),
			width = abs(dx)**.8,height = abs(dx)**.8,theta1=arc_angs[0],theta2=arc_angs[1])
		self.lines += [self.ax.add_patch(ang_draw)]
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		print(self.pointer_loc)
		print('dx = %.2f'%(dx))
		print('dy = %.2f'%(dy))
		print('dr = %.2f'%L)
		print('Elev: %.2f'%(sml_ang))


	def disconnect(self):
		self.fig.canvas.mpl_disconnect(self.cidpress)
		self.fig.canvas.mpl_disconnect(self.cidrelease)
		self.fig.canvas.mpl_disconnect(self.cidmotion)
		plt.close(self.fig)
		print('We are Disconnected: Press Any key to continue')


def img_meat(img):
	meat_show = measure(img)
	meat_show.connect()

def part_plot(part_groups):
	i = 0
	for nam in part_groups:
		i += 1
		for part in part_groups[nam]:
			# print(part)
			part = np.concatenate((part[-1:,:],part))
			cmap = plt.cm.get_cmap('hsv')
			plt.plot(part[:,0],part[:,1],color = cmap(i/len(part_groups)))
				# plt.xlim(xlim)
				# plt.ylim(ylim)
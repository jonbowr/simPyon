from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from matplotlib.widgets import Button

class img_measure:
	def __init__(self,img):
		self.click_num = 0
		self.fig,self.ax = plt.subplots()
		self.ax.imshow(img,origin = 'lower')
		self.img = img
		self.pointer_loc = np.zeros(2)
		# self.lines = []
		# ax_clear = fig.add_subplot([0.7, 0.05, 0.1, 0.075])
		# ax_clear_all = fig.add_subplot([0.81, 0.05, 0.1, 0.075])
		# self.clear_butt = Button(ax_clear,'Undo')
		# self.clear_all_butt = Button(ax_clear_all,'Clear All')

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

class measureog:
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
		self.active = True

	def connect(self):
		self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
		self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
		return(self)
		# input('')
	
	def onclick(self,event):
		# print(self.click_num)
		# measure.lock = self

		self.grab = [pt.contains(event)[0] for pt in self.pts]

		if plt.get_current_fig_manager().toolbar.mode != '': return
		
		if self.active == False: return

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
				# self.line.set_xdata(self.pointer_loc[:,0])
				# self.line.set_ydata(self.pointer_loc[:,1])
			# # print(self.pointer_loc)

				# self.pts += [self.ax.plot(self.pointer_loc[-1,0],self.pointer_loc[-1,1],'+',color = 'k')[0]]
				self.set_pts()
			self.click_num += 1
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
			# self.line.set_xdata(self.pointer_loc[:,0])
			# self.line.set_ydata(self.pointer_loc[:,1])
			if any(self.grab) and self.click_num > 2:
				self.clear()
				self.set_pts()
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
		self.lines += [self.ax.annotate('(%.2f,%.2f)'%(self.pointer_loc[0,0],self.pointer_loc[0,1]),
						self.pointer_loc[0,:],
						xytext = (-dx/abs(dx)*40 if dx > 0 else 4,
							-dy/abs(dy)*14 if dy>0 else 4),textcoords = 'offset pixels')]
		self.lines += [self.ax.annotate('(%.2f,%.2f)'%(self.pointer_loc[1,0],self.pointer_loc[1,1]),
						self.pointer_loc[1,:],
						xytext = (dx/abs(dx)*40 if dx < 0 else 4,
							dy/abs(dy)*14 if dy<0 else 4),textcoords = 'offset pixels')]
		self.lines += [self.ax.annotate('%.2f$\degree$'%sml_ang,
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
		# print(self.pointer_loc)
		# print('dx = %.2f'%(dx))
		# print('dy = %.2f'%(dy))
		# print('dr = %.2f'%L)
		# print('Elev: %.2f'%(sml_ang))


	def disconnect(self):
		self.fig.canvas.mpl_disconnect(self.cidpress)
		self.fig.canvas.mpl_disconnect(self.cidrelease)
		self.fig.canvas.mpl_disconnect(self.cidmotion)


class measure:
	lock = None #  only one can be animated at a time
	def __init__(self,fig,ax):
		self.click_num = 0
		self.fig = fig
		self.ax = ax
		self.pointer_loc = np.zeros((2,2))
		self.lines = []
		self.txt = []
		self.tot_lines = []
		self.tot_txt = []
		self.tot_pts = []
		self.pts = []
		self.grab = [False,False]
		self.multi_grab = [False,False]
		self.active = True
		self.pick_pt = -1

	def connect(self):
		self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
		self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
		return(self)
		# input('')
	def disconnect(self):
		self.fig.canvas.mpl_disconnect(self.cidpress)
		self.fig.canvas.mpl_disconnect(self.cidrelease)
		self.fig.canvas.mpl_disconnect(self.cidmotion)
	
	def onclick(self,event):

		if plt.get_current_fig_manager().toolbar.mode != '': return

		self.grab = [pt.contains(event)[0] for pt in self.pts]

		self.multi_grab = []
		for pts in self.tot_pts:
			self.multi_grab.append([pt.contains(event)[0] for pt in pts])

		if self.active == False: return
		
		if event.ydata != None and event.dblclick == True:
			self.click_num = 0
			self.grab = [False,False]

			self.multi_grab = [[False,False]]*(len(self.tot_pts) if len(self.tot_pts)>1 else 1)

			self.pts = []
			self.pointer_loc = np.zeros((2,2))

		elif event.ydata == None and event.dblclick == True:
			self.tot_clear()
			self.click_num = 0
			self.grab = [False,False]

			self.multi_grab = [[False,False]]*(len(self.tot_pts) if len(self.tot_pts)>1 else 1)
			return

		if event.ydata != None and event.xdata != None:
			if self.click_num == 0:

				self.grab = [False,False]
				self.multi_grab = [[False,False]]*(len(self.tot_pts) if len(self.tot_pts)>1 else 1)

				self.lines = []
				self.pts = [self.ax.plot(event.xdata,event.ydata,'+',color = 'k')[0],
					self.ax.plot(event.xdata,event.ydata,'+',color = 'k')[0]]

				for pt in self.pts:
					pt.set_xdata(event.xdata)
					pt.set_ydata(event.ydata)
				self.pointer_loc = np.array([[pt.get_xdata(),pt.get_ydata()] for pt in self.pts])
				self.fig.canvas.draw()
				self.fig.canvas.flush_events()

			elif self.click_num == 1:
				self.grab = [False,False]
				self.multi_grab = [[False,False]]*(len(self.tot_pts) if len(self.tot_pts)>1 else 1)
				self.pts[1].set_xdata(event.xdata)
				self.pts[1].set_ydata(event.ydata)
				self.pointer_loc = np.array([[pt.get_xdata(),pt.get_ydata()] for pt in self.pts])
				self.set_pts()
				self.get_locs()
				self.update_pts()
				self.tot_lines.append(self.lines)
				self.tot_pts.append(self.pts)
				self.tot_txt.append(self.txt)

			self.click_num += 1

		elif event.ydata == None or event.xdata == None:
			self.grab =  [False,False]
			self.multi_grab = [[False,False]]*(len(self.tot_pts) if len(self.tot_pts)>1 else 1)

	def on_motion(self,event):
		if np.any(self.multi_grab) and self.click_num > 1 and event.xdata!=None:
			grab = np.argwhere(self.multi_grab)[0]
			self.picked = grab[0]

			self.pts = self.tot_pts[grab[0]]
			self.txt = self.tot_txt[grab[0]]
			self.lines = self.tot_lines[grab[0]]

			self.pts[grab[1]].set_xdata(event.xdata)
			self.pts[grab[1]].set_ydata(event.ydata)
			self.pointer_loc = np.array([[pt.get_xdata(),pt.get_ydata()] for pt in self.pts])
			
			if np.any(self.multi_grab) and self.click_num > 1:
				self.get_locs()
				self.update_pts()

			self.fig.canvas.draw()
			self.fig.canvas.flush_events()

	def on_release(self,event):
		self.grab = [False,False]
		self.multi_grab = [[False,False]]*(len(self.tot_pts) if len(self.tot_pts)>1 else 1)
		

	def clear(self):
		if len(self.tot_lines)>0:
			for things in [self.lines,self.txt]:
				for lin in things.values():
					lin.remove()

			for pt in self.pts: pt.remove()


			for thing in [self.tot_lines,self.tot_txt,self.tot_pts]:
				del(thing[self.pick_pt])
			self.pick_pt = len(self.tot_lines)-1
			if self.pick_pt>=0:
				self.lines = self.tot_lines[self.pick_pt]
				self.txt = self.tot_txt[self.pick_pt]
				self.pts = self.tot_pts[self.pick_pt]
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()

	def tot_clear(self):
		for stuff in [self.tot_lines,self.tot_txt]:
			for things in stuff:
				for lin in things.values():
					lin.remove()
		for pts in self.tot_pts:
			for pt in pts: pt.remove()
		self.tot_lines = []
		self.tot_txt = []
		self.tot_pts = []
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

	def set_pts(self):
		linz = {}
		txt = {}

		org = [self.pointer_loc[1,0],self.pointer_loc[0,1]]
		L = np.sqrt(np.sum((self.pointer_loc - org)**2))
		dx = self.pointer_loc[1,0]-self.pointer_loc[0,0]
		dy = self.pointer_loc[1,1]-self.pointer_loc[0,1]
		ang = 180/np.pi*np.arctan2(dy,dx)
		sml_ang = 180/np.pi*np.arctan2(dy,np.abs(dx))
		guides = np.stack((self.pointer_loc[0,:],org,self.pointer_loc[1,:]))
		
		linz['hyp'] = self.ax.plot(self.pointer_loc[:,0],self.pointer_loc[:,1],color = 'k')[0]
		linz['guides'] = self.ax.plot(guides[:,0],guides[:,1],color = 'k',linestyle = 'dashed')[0]


		txt['dx'] = self.ax.annotate('',(0,0),
						horizontalalignment = 'center')
		txt['dy'] = self.ax.annotate('',(0,0),
						rotation = 'vertical',verticalalignment = 'center')
		txt['p1'] = self.ax.annotate('',(0,0))		
		txt['p2'] = self.ax.annotate('',(0,0))
		txt['hyp'] = self.ax.annotate('',(0,0),
						rotation = (ang if dx>0 else ang-180),
						rotation_mode = 'anchor',
						verticalalignment = 'center',
						horizontalalignment = 'center')
		txt['angl'] = self.ax.annotate('',(0,0),
						horizontalalignment = 'center',
						verticalalignment = 'center')

		arc_angs = np.sort([ang,ang-sml_ang*dx/abs(dx)])
		ang_draw = patches.Arc((self.pointer_loc[0,0],self.pointer_loc[0,1]),
			width = abs(dx)**.8,height = abs(dx)**.8,theta1=arc_angs[0],theta2=arc_angs[1])
		linz['ang'] = self.ax.add_patch(ang_draw)

		self.lines = linz
		self.txt = txt

	def set_patches(self):
		self.lines['ang'].remove()

		org = [self.pointer_loc[1,0],self.pointer_loc[0,1]]
		L = np.sqrt(np.sum((self.pointer_loc - org)**2))
		dx = self.pointer_loc[1,0]-self.pointer_loc[0,0]
		dy = self.pointer_loc[1,1]-self.pointer_loc[0,1]
		ang = 180/np.pi*np.arctan2(dy,dx)
		sml_ang = 180/np.pi*np.arctan2(dy,np.abs(dx))
		guides = np.stack((self.pointer_loc[0,:],org,self.pointer_loc[1,:]))

		arc_angs = np.sort([ang,ang-sml_ang*dx/abs(dx)])
		ang_draw = patches.Arc((self.pointer_loc[0,0],self.pointer_loc[0,1]),
			width = abs(dx)**.8,height = abs(dx)**.8,theta1=arc_angs[0],theta2=arc_angs[1])
		self.lines['ang'] = self.ax.add_patch(ang_draw)

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()


	def get_locs(self):
		linz = {}
		txt = {}

		linz['hyp']  = {}
		linz['hyp']['x'] = self.pointer_loc[:,0]
		linz['hyp']['y'] = self.pointer_loc[:,1]

		org = [self.pointer_loc[1,0],self.pointer_loc[0,1]]
		L = np.sqrt(np.sum((self.pointer_loc - org)**2))
		dx = self.pointer_loc[1,0]-self.pointer_loc[0,0]
		dy = self.pointer_loc[1,1]-self.pointer_loc[0,1]
		ang = 180/np.pi*np.arctan2(dy,dx)
		sml_ang = 180/np.pi*np.arctan2(dy,np.abs(dx))
		guides = np.stack((self.pointer_loc[0,:],org,self.pointer_loc[1,:]))
		
		linz['guides'] = {}
		linz['guides']['x'] = guides[:,0]
		linz['guides']['y'] = guides[:,1]

		txt['dx'] = {}
		txt['dx']['txt'] = '%.2f'%dx
		txt['dx']['pos'] = [np.mean(self.pointer_loc[:,0]),org[1]]
		txt['dx']['xytext'] = [0,-dy/abs(dy)*14 if dy>0 else 4]
		txt['dx']['rotation'] = 0

		txt['dy'] = {}
		txt['dy']['txt'] = '%.2f'%dy
		txt['dy']['pos'] = [org[0],np.mean(self.pointer_loc[:,1])]
		txt['dy']['xytext'] =(dx/abs(dx)*14 if dx <0 else 4,0)
		txt['dy']['rotation'] = 'vertical'

		txt['p1'] = {}
		txt['p1']['txt'] = '(%.2f,%.2f)'%(self.pointer_loc[0,0],self.pointer_loc[0,1])
		txt['p1']['pos'] = self.pointer_loc[0,:]
		txt['p1']['xytext'] = [-dx/abs(dx)*40 if dx > 0 else 4,
							-dy/abs(dy)*14 if dy>0 else 4]
		txt['p1']['rotation'] = 0 

		txt['p2'] = {}
		txt['p2']['txt'] = '(%.2f,%.2f)'%(self.pointer_loc[1,0],self.pointer_loc[1,1])
		txt['p2']['pos'] = self.pointer_loc[1,:]
		txt['p2']['xytext'] =[dx/abs(dx)*40 if dx < 0 else 4,
							dy/abs(dy)*14 if dy<0 else 4]
		txt['p2']['rotation'] = 0 

		txt['hyp'] = {}
		txt['hyp']['txt'] = '%.2f'%L
		txt['hyp']['pos'] = np.mean(self.pointer_loc,axis = 0)
		txt['hyp']['xytext'] =[-dx/abs(dx)*abs(10*dy/L),
								dy/abs(dy)*abs(10*dx/L)]
		txt['hyp']['rotation'] = (ang if dx>0 else ang-180)

		txt['angl'] = {}
		txt['angl']['txt'] = '%.2f$\degree$'%sml_ang
		# txt['angl']['pos'] = (np.mean(self.pointer_loc[:,0]),self.pointer_loc[0,1])
		txt['angl']['pos'] = [np.mean(self.pointer_loc[:,0]),np.mean([txt['hyp']['pos'][1],
						                                        txt['dx']['pos'][1]])]
		txt['angl']['xytext'] =[0,0]
		txt['angl']['rotation'] = 0

		self.line_locs = linz
		self.txt_locs = txt 

	def update_pts(self):
		for txts in self.txt_locs:
			ha = self.txt[txts].get_horizontalalignment()
			va = self.txt[txts].get_verticalalignment()
			rm = self.txt[txts].get_rotation_mode()

			self.txt[txts].set_text(self.txt_locs[txts]['txt'])
			self.txt[txts].xy = self.txt_locs[txts]['pos']
			self.txt[txts].set_anncoords('offset pixels')
			self.txt[txts].set_x(self.txt_locs[txts]['xytext'][0])
			self.txt[txts].set_y(self.txt_locs[txts]['xytext'][1])

			self.txt[txts].set_horizontalalignment(ha)
			self.txt[txts].set_verticalalignment(va)
			self.txt[txts].set_rotation_mode(rm)
			self.txt[txts].set_rotation(self.txt_locs[txts]['rotation'])

		for lins in self.line_locs:
			self.lines[lins].set_xdata(self.line_locs[lins]['x'])
			self.lines[lins].set_ydata(self.line_locs[lins]['y'])

		self.set_patches()

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()



class mark:
	def __init__(self,fig,ax,ref_lines= False):
		self.fig = fig
		self.ax = ax
		self.grab = [False]
		self.grab_txt = [False]
		self.pts = []
		self.pts_info = []
		self.info_lines = []
		self.x_lines = []
		self.y_lines = []
		self.ref_lines = ref_lines
		self.pick_pt = -1
		ax.plot(ax.get_xlim(),[0,0],color = 'black',alpha = .7)
		ax.plot([0,0],ax.get_ylim(),color = 'black',alpha = .7)
		self.active = True

	def connect(self):
		# print('Fig Measure connected:\nDouble Click outside Plot to Remove Marks')
		self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
		self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
		return(self)
		# input('')
	
	def onclick(self,event):
		self.grab = [pt.contains(event)[0] for pt in self.pts]
		self.grab_txt = [txt.contains(event)[0] for txt in self.pts_info]
		if plt.get_current_fig_manager().toolbar.mode != '': return
		if self.active == False: return
		
		if event.ydata != None and event.dblclick == True:
			self.grab = [False]
			pt = self.ax.plot(event.xdata,event.ydata,'+',color = 'k')[0]
			pt_info = self.ax.annotate('(%.2f,%.2f)'%(event.xdata,event.ydata),
						[event.xdata,event.ydata],
						xytext = (4,4),textcoords = 'offset pixels')
			if self.ref_lines == True:	
				x_line = self.ax.plot([0,event.xdata],[event.ydata,event.ydata],
				                      '--',color = 'black', alpha = .4)[0]
				y_line = self.ax.plot([event.xdata,event.xdata],[0,event.ydata],
				                      '--',color = 'black', alpha = .4)[0]
				self.x_lines.append(x_line)
				self.y_lines.append(y_line)

			self.pts.append(pt)
			self.pts_info.append(pt_info)
			self.info_lines.append(self.ax.plot([event.xdata]*2,
											   [event.ydata]*2,color = 'gray')[0])

			self.pointer_loc = np.array([[pt.get_xdata(),
										pt.get_ydata()] for pt in self.pts])
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
		elif event.dblclick == False and event.ydata==None:
			self.clear()

	def on_motion(self,event):
		if any(self.grab):
			pick_pt = np.argwhere(self.grab)[0][0] 
			self.pick_pt= pick_pt
			self.pts[pick_pt].set_xdata(event.xdata)
			self.pts[pick_pt].set_ydata(event.ydata)
			self.pts_info[pick_pt].remove()
			self.pts_info[pick_pt] = self.ax.annotate('(%.2f,%.2f)'%(event.xdata,event.ydata),
						[event.xdata,event.ydata],
						xytext = (4,4),textcoords = 'offset pixels')
			self.info_lines[pick_pt].set_xdata([event.xdata]*2)
			self.info_lines[pick_pt].set_ydata([event.ydata]*2)

			if self.ref_lines == True:
				self.x_lines[pick_pt].set_xdata([0,event.xdata])
				self.x_lines[pick_pt].set_ydata([event.ydata,event.ydata])

				self.y_lines[pick_pt].set_xdata([event.xdata,event.xdata])
				self.y_lines[pick_pt].set_ydata([0,event.ydata])
			
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()

		elif any(self.grab_txt):
			pick_pt =  np.argwhere(self.grab_txt)[0][0] 
			self.pick_pt= pick_pt
			self.pts_info[pick_pt].remove()
			dy = event.ydata-self.pts[pick_pt].get_ydata()

			self.pts_info[pick_pt] = self.ax.annotate(\
						'(%.1f,%.1f)'%(self.pts[pick_pt].get_xdata(),
										self.pts[pick_pt].get_ydata()),
						[event.xdata,event.ydata],
						verticalalignment = 'center',
						horizontalalignment = 'center',
						xytext = (0,(6 if dy >= 0 else -6)),textcoords = 'offset pixels')
			self.info_lines[pick_pt].set_xdata([self.pts[pick_pt].get_xdata(),
											  event.xdata])
			self.info_lines[pick_pt].set_ydata([self.pts[pick_pt].get_ydata(),
											  event.ydata])
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()


	def on_release(self,event):
		self.grab = [False]
		self.grab_txt=[False]
		

	def clear(self):
		if len(self.pts)>0:
			# for thing in [self.pts[-1],
			# 			 self.pts_info[-1],
			# 			 self.info_lines[-1]]:
			# 	thing.remove()
			# 	del(thing)
			self.pts[self.pick_pt].remove()
			del(self.pts[self.pick_pt])

			self.pts_info[self.pick_pt].remove()
			del(self.pts_info[self.pick_pt])
			
			self.info_lines[self.pick_pt].remove()
			del(self.info_lines[self.pick_pt])

			if self.ref_lines == True:
				self.y_lines[self.pick_pt].remove()
				del(self.y_lines[self.pick_pt])

				self.x_lines[self.pick_pt].remove()
				del(self.x_lines[self.pick_pt])
			self.pick_pt = len(self.pts)-1
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()

	def tot_clear(self):
		for pt,info,line in zip(self.pts,self.pts_info,self.info_lines):
			pt.remove()
			info.remove()
			line.remove()
		self.pts = []
		self.pts_info = []
		self.info_lines = []
		self.fig.canvas.draw()
		self.fig.canvas.flush_events()


	def disconnect(self):
		self.fig.canvas.mpl_disconnect(self.cidpress)
		self.fig.canvas.mpl_disconnect(self.cidrelease)
		self.fig.canvas.mpl_disconnect(self.cidmotion)

class label:
	def __init__(self,fig,ax,names,locs):
		self.fig = fig
		self.ax = ax
		self.grab = [False]
		self.grab_pt = [False]
		self.words = []
		self.lines = []
		self.points = []
		self.bbox_props =dict(boxstyle="round", fc="w", ec="0.5", alpha=0.6)
		for nam,loc in zip(names,locs):
			self.words.append(self.ax.annotate(nam,loc,ha= 'center',
											   bbox = self.bbox_props))
			self.lines.append(self.ax.plot([loc[0]]*2,[loc[1]]*2,color = 'gray')[0])
			self.points.append(self.ax.plot(loc[0],loc[1],'.',color = 'black')[0])

	def connect(self):
		print('Fig Measure connected:\nDouble Click outside Plot to Disconnect')
		self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
		self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
		self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
	
	def onclick(self,event):
		if plt.get_current_fig_manager().toolbar.mode != '': return
		
		self.grab = [word.contains(event)[0] for word in self.words]
		self.grab_pt = [point.contains(event)[0] for point in self.points]
		if event.ydata == None and event.dblclick == True:
			self.disconnect()
		
	def on_motion(self,event):
		if any(self.grab):
			pick_pt = np.argwhere(self.grab)[0][0]
			self.words[pick_pt].set_position([event.xdata,event.ydata])
			self.lines[pick_pt].set_ydata([self.points[pick_pt].get_ydata(),event.ydata])
			self.lines[pick_pt].set_xdata([self.points[pick_pt].get_xdata(),event.xdata])
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
		elif any(self.grab_pt):
			pick_pt = np.argwhere(self.grab_pt)[0][0]
			self.points[pick_pt].set_xdata(event.xdata)
			self.points[pick_pt].set_ydata(event.ydata)

			self.lines[pick_pt].set_ydata([self.words[pick_pt].get_position()[1],event.ydata])
			self.lines[pick_pt].set_xdata([self.words[pick_pt].get_position()[0],event.xdata])
			self.fig.canvas.draw()
			self.fig.canvas.flush_events()
	def on_release(self,event):
		self.grab = [False]
		self.grab_pt = [False]

	def disconnect(self):
		self.fig.canvas.mpl_disconnect(self.cidpress)
		self.fig.canvas.mpl_disconnect(self.cidrelease)
		self.fig.canvas.mpl_disconnect(self.cidmotion)
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



from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

class label:
    def __init__(self,fig,ax,names,locs,cols = [],alpha = .6):
        self.fig = fig
        self.ax = ax
        self.grab = [False]
        self.grab_pt = [False]
        self.words = []
        self.lines = []
        self.points = []
        self.bbox_props =dict(boxstyle="round", fc="w", ec="0.5", alpha=alpha)
        for nam,loc in zip(names,locs):
            self.words.append(self.ax.annotate(nam,loc,ha= 'center',
                                               bbox = self.bbox_props))
            self.lines.append(self.ax.plot([loc[0]]*2,[loc[1]]*2,color = 'gray')[0])
            self.points.append(self.ax.plot(loc[0],loc[1],'.',color = 'black')[0])
        self.active = True

    def connect(self):
        print('Fig Measure connected:\nDouble Click outside Plot to Disconnect')
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
    
    def onclick(self,event):
        if plt.get_current_fig_manager().toolbar.mode != '': return
        if self.active == False: return
        
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

    def clear(self,event):
        for line in self.lines:
            line.remove()
        for txt in self.words:
            txt.remove()
        for pt in self.points:
            pt.remove()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)
        print('We are Disconnected: Press Any key to continue')


def gem_label(gem_file,elec_names = [],origin = [0,0],
                  fig = [],ax = []):

    keys = np.array(list(electrodes.keys()))
    elec_center = {}

    for nam in electrodes:
        if len(xy)!=0:
            xy = np.concatenate(xy,axis = 0)
            elec_center[nam] = [xy[np.argmin(xy[:,1]),0],
            xy[np.argmin(xy[:,1]),1]]
        # np.mean(xy[:,0]),np.max(xy[:,1])]

    if annotate ==True:
        numbers = list(elec_center)
        locations = list(elec_center.values())
        bbox_props =dict(boxstyle="round", fc="w", ec="0.5", alpha=0.6)
        if elec_names ==[]:
            nams = numbers
        else:
            nams = [elec_names[num] for num in numbers]
        ax.label = meat.label(fig,ax,nams,locations)
        ax.label.connect()
    if path_out == False:
        return(fig,ax)
    else:
            return(fig,ax,electrodes)
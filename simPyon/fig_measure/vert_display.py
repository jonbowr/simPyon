from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np

class mark:
    def __init__(self,fig,ax,ref_lines= False):
        self.fig = fig
        self.ax = ax
        self.grab = [False]
        self.grab_txt = [False]
        self.pts = []
        for lin in ax.get_lines():
            self.pts.append(lin.get_xydata())
        self.pts = np.concatenate(self.pts)
        self.pts_info = []
        self.info_lines = []
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
                        '(%.2f,%.2f)'%(self.pts[pick_pt].get_xdata(),
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
            #            self.pts_info[-1],
            #            self.info_lines[-1]]:
            #   thing.remove()
            #   del(thing)
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
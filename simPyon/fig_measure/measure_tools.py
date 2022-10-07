from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from shapely import geometry as geo
from descartes import PolygonPatch
import numpy as np


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
    def __init__(self,fig,ax,ref_lines= False,ref_points = []):
        self.fig = fig
        self.ax = ax
        self.grab = [False]
        self.grab_txt = [False]
        
        self.pts = []
        self.pts_info = []
        self.info_lines = []

        # if ref_points:
        # self.ref_pts = ax.plot(ref_points[:,0],ref_points[:,1],'.')[0]


        # self.ref_pts = []
        # for xy in ref_points:
        #     # pt,pt_info,pt_line = self.set_pt(xy[0],xy[1],self.ax)
        #     # ax.plot(x,y,'+',color = 'k')[0]
        #     self.ref_pts.append(ax.plot(xy[0],xy[1],'.')[0])
            # self.pts_info.append(pt_info)
            # self.info_lines.append(pt_line)

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
    

    def set_pt(self,x,y,ax):
        pt = ax.plot(x,y,'+',color = 'k')[0]
        pt_info = ax.annotate('(%.2f,%.2f)'%(x,y),
                    [x,y],
                    xytext = (4,4),textcoords = 'offset pixels')
        info_line = ax.plot([y]*2,[y]*2,color = 'gray')[0]
        return(pt,pt_info,info_line)

    def onclick(self,event):
        self.grab = [pt.contains(event)[0] for pt in self.pts]
        self.grab_txt = [txt.contains(event)[0] for txt in self.pts_info]
        
        if plt.get_current_fig_manager().toolbar.mode != '': return
        if self.active == False: return
        

        # print(np.any(self.grab))
        # if self.ref_pts.contains(event)[0] and np.any(self.grab) == False:
        #     ind = self.ref_pts.contains(event)[1]['ind'][0]
        #     # print(self.ref_pts.contains(event)[1]['ind'][0])
        #     # print(self.ref_pts.get_xdata())
        #     x = self.ref_pts.get_xdata()[ind]
        #     y = self.ref_pts.get_ydata()[ind]
        # else:
        # self.click_ref = [pt.contains(event)[0] for pt in self.ref_pts]
        
        # if self.click_ref:
        #     pick_pt = np.argwhere(self.click_ref)[0][0] 
        #     x = self.ref_pts[pick_pt].get_xdata()
        #     y = self.ref_pts[pick_pt].get_ydata()
        # else:
        x = event.xdata
        y = event.ydata

        if event.ydata != None and event.dblclick == True:
            self.grab = [False]
            # pt = self.ax.plot(event.xdata,event.ydata,'+',color = 'k')[0]
            # pt_info = self.ax.annotate('(%.2f,%.2f)'%(event.xdata,event.ydata),
            #             [event.xdata,event.ydata],
            #             xytext = (4,4),textcoords = 'offset pixels')
            pt,pt_info,pt_line = self.set_pt(x,y,self.ax)
            if self.ref_lines == True:  
                x_line = self.ax.plot([0,x],[y,y],
                                      '--',color = 'black', alpha = .4)[0]
                y_line = self.ax.plot([X,X],[0,y],
                                      '--',color = 'black', alpha = .4)[0]
                self.x_lines.append(x_line)
                self.y_lines.append(y_line)

            self.pts.append(pt)
            self.pts_info.append(pt_info)
            # self.info_lines.append(self.ax.plot([event.xdata]*2,
            #                                    [event.ydata]*2,color = 'gray')[0])
            self.info_lines.append(pt_line)

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



class line_draw:
    def __init__(self,fig,ax):
        self.fig = fig
        self.ax = ax
        self.lines = []
        self.line_verts = []
        self.clickd = False
        self.active = True
        self.fatter = False
        self.fat_line = []
        self.poly_verts = []

    def connect(self):
        print('Fig Measure connected:\nDouble Click outside Plot to Disconnect')
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        return(self)

    def onclick(self,event):
        if plt.get_current_fig_manager().toolbar.mode != '': return
        if self.active==False: return

        if event.xdata != None and event.ydata !=None and event.dblclick == False:
            self.clickd = True
            self.lines.append(self.ax.plot(event.xdata,event.ydata)[0])
            self.line_verts.append([[event.xdata,event.ydata]])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        elif event.ydata == None and event.dblclick == True:
            self.disconnect()
        elif event.ydata != None and event.dblclick == True:
            def smooth(arr,itterations=1):
                for i in range(itterations):
                    smoot=np.zeros(len(arr))
                    smoot[0]=np.nanmean([arr[0],arr[1]])
                    smoot[1:-1]=np.nanmean(np.stack([arr[0:-2],arr[1:-1],arr[2:]]),axis = 0)
                    smoot[-1]=np.nanmean([arr[-1],arr[-2]])
                    arr=smoot
                return(arr)
            d = 2
            w = 1
            pts_mm = 5
            from scipy.interpolate import interp1d
            if self.fatter ==False:

                self.fatter = True
                line = self.lines[0]
                line.set_solid_capstyle('round')
                line_verts = np.stack(line.get_data()).T
                poly_verts = np.concatenate((line_verts,np.flipud(line_verts)[1:]))

                self.fat_line = self.ax.add_patch(patches.Polygon(poly_verts))
                # self.fat_line = self.ax.plot(poly_verts[:,0],poly_verts[:,1])
                self.fig.canvas.draw()
                # check = '1'
                
                poly_verts[:,0] = smooth(poly_verts[:,0],2)
                poly_verts[:,1] = smooth(poly_verts[:,1],2)
                self.poly_verts = poly_verts
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                # qeqFREQFGRS
            poly_verts = self.poly_verts

            L = np.zeros(len(poly_verts)) 

            L[1:] = np.cumsum(np.sqrt(np.sum((poly_verts[1:,:]-poly_verts[:-1,:])**2,
                                         axis = 1)))
            fx = interp1d(L,poly_verts[:,0],kind = 'cubic')
            fy = interp1d(L,poly_verts[:,1],kind = 'cubic')
            # f_got = interp1d(L,got_edge,kind = 'linear')

            poly_verts = np.zeros((int(L[-1]*pts_mm),2))
            # # poly_verts = np.zeros((len(og_verts)*w,2))
            poly_verts[:,0] = fx(np.linspace(min(L),max(L),len(poly_verts)))
            poly_verts[:,1] = fy(np.linspace(min(L),max(L),len(poly_verts)))

            rec_verts = np.zeros((len(poly_verts)+2,poly_verts.shape[1]))
            rec_verts[1:-1,:] = poly_verts[:,:]
            rec_verts[0,:] = poly_verts[-2,:]
            rec_verts[-1,:] = poly_verts[1,:]

            poly_delt = ((rec_verts[2:,:]-rec_verts[:-2,:])+
                     (rec_verts[1:-1,:]-rec_verts[:-2,:])+
                     (rec_verts[2:,:]-rec_verts[1:-1,:]))/3

            ang = np.arctan2(poly_delt[:,1],poly_delt[:,0])
            dx = np.sin(ang)*d
            dy = np.cos(ang)*d
            poly_verts[:,0] += -dx
            poly_verts[:,1] += dy

            over = self.fat_line.get_path().contains_points(poly_verts)
            poly_verts = poly_verts[~over]

            self.fat_line.set_xy(poly_verts)
            self.poly_verts = poly_verts
            self.ax.plot(self.poly_verts[:,0],self.poly_verts[:,1])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


    def on_motion(self,event):
        if self.clickd:

            self.line_verts[-1].append([event.xdata,event.ydata])
            self.lines[-1].set_xdata(np.array(self.line_verts[-1])[:,0])
            self.lines[-1].set_ydata(np.array(self.line_verts[-1])[:,1])
            self.fig.canvas.draw()
    
    def on_release(self,event):
        self.clickd = False

    def tot_clear(self):
        for lin in self.lines:
            lin.remove()
        self.lines = []
        self.line = self.ax.plot([0,0],[0,0])[0]
        self.line_verts = []
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def clear(self):
        self.lines[-1].remove()
        self.lines = self.lines[:-1]
        self.line_verts = self.line_verts[:-1]
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)
        self.active = False
        print('We are Disconnected: Press Any key to continue')





class show_pts:
    def __init__(self,fig,ax,ref_lines= False):
        self.fig = fig
        self.ax = ax
        self.grab = [False]
        self.grab_txt = [False]
        
        self.pts = []
        self.pts_info = []
        self.info_lines = []

        # if ref_points:
        self.ref_pts = ax.get_lines()[0]#ax.plot(ref_points[:,0],ref_points[:,1],'.')[0]
        # self.ref_pts = []
        # for xy in ref_points:
        #     # pt,pt_info,pt_line = self.set_pt(xy[0],xy[1],self.ax)
        #     # ax.plot(x,y,'+',color = 'k')[0]
        #     self.ref_pts.append(ax.plot(xy[0],xy[1],'.')[0])
            # self.pts_info.append(pt_info)
            # self.info_lines.append(pt_line)

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
    

    def set_pt(self,x,y,ax):
        pt = ax.plot(x,y,'+',color = 'k')[0]
        pt_info = ax.annotate('(%.2f,%.2f)'%(x,y),
                    [x,y],
                    xytext = (4,4),textcoords = 'offset pixels')
        info_line = ax.plot([y]*2,[y]*2,color = 'gray')[0]
        return(pt,pt_info,info_line)

    def onclick(self,event):
        self.grab = [pt.contains(event)[0] for pt in self.pts]
        self.grab_txt = [txt.contains(event)[0] for txt in self.pts_info]
        
        if plt.get_current_fig_manager().toolbar.mode != '': return
        if self.active == False: return
        

        # print(np.any(self.grab))
        if self.ref_pts.contains(event)[0] and np.any(self.grab)==False:
            ind = self.ref_pts.contains(event)[1]['ind'][0]
            # print(self.ref_pts.contains(event)[1]['ind'][0])
            # print(self.ref_pts.get_xdata())
            x = self.ref_pts.get_xdata()[ind]
            y = self.ref_pts.get_ydata()[ind]
        # else:
        # self.click_ref = [pt.contains(event)[0] for pt in self.ref_pts]
        
        # if self.click_ref:
        #     pick_pt = np.argwhere(self.click_ref)[0][0] 
        #     x = self.ref_pts[pick_pt].get_xdata()
        #     y = self.ref_pts[pick_pt].get_ydata()
        # else:
            # x = event.xdata
            # y = event.ydata

        # if self.ref_pts.contains(event)[0] and np.any(self.grab) == False:
            # self.grab = [False]
            # pt = self.ax.plot(event.xdata,event.ydata,'+',color = 'k')[0]
            # pt_info = self.ax.annotate('(%.2f,%.2f)'%(event.xdata,event.ydata),
            #             [event.xdata,event.ydata],
            #             xytext = (4,4),textcoords = 'offset pixels')
            pt,pt_info,pt_line = self.set_pt(x,y,self.ax)
            if self.ref_lines == True:  
                x_line = self.ax.plot([0,x],[y,y],
                                      '--',color = 'black', alpha = .4)[0]
                y_line = self.ax.plot([X,X],[0,y],
                                      '--',color = 'black', alpha = .4)[0]
                self.x_lines.append(x_line)
                self.y_lines.append(y_line)

            self.pts.append(pt)
            self.pts_info.append(pt_info)
            # self.info_lines.append(self.ax.plot([event.xdata]*2,
            #                                    [event.ydata]*2,color = 'gray')[0])
            self.info_lines.append(pt_line)

            self.pointer_loc = np.array([[pt.get_xdata(),
                                        pt.get_ydata()] for pt in self.pts])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        elif event.dblclick == False and event.ydata==None:
            self.clear()

    def on_motion(self,event):

        if any(self.grab_txt):
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
        # self.grab = [False]
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

class find_surface:
        def __init__(self,fig,ax,geom):
            self.fig = fig
            self.ax = ax
            self.lines = []
            self.line_verts = []
            self.clickd = False
            self.active = True
            self.fatter = False
            self.fat_line = []
            self.fat_patch = []
            self.poly_verts = []
            self.geom = geom
            self.w = .5
            self.g_verts = []

        def connect(self):
            print('Surface Finder connected:\nDouble Click outside Plot to Disconnect')
            print('Draw line, Double click within plot to grow')
            print('Selected Vertices determined on disconnect')
            self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
            self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
            return(self)

        def onclick(self,event):
            if plt.get_current_fig_manager().toolbar.mode != '': return
            if self.active==False: return

            if event.xdata != None and event.ydata !=None and event.dblclick == False:
                self.clickd = True
                self.lines.append(self.ax.plot(event.xdata,event.ydata)[0])
                self.line_verts.append([[event.xdata,event.ydata]])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            elif event.ydata == None and event.dblclick == True:
                self.disconnect()
            elif event.ydata != None and event.dblclick == True:
                def smooth(arr,itterations=1):
                    for i in range(itterations):
                        smoot=np.zeros(len(arr))
                        smoot[0]=np.nanmean([arr[0],arr[1]])
                        smoot[1:-1]=np.nanmean(np.stack([arr[0:-2],arr[1:-1],arr[2:]]),axis = 0)
                        smoot[-1]=np.nanmean([arr[-1],arr[-2]])
                        arr=smoot
                    return(arr)

                
                pts_mm = 5
                from scipy.interpolate import interp1d
                if self.fatter ==False:

                    self.fatter = True
                    line = self.lines[0]
                    line.set_solid_capstyle('round')
                    line_verts = np.stack(line.get_data()).T
                    # poly_verts = np.concatenate((line_verts,np.flipud(line_verts)[1:]))

                    self.fat_line = geo.LineString(self.line_verts[0])
                    
                    self.fat_line = self.fat_line.buffer(self.w)
                    self.fat_line = self.fat_line.difference(self.geom)
                    if type(self.fat_line) == geo.multipolygon.MultiPolygon:
                            self.fat_line = self.fat_line[0]
                    self.poly_verts = np.array(self.fat_line.exterior.coords.xy).T
                    self.fat_patch = self.ax.add_patch(Polygon(self.poly_verts))

                else:
                    try:
                        self.fat_line = self.fat_line.buffer(self.w)
                        self.fat_line = self.fat_line.difference(self.geom)
                        if type(self.fat_line) == geo.multipolygon.MultiPolygon:
                            self.fat_line = self.fat_line[0]
                        poly_verts = np.array(self.fat_line.exterior.coords.xy).T
                        self.fat_patch.set_xy(poly_verts)
                    except: pass

                self.poly_verts = poly_verts
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()


        def on_motion(self,event):
            if self.clickd:

                self.line_verts[-1].append([event.xdata,event.ydata])
                self.lines[-1].set_xdata(np.array(self.line_verts[-1])[:,0])
                self.lines[-1].set_ydata(np.array(self.line_verts[-1])[:,1])
                self.fig.canvas.draw()
        
        def on_release(self,event):
            self.clickd = False

        def tot_clear(self):
            for lin in self.lines:
                lin.remove()
            self.lines = []
            self.line = self.ax.plot([0,0],[0,0])[0]
            self.line_verts = []
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        def clear(self):
            self.lines[-1].remove()
            self.lines = self.lines[:-1]
            self.line_verts = self.line_verts[:-1]
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        def disconnect(self):
            self.fig.canvas.mpl_disconnect(self.cidpress)
            self.fig.canvas.mpl_disconnect(self.cidrelease)
            self.fig.canvas.mpl_disconnect(self.cidmotion)
            self.active = False
            try:
                self.g_verts = np.concatenate([
                                    np.concatenate([
                                        np.array(c.coords),
                                        np.ones([1,2])*np.nan],
                                        axis = 0)
                                 for c in geo.LineString(self.poly_verts).intersection(self.geom.buffer(self.w/100))])
            except: pass
            self.ax.plot(*self.g_verts.T,'.')
            print('We are Disconnected: Press Any key to continue')
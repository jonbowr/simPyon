import numpy as np
import os
import skimage as skim
from skimage import feature
from skimage import measure
from matplotlib import pyplot as plt
from matplotlib import widgets
from shapely.geometry import Polygon as spoly
from shapely.geometry import MultiPolygon as mpoly
from matplotlib.patches import Polygon,Ellipse
from matplotlib.collections import PatchCollection
from matplotlib import path
from mpl_toolkits import mplot3d
import time
from . import fig_measure as meat
from matplotlib import style

def log_grad(image):
    grad = np.abs(np.gradient(image)) 
    return((grad[0]+grad[1]).astype(bool))

def pathfind(image,vertex,end_vert):
    path_img = np.zeros(image.shape)
    path_img[vertex[0],vertex[1]] = 1
    i=0
    while np.all(path_img[end_vert[:,0],end_vert[:,1]].astype(bool) == False):
        path_img += log_grad(path_img)
        path_img = np.logical_and(path_img,image).astype(int)
        i+=1
    vert_loc = np.argwhere(path_img[end_vert[:,0],end_vert[:,1]].astype(bool) == True)
    next_vert = end_vert[vert_loc]
    return(next_vert,vert_loc)

class part_group:
    def __init__(self,part_img):
        self.click_num = 0
        self.fig,self.ax = plt.subplots()
        self.ax.imshow(part_img,origin = 'lower')
        self.img = part_img
        self.og_parts = np.unique(part_img)
        self.group_parts = np.unique(part_img)
        self.part_num = 0

    def connect(self):
        print('Part Group connected:\nDouble Click outside Plot to Disconnect')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        input('')
    
    def onclick(self,event):
        if event.ydata == None and event.dblclick == True:
            # return([self.og_parts,self.group_parts])
            self.disconnect()
        elif event.ydata != None and event.xdata != None:
            if self.click_num % 2 == 0:
                self.part_num = self.img[int(event.ydata), int(event.xdata)]
                print('Electrode %d selected' % self.part_num ) 
                # self.text_box.label = 'Part Grouping: %f'%self.part_num
                self.click_num += 1
                print(self.click_num)
            else:
                print('Electrode %d grouped to %d' % (self.img[int(event.ydata),
                                     int(event.xdata)],self.part_num) )
                self.group_parts[self.group_parts == self.img[int(event.ydata), int(event.xdata)]] = self.part_num
                self.img[self.img == self.img[int(event.ydata), int(event.xdata)]] = self.part_num

                self.ax.imshow(self.img,origin ='lower')
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                self.click_num +=1

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        self.part_groups = np.transpose(np.stack((self.og_parts,self.group_parts)))[1:,:]
        plt.close(self.fig)
        print(np.stack((self.og_parts,self.group_parts)))
        print('We are Disconnected: Press Any key to continue')

    # def submit(self,text):
    #   print(text)
        # self.group_dict[group] = text


class part_name:
    def __init__(self,part_img):
        self.fig,self.ax = plt.subplots()
        self.ax.imshow(part_img)
        self.img = part_img
        self.part_names = {}

    def connect(self):
        print('Part Group connected:\nDouble Click outside Plot to Disconnect')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        input('')
    
    def onclick(self,event):
        if event.ydata == None and event.dblclick == True:
            self.disconnect()
        elif event.ydata != None and event.xdata != None:
            part_num = self.img[int(event.ydata), int(event.xdata)]
            print('Electrode %d selected' % part_num ) 
            self.part_names[part_num] = input('input name:')
            
    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cid)
        self.part_groups = np.transpose(np.stack((self.og_parts,self.group_parts)))[1:,:]
        plt.close(self.fig)
        for part in part_names:
            print("Electode %d Named: %s"%(part,part_names[part]))
        print('We are Disconnected: Press Any key to continue')

def bmp_get_verts(image,gaus_w = .8):
    vert_coords = feature.corner_peaks(feature.corner_harris(image,method = 'eps',eps = 2))
    coords_subpix = feature.corner_subpix(image, vert_coords,alpha = .1)
    img_outline,parts = measure.label(feature.canny(image),connectivity = 2,return_num = True)
    img_parts = measure.label(image)
    grouping = part_group(img_parts)
    grouping.connect()
    part_groups = grouping.part_groups
    input('Hit any key once grouping is complete')
    sorted_verts = []
    for part in range(1,parts+1):
        print(part)
        part_img = np.zeros(img_outline.shape)
        part_img[img_outline == part] = 1
        part_img = skim.filters.gaussian(part_img,sigma = gaus_w,preserve_range = True)
        sml_vert = vert_coords[part_img[vert_coords[:,0],vert_coords[:,1]] != 0]
        srt_pts = np.zeros(sml_vert.shape).astype(int)
        srt_pts[0,:] =  sml_vert[0,:]
        sml_vert = np.delete(sml_vert,0,axis = 0)
        for n in range(len(sml_vert)):
            dr = srt_pts[n,:] - sml_vert
            r_srt = np.argsort(np.sqrt(np.sum(dr**2,axis = 1)))
            m = 0
            line_h,line_l = skim.draw.line(srt_pts[n,0],srt_pts[n,1],
                                           sml_vert[r_srt[m],0],sml_vert[r_srt[m],1])
            line_sum = []
            while(np.any(part_img[line_h,line_l] == 0) and m+1 < len(sml_vert)):
                m += 1
                line_h,line_l = skim.draw.line(srt_pts[n,0],srt_pts[n,1],
                                               sml_vert[r_srt[m],0],sml_vert[r_srt[m],1])
            if m == len(sml_vert)-1 and len(sml_vert)-1 != 0:
                m = pathfind(part_img,srt_pts[n,:],sml_vert[r_srt])[1]
                print(m)
                print(len(sml_vert))
            srt_pts[n+1,:] = sml_vert[r_srt[m],:]
            sml_vert = np.delete(sml_vert,r_srt[m],axis = 0)
        sorted_verts += [srt_pts]
    return(sorted_verts,part_groups)

def clean_verts(sorted_verts):
    small_verts = []
    for part_vert in sorted_verts:
        good_vert = np.array([part_vert[0,:]])
        m=0
        for n in range(1,len(part_vert)-1):
            line_h,line_l = skim.draw.line(good_vert[m,0],good_vert[m,1],part_vert[n+1,0],part_vert[n+1,1])
            if np.any(np.all(np.transpose(np.array([line_h,line_l])) == part_vert[n,:],axis = 1)):
                pass
            else:
                good_vert = np.concatenate([good_vert,[part_vert[n,:]]],axis = 0)
                m += 1
                # print(good_vert)
        good_vert = np.concatenate([good_vert,[part_vert[-1,:]]],axis = 0)
        small_verts += [good_vert]
    return(small_verts)

def print_verts(vert_parts,file_nam, scale = 1,shift = 0,
                part_groups = [],part_lable = []):
    if part_groups == []:
        part_groups = np.stack([np.arange(len(part_groups))]*2)
    if part_lable == []:
        part_lable = dict((i,'') for i in len(np.unique(part_groups)))
        # part_lable = ['']*len(np.unique(part_groups))
    with open(file_nam, 'w') as gemfil:
        for i in part_lable:
            gemfil.write('\n;' + part_lable[i] + '\n')
            gemfil.write('electrode('+str(i)+ ')\n{\n')
            gemfil.write('fill{\n')
            for verts in [vert_parts[j[0]] for j in np.argwhere(part_groups[:,1]==i)]:
                vert_string = np.array2string(np.concatenate(np.fliplr(verts*scale+shift)), 
                                separator = ',',formatter={'float_kind':lambda x: "%.1f" % x},threshold = len(verts)*2)[1:-1]
                gemfil.write('within{polyline('+vert_string+')}\n')
            gemfil.write('}\n}\n')
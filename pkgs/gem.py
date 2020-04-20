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
# style.use('default')
# import plotly as pltly
plt.ion()

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

def find_all(a_str, sub):
    start = 0
    locs = []
    while True:
        start = a_str.find(sub, start)
        if start == -1: return locs
        locs = locs + [start]
        start += len(sub) # use start += 1 to find overlapping matches

def com_type(line):
    if line.lower().count('box(') != 0:
        draw_type = 'box'
    elif line.lower().count('polyline(') != 0:
        draw_type = 'polyline'    
    elif line.lower().count('circle(') != 0:
        draw_type = 'circle'
    else: draw_type = 'none'
    return draw_type

def draw_style(line):
    if line.lower().count('within{') != 0:
        draw_type = 'in'
    elif line.lower().count('notin{') != 0:
        draw_type = 'not_in'
    else: draw_type = 0
    return draw_type
        
def within(line , brac_open, brac_close):
    return line[line.find(brac_open)+len(brac_open):line.find(brac_close)]
    
def relocate(line,loc_pos):
    draw_type = com_type(line)  
    if draw_type == 'none':
        return line     
    elif draw_type == 'box' or draw_type == 'polyline':
        old_pos = np.fromstring(within(line,'(',')'),sep =',')
        old_pos = old_pos.reshape(int(len(old_pos)/2),2)
        new_pos = old_pos + loc_pos
        new_pos = new_pos.reshape(new_pos.size)
    elif draw_type == 'circle':        
        old_pos = np.fromstring(within(line,'(',')'),sep =',')
        new_pos = old_pos
        new_pos[:2] = old_pos[:2] + loc_pos
    return (line.replace(within(line,'(',')'),np.array2string(new_pos, 
                        separator = ',',formatter={'float_kind':lambda x: "%.1f" % x})[1:-1].replace(' ','').replace('\n','\n      ')))

def get_vtex(line,draw_type = []):
    if draw_type == []:
        draw_type = com_type(line)  
    if draw_type == 'none':
        return     
    elif draw_type == 'polyline':
        pos = np.fromstring(within(line,'(',')'),sep =',')

        # pos = within(line,'(',')').split(',')
        # p2 = []
        # for p in pos:
        #     p2.append(p.split(' '))
        # print(p2)
        # pos = np.array(p2).astype(float)
        pos = pos.reshape(-1,2)
    elif draw_type == 'box':
        pos = np.zeros((4,2))


        # pos_sim = within(line,'(',')').split(',')
        # p2 = []
        # for p in pos_sim:
        #     p2.append(p.split(' '))
        # pos_sim = np.array(p2).astype(float)

        pos_sim = np.fromstring(within(line,'(',')'),sep =',')

        pos_sim = pos_sim.reshape(-1,2)
        pos[:,0] = np.array([pos_sim[0,0],pos_sim[0,0],pos_sim[1,0],pos_sim[1,0]])
        pos[:,1] = np.array([pos_sim[0,1],pos_sim[1,1],pos_sim[1,1],pos_sim[0,1]])
    elif draw_type == 'circle':        
        pos = np.fromstring(within(line,'(',')'),sep =',')

        # pos = within(line,'(',')').split(',')
        # p2 = []
        # for p in pos:
        #     p2.append(p.split(' '))
        # pos = np.array(p2).flatten().astype(float)
        # pos = Ellipse([vals[0],vals[1]],vals[2],vals[3]).get_path().vertices

    return (pos)

def draw_from_gem(line,canvas_shape,pxls_mm = 1):
    draw_type = com_type(line)
    v_tex = (get_vtex(line)*pxls_mm).astype(int)
    # if draw_type == 'none':
    #     img = np.zeros(canvas_shape)    
    img = np.zeros(canvas_shape)
    if draw_type == 'polyline':
        locs = skim.draw.polygon(v_tex[:,0],v_tex[:,1],shape = canvas_shape)
        img[locs[0],locs[1]] = 1
    if draw_type == 'box':
        locs = skim.draw.polygon(v_tex[:,0],v_tex[:,1],shape = canvas_shape)
        # locs = skim.draw.rectangle(v_tex[0,:],v_tex[1,:],shape = canvas_shape)
        img[locs[0],locs[1]] = 1
    elif draw_type == 'circle':        
        locs = skim.draw.ellipse(v_tex[0],v_tex[1],v_tex[2],v_tex[3],shape = canvas_shape)
        img[locs[0],locs[1]] = 1
    return (img.astype(bool))

def next_up(str_list, start, stop):
        z = 0
        line = str_list[0]
        l = line.lower().find(start)
        char = line[l]
        while char != stop:
            l += 1
            if l == len(line):
                z += 1
                line = str_list[z]
                l = 0
            char = line[l]
        return (z,l)

def gem_draw_og(gem_file, canvas = [],pxls_mm = None,plot = True):
    with open(gem_file) as lines:
        file_lines = lines.readlines()

    reloc = []
    new_lines = ['']*len(file_lines)

    i = 0
    # clean the file of comments,spaces and black lines
    for line in file_lines:
        line = line[:line.find(';')].strip('\n').strip(' ')
        if line != '':
            if line.find(',') !=-1 and line[-1] ==',':
                new_lines[i]+=line.strip(' ')
            else:
                new_lines[i]+=line.strip(' ')
                i+=1
    for line,n in zip(new_lines,range(len(new_lines))):
        if line == '':
            del(new_lines[i])

    elec_count = 0
    # electrodes = []
    n = 0
    # pxls_mm = None
    for line in new_lines:
        if line.replace(' ','').lower().count('pa_define') != 0:
            canvas_size = np.fromstring(within(line,'(',',1'),sep =',')[:2].astype(int)  
            canvas = np.zeros(canvas_size).astype(bool)
        if line.replace(' ','').lower().count('locate(') != 0 and pxls_mm == None:
            pxls_mm = np.fromstring(within(line,'(',')'),sep =',')[-1]

        if line.replace(' ','').lower().count('fill') != 0 and line.replace(' ','').lower().count('rotate_fill') == 0:
            electrode = np.zeros(canvas.shape)
            # identify the position in x,y of the locate call
            
            op_brac = line[line.lower().find('fill'):].count('{')
            close_brac = 0
            m = 0

            while op_brac > close_brac or op_brac == 0:
                sub_line = new_lines[n+m]       
                op_brac = op_brac + sub_line.count('{')
                close_brac = close_brac + sub_line.count('}')
                # try:
                if draw_style(sub_line) == 'in':
                    electrode += draw_from_gem(sub_line,canvas.shape,pxls_mm)
                elif draw_style(sub_line) == 'not_in':
                    electrode[draw_from_gem(sub_line,canvas.shape,pxls_mm)] = False
                # except(TypeError):
                #     print(sub_line)
                # if draw_style(sub_line) == 'in':
                #     electrodes[elec_count] += draw_from_gem(sub_line,canvas.shape,pxls_mm)
                # elif draw_style(sub_line) == 'not_in':
                #     electrodes[elec_count][draw_from_gem(sub_line,canvas.shape,pxls_mm)] = False
                    # np.logical_or(electrodes[elec_count],
                    #                 ~draw_shapes(sub_line,canvas.shape,pxls_mm))
                m += 1
            canvas = np.logical_or(canvas,electrode)
            elec_count += 1
        n += 1

    # for elec in electrodes:
    #     canvas = np.logical_or(canvas,elec)
    if plot == True:
        plt.imshow(~np.transpose(canvas),origin = 'lower')
        plt.show()
    return(~np.transpose(canvas))


def gem_draw(gem_file, canvas = [],pxls_mm = None,plot = True):
 
    elec_verts,exclude_verts = get_verts(gem_file)

    canvas_size,pxls_mm = get_canvas(gem_file)

    canvas = np.zeros(canvas_size).astype(bool)
    electrodes = {}
    excludes = {}
    for nam in elec_verts:
        electrode = np.zeros(canvas.shape)
        for part in elec_verts[nam]:
            locs = skim.draw.polygon((part[:,0]*pxls_mm).astype(int),
                                     (part[:,1]*pxls_mm).astype(int),
                                     shape = canvas.shape)
            electrode[locs[0],locs[1]] = True
        for ex in exclude_verts[nam]:
            
            locs = skim.draw.polygon((ex[:,0]*pxls_mm).astype(int),
                                     (ex[:,1]*pxls_mm).astype(int),
                                     shape = canvas.shape)
            electrode[locs[0],locs[1]] = False
        canvas = np.logical_or(canvas,electrode)

    if plot == True:
        plt.imshow(~np.transpose(canvas),origin = 'lower')
        plt.show()
    return(canvas)


def gem_draw_poly(gem_file,measure = False,
                  mark=False,
                  annotate = False,
                  elec_names = [],origin = [0,0],
                  path_out = False,
                  fig = [],ax = [],return_patch = False):
    from matplotlib import cm
    elec_verts,exclude_verts = get_verts(gem_file)
    electrodes = {}
    excludes = {}
    for nam in elec_verts:
        electrodes[nam] = []
        excludes[nam] = []
        for part in elec_verts[nam]:
            electrodes[nam].append(Polygon(part-origin))
        for ex in exclude_verts[nam]:
            excludes[nam].append(Polygon(ex-origin,color = 'white'))
    
    patches = {}
    keys = np.array(list(electrodes.keys()))
    clip = {}
    elec_center = {}

    for nam in electrodes:
        patches[nam] = PatchCollection(electrodes[nam],alpha = 1)
        patches[nam].set_facecolor(cm.viridis(nam/max(keys)))
        clip[nam] = PatchCollection(excludes[nam])
        # ax.add_collection(patches[nam])
        xy = []
        for part in electrodes[nam]:
            xy.append(part.get_xy())
        if len(xy)!=0:
            xy = np.concatenate(xy,axis = 0)
            elec_center[nam] = [xy[np.argmin(xy[:,1]),0],
            xy[np.argmin(xy[:,1]),1]]
        # np.mean(xy[:,0]),np.max(xy[:,1])]
    # if ax_3d == True:
    #     from mpl_toolkits.mplot3d import art3d
    #     p = art3d.Patch3DCollection(patches,alpha = 1)
    #     # p.set_array(np.array(keys))
    #     ax.add_collection(p)
    #     # art3d.patch_collection_2d_to_3d(p)
    #     return(fig,ax)
    if return_patch == True:
        # from mpl_toolkits.mplot3d import Axes3D,art3d
        # for pat in patches.values():
        #     # print(pat)
        #     art3d.patch_collection_2d_to_3d(pat,zdir = 'y')
        #     ax.add_collection3d(pat)
        # plt.show()
        for el in elec_verts.values():
            for vt in el:
                ax.plot(vt[:,0],vt[:,1],np.zeros(len(vt)),color = 'black')
        return()

    if fig == [] and ax == []:
        fig,ax = plt.subplots()
    for nam in patches:
        ax.add_collection(patches[nam])
    ax.autoscale(enable = True)
    ax.autoscale(enable = False)
    for nam in patches:
        ax.add_collection(patches[nam])
        ax.add_collection(clip[nam])
        clip[nam].set_facecolor('white')
    
    ax.set_aspect('equal')
    plt.show()

    if measure == True:
        # import fig_measure as meat
        measur = meat.measure(fig,ax)
        measur.connect()
    elif mark==True:
        mrk = meat.mark(fig,ax)
        mrk.connect()
    elif annotate ==True:
        numbers = list(elec_center)
        locations = list(elec_center.values())
        bbox_props =dict(boxstyle="round", fc="w", ec="0.5", alpha=0.6)
        if elec_names ==[]:
            nams = numbers
        else:
            nams = [elec_names[num] for num in numbers]
        lab = meat.label(fig,ax,nams,locations)
        lab.connect()
    if path_out == False:
        return(fig,ax)
    else:
            return(fig,ax,electrodes)


def get_canvas(gem_file):
    with open(gem_file) as lines:
        file_lines = lines.readlines()

    reloc = []
    new_lines = ['']*len(file_lines)

    i = 0
    # clean the file of comments,spaces and black lines
    for line in file_lines:
        line = line[:line.find(';')].strip('\n').strip(' ')
        if line != '':
            if line.find(',') !=-1 and line[-1] ==',':
                new_lines[i]+=line.strip(' ')
            else:
                new_lines[i]+=line.strip(' ')
                i+=1
    for line,n in zip(new_lines,range(len(new_lines))):
        if line == '':
            del(new_lines[i])


    for line in new_lines:
        if line.lower()[:line.find(';')].find('pa_define') != -1:
            canvas_size = np.array(within(line,'(',')').split(',')[:2]).astype(int)
            # canvas_size = np.fromstring(within(line,'(',',1'),sep =',')[:2].astype(int) 
        if line.lower()[:line.find(';')].find('locate') != -1:
            pxls_mm = np.fromstring(within(line,'(',')'),sep =',')[-1]
            break
    return(canvas_size,pxls_mm)

def get_verts(gem_file):
    with open(gem_file) as lines:
        file_lines = lines.readlines()

    reloc = []
    new_lines = ['']*len(file_lines)

    i = 0
    # clean the file of comments,spaces and black lines
    for line in file_lines:
        line = line[:line.find(';')].strip('\n').strip(' ')
        if line != '':
            if line.find(',') !=-1 and line[-1] ==',':
                new_lines[i]+=line.strip(' ')
            else:
                new_lines[i]+=line.strip(' ')
                i+=1
    for line,n in zip(new_lines,range(len(new_lines))):
        if line == '':
            del(new_lines[i])

    elec_count = 0
    n = 0
    electrodes = {}
    excludes = {}
    num = 0
    for line in new_lines:
        if line != '':
            if line.lower()[:line.find(';')].find('electrode') != -1:
                num = int(line[line.find('(')+1:line.find(')')])
                if num not in electrodes:
                    electrodes[num] = []
                    excludes[num] = []
        if line.replace(' ','').lower().count('fill') != 0 and line.replace(' ','').lower().count('rotate_fill') == 0:
            # print(num)
            op_brac = line[line.lower().find('fill'):].count('{')
            close_brac = 0
            m = 0
            while op_brac > close_brac or op_brac == 0:
                sub_line = new_lines[n+m]       
                op_brac = op_brac + sub_line.count('{')
                close_brac = close_brac + sub_line.count('}')
                if draw_style(sub_line) == 'in':
                    # if com_type(sub_line) == 'polyline' or com_type(sub_line) == 'box':
                    electrodes[num].append(get_vtex(sub_line))
                    # elif com_type(sub_line) == 'circle':
                        # electrodes[num].append(get_vtex(sub_line))
                elif draw_style(sub_line) == 'not_in':
                    # if com_type(sub_line) == 'polyline' or com_type(sub_line) == 'box':
                    excludes[num].append(get_vtex(sub_line))
                m += 1
            elec_count += 1
        n += 1
    # using the calculated verts, clip the excludes
    elec_clip = {}
    for nam in electrodes:
        elec_clip[nam] = []
        elec_poly = []
        for part in electrodes[nam]:
            poly_part = spoly([[p[0],p[1]] for p in part])
            part_max = np.max(part,axis = 0)
            part_min = np.min(part,axis = 0)
            for clip in excludes[nam]:
                clip_poly = spoly([[p[0],p[1]] for p in clip])
                if poly_part.intersects(clip_poly) ==True:
                    x,y = clip_poly.exterior.coords.xy
                    # plt.plot(x,y)
                    poly_part=(poly_part.difference(clip_poly))
            elec_poly.append(poly_part)

        for ppart in list(mpoly(elec_poly)):
            x,y = ppart.exterior.coords.xy
            elec_clip[nam].append(np.stack([x,y]).T)
    return(elec_clip,excludes)

def gem_draw_3d(gem_file,measure = False):
    with open(gem_file) as lines:
        file_lines = lines.readlines()

    reloc = []
    new_lines = ['']*len(file_lines)

    i = 0
    # clean the file of comments,spaces and black lines
    for line in file_lines:
        line = line[:line.find(';')].strip('\n').strip(' ')
        if line != '':
            if line.find(',') !=-1 and line[-1] ==',':
                new_lines[i]+=line.strip(' ')
            else:
                new_lines[i]+=line.strip(' ')
                i+=1
    for line,n in zip(new_lines,range(len(new_lines))):
        if line == '':
            del(new_lines[i])

    elec_count = 0
    n = 0
    electrodes = {}
    excludes = []
    num = 0
    for line in new_lines:
        if line != '':
            if line.lower()[:line.find(';')].find('electrode') != -1:
                num = int(line[line.find('(')+1:line.find(')')])
                if num not in electrodes:
                    electrodes[num] = []
        if line.replace(' ','').lower().count('fill') != 0 and line.replace(' ','').lower().count('rotate_fill') == 0:
            # print(num)
            op_brac = line[line.lower().find('fill'):].count('{')
            close_brac = 0
            m = 0
            while op_brac > close_brac or op_brac == 0:
                sub_line = new_lines[n+m]       
                op_brac = op_brac + sub_line.count('{')
                close_brac = close_brac + sub_line.count('}')
                if draw_style(sub_line) == 'in':
                    if com_type(sub_line) == 'polyline' or com_type(sub_line) == 'box':
                        electrodes[num] += [get_vtex(sub_line)]
                # elif draw_style(sub_line) == 'not_in':
                #   if com_type(sub_line) == 'polyline' or com_type(sub_line) == 'box':
                #       excludes.append(Polygon(get_vtex(sub_line),color = 'white'))
                m += 1
            elec_count += 1
        n += 1
    patches = []
    keys = []

    ax = plt.subplot(projection = '3d')
    # print([nam for nam in electrodes])
    for elec in [electrodes[16]]:
        for part in elec:
            # z = np.linspace(40,40,40)
            theta = np.linspace(-np.pi,np.pi,200)
            rr,tt = np.meshgrid(part[:,1],theta)
            xx = np.meshgrid(part[:,0],theta)[0]
            yy,zz = rr*np.cos(tt),rr*np.sin(tt)
            # print([len(l) for l in [xx,yy,zz]])
            # part = mesh.Trimesh(vert)
            if np.all(xx < 100):
                ax.plot_wireframe(xx,zz,yy,
                    rcount = 100,ccount = 100)
            # surf = pltly.graph_objs.Surface(x=xx,y=yy,z=zz)
            # fig = pltly.graph_objs.Figure(data = [surf])

            # fig.show()
            # pltly.offline.plot([surf])
            # theta = np.linspace(0,np.pi/2,100)
            # y = np.cos(theta)*part[:,0]
    # ax.set_aspect('equal')

    ax.set_ylim(-70,70)
    ax.set_xlim(0,140)
    ax.set_zlim(-70,70)
    ax.view_init(10,-170)
    
    return(ax)

def gem_relocate(fil_in,fil_out):
    def find_all(a_str, sub):
        start = 0
        locs = []
        while True:
            start = a_str.find(sub, start)
            if start == -1: return locs
            locs = locs + [start]
            start += len(sub) # use start += 1 to find overlapping matches

    def com_type(line):
        if line.lower().count('box(') != 0:
            draw_type = 'box'
        elif line.lower().count('polyline(') != 0:
            draw_type = 'polyline'    
        elif line.lower().count('circle(') != 0:
            draw_type = 'circle'
        else: draw_type = 'none'
        return draw_type
            
    def within(line , brac_open, brac_close):
        return line[line.find(brac_open)+len(brac_open):line.find(brac_close)]
        
    def relocate(line,loc_pos,scale_fact = 1):
        draw_type = com_type(line)  
        if draw_type == 'none':
            return line     
        elif draw_type == 'box' or draw_type == 'polyline':
            # old_pos = np.fromstring(within(line,'(',')'),sep =',')
            od = within(line,'(',')').strip().split(',')
            old_pos = []
            for o in od:
                old_pos += o.split()
            old_pos = np.array(old_pos).flatten().astype(float)


            old_pos = old_pos.reshape(-1,2)
            # print(old_pos)
            # print(scale_fact)
            # print(loc_pos)
            # print('====')
            new_pos = (old_pos + loc_pos)*scale_fact
            new_pos = new_pos.reshape(new_pos.size)
        elif draw_type == 'circle':        
            old_pos = np.fromstring(within(line,'(',')'),sep =',')
            new_pos = old_pos
            new_pos[:2] = old_pos[:2] + loc_pos
        return (line.replace(within(line,'(',')'),np.array2string(new_pos, 
                            separator = ',',threshold = np.inf ,formatter={'float_kind':lambda x: "%.1f" % x})[1:-1].replace(' ','')))

    def next_up(str_list, start, stop):
            z = 0
            line = str_list[0]
            l = line.lower().find(start)
            char = line[l]
            while char != stop:
                l += 1
                if l == len(line):
                    z += 1
                    line = str_list[z]
                    l = 0
                char = line[l]
            return (z,l)
    #%%
    # fil_loc = 'C:\\Users\\Jon\\Google Drive\\research\\IMAP\\prelim_geom\\full_version\\IMAP_lo\\'
    # fil_nam = 'Collimator_9deg_v1.GEM'

    # lines = open(fil_nam)

    with open(fil_in) as lines:
        og_lines = lines.readlines()

    file_lines = ['']*len(og_lines)
    i = 0
    # clean the file of comments,spaces and black lines
    for line in og_lines:
        # line = line.strip('\n').strip(' ')
        if line != '':
            if line.find(',') !=-1 and line.strip('\n')[-1] ==',':
                file_lines[i]+=line.strip(' ')
            else:
                file_lines[i]+=line.strip(' ')
                i+=1
    for line,n in zip(file_lines,range(len(file_lines))):
        if line == '':
            del(file_lines[i])
    #%%
    n = 0

    reloc = []
    new_lines = []
    new_lines += file_lines
    #%
    reloc_count = 0
    for line in file_lines:
        # print(line)
        line.strip(';')
        if line.replace(' ','').lower().count('locate(') != 0 and line.count('RELOC_SKIP') == 0:
            if ';' in line:
                line = line[:line.find(';')]
            reloc_count += 1
            # identify the position in x,y of the locate call
            # loc_pos = np.fromstring(within(line,'(',')'),sep =',')[:2]
            loc_pos = np.array(within(line,'(',')').split(',')[:2])
            loc_pos[loc_pos=='']='0' 
            loc_pos = loc_pos.astype(float)  
            # scale_fact = np.fromstring(within(line,'(',')'),sep =',')[-1]
            # if scale_fact == 0 or scale_fact =='':
            #     scale_fact = 1
            # scale_fact = within(line,'(',')').split(',')[-1]
            # if scale_fact == '0' or scale_fact =='':
            #     scale_fact = 1
            # else:
            #     scale_fact = float(scale_fact)
            scale_fact = 1

            op_brac = line[line.lower().find('locate('):].count('{')
            close_brac = 0
            m = -1
    #        print('thinggs')
            while op_brac > close_brac or op_brac == 0:
                m += 1
                sub_line = file_lines[n+m]       
                op_brac = op_brac + sub_line.count('{')
                close_brac = close_brac + sub_line.count('}')
                new_lines[n+m] = relocate(sub_line.strip(), loc_pos,scale_fact = scale_fact)
    #            print(new_lines[n+m].count('\n'))

            # remove locate calls and their brackets
            z,l = next_up(new_lines[n:],'locate(','{')
            brac_num =op_brac - new_lines[n+m].count('{') - (close_brac - new_lines[n+m].count('}'))
            end_bracs = find_all(new_lines[n+m],'}')[brac_num - 1]
            
            new_lines[n] = line.replace(line[line.find('locate'):line.find(')')+1],' ')
            new_lines[n+z] = new_lines[n+z][:l] + '' + new_lines[n+z][l+1:]        
            new_lines[n+m] = new_lines[n+m][:end_bracs] + '' + new_lines[n+m][end_bracs+1:] 
         
        n += 1

    #%%
    with open(fil_out, 'w') as f:
        for item in new_lines:
    #        print(item)
            f.write(item.replace('within{','\nwithin{').replace('notin{','\nnotin{'))
    return(new_lines)


def gem_indent_correct(gem_fil_in,gem_fil_out):
    lines = [line.strip() for line in open(gem_fil_in).readlines()]
    bracs = 0
    brac_cnt = []
    op_brac = []
    clo_brac = []
    last_line = ' '
    with open(gem_fil_out,'w') as fil:
        for line in lines:  
            # if last_line != '' and last_line[-1] != ',':
            if line == '}':
                bracs = bracs - line.count('}')
                fil.write('  '*bracs +line+'\n')
                bracs = bracs + line.count('{')
            elif line == '' and last_line =='':
                pass 
            else:
                fil.write('  '*bracs +line+'\n')
                bracs = bracs - line.count('}')
                bracs = bracs + line.count('{')
            last_line = line
    print(bracs)

def gem_cleanup(gem_fil_in,gem_fil_out):
    gem_relocate(gem_fil_in,gem_fil_out)
    gem_indent_correct(gem_fil_out,gem_fil_out)


def gem_vert_chng(shift_gem, out_gem, shift_names, shift_vals):
    with open(shift_gem) as lines:
        shift_lines = lines.readlines()
        for tag, val in zip(shift_names, shift_vals):
            n = 0
            for line in shift_lines:
                if tag in line:
                    shift_lines[n] = line.replace(tag, str(val))
                n += 1
    with open(out_gem, 'w') as fil_out:
        for line in shift_lines:
            fil_out.write(line)


def check_voltage(gemfile,volts):
    import matplotlib.pylab as pl
    import matplotlib as mpl
    elec_verts = get_verts(gemfile)[0]
    for nam in elec_verts:
        elec_verts[nam] = np.concatenate(elec_verts[nam],axis = 0)

    elec_check = {}
    for nam,verts in elec_verts.items():

        elec_check[nam] = {} 
        elec_check[nam] = [np.concatenate([verts[np.argmin(verts,
                            axis = 0)],verts[np.argmax(verts,axis = 0)]])]
        # print(elec_check[nam]['check_pts'])
        # elec-Vertx|verty|r|
        elec_check[nam].append([])
        for pt in elec_check[nam][0]:
            elec_check[nam][1].append({})
            for sub_nam,sub_verts in elec_verts.items():
                if sub_nam != nam:
                    elec_check[nam][1][-1][sub_nam] = []
                    r_pt = np.sqrt(np.sum((sub_verts-pt)**2,axis = 1))
                    elec_check[nam][1][-1][sub_nam].append(min(r_pt))
                    elec_check[nam][1][-1][sub_nam].append(sub_verts[\
                                                    np.argmin(r_pt)])
                    print((volts[nam]-volts[sub_nam])/min(r_pt))
                    elec_check[nam][1][-1][sub_nam].append(
                                        (volts[nam]-volts[sub_nam])/min(r_pt))
                    if elec_check[nam][1][-1][sub_nam][-1]>=1000:
                        print('\nWARNING: exceeding 1kV/mm between electrodes %s and %s'%(nam,sub_nam))
                        print('Location:'+str(pt))
                        print('deltaV = %d'%elec_check[nam][1][-1][sub_nam][-1])
    gem_draw_poly(gemfile)
    for nam in elec_check:
        for pt,n_check in zip(elec_check[nam][0],elec_check[nam][1]):
            for sub_nam,arr in n_check.items():
                # mpl.colors.Normalize(vmin = 0,vmax = 1000)
                if sub_nam<16 and nam<16:
                    plt.plot([pt[0],arr[1][0]],[pt[1],arr[1][1]],
                             color = pl.cm.plasma(arr[-1]/1000))
    return(elec_check)


class line_draw:
    def __init__(self,fig,ax):
        # self.fig,self.ax = plt.subplots(1)
        self.fig = fig
        self.ax = ax
        self.line = self.ax.plot([0,0],[0,0])[0]
        # self.img_handle = self.ax.imshow(img.astype(int),origin = 'lower',animated = True)
        self.clickd = False
        # self.img = img.astype(int)

    def connect(self):
        print('Fig Measure connected:\nDouble Click outside Plot to Disconnect')
        self.cidpress = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        input('')
        return(self.line)
    
    def onclick(self,event):
        if plt.get_current_fig_manager().toolbar.mode != '': return
        
        if event.xdata != None and event.ydata !=None:
            self.clickd = True
            self.line.set_xdata(event.xdata)
            self.line.set_ydata(event.ydata)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        elif event.ydata == None and event.dblclick == True:
            self.disconnect()
    def on_motion(self,event):
        if self.clickd:
            # self.img[event.ydata.astype(int),
            #         event.xdata.astype(int)] = 4
            # self.line.get_xdata().append(event.xdata)
            # self.line.get_ydata().append(event.ydata)
            self.line.set_xdata(np.append(self.line.get_xdata(),np.array(event.xdata)))
            self.line.set_ydata(np.append(self.line.get_ydata(),np.array(event.ydata)))
            
            # self.ax.imshow(self.img,origin = 'lower',animated = True)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def on_release(self,event):
        self.clickd = False

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)
        print('We are Disconnected: Press Any key to continue')

# def find_surface(gemfile,img = [], d = .2,pts_mm = 5):
#     #==============================================================================
#     # Function smooth: inputs 1d array of data, performs a running average to smooth 
#     #       data
#     # input:
#     #       arr (array length N): continuous array of data to be smoothed
#     #       itterations: how many times to perform the smoothing
#     # output:
#     #       arr(smooth) (array length N): of smoothed data  
#     # Author: Jonathan Bower, University of New Hampshire 2018
#     #==============================================================================
#     def smooth(arr,itterations=1):
#         for i in range(itterations):
#             smoot=np.zeros(len(arr))
#             smoot[0]=np.nanmean([arr[0],arr[1]])
#             smoot[1:-1]=np.nanmean(np.stack([arr[0:-2],arr[1:-1],arr[2:]]),axis = 0)
#             smoot[-1]=np.nanmean([arr[-1],arr[-2]])
#             arr=smoot
#         return(arr)


#     from scipy.interpolate import interp1d
#     if img == []:
#         img = gem_draw(gemfile,plot = False)
#     canvas_shape,pxls_mm = get_canvas(gemfile)
    
#     fig,ax,elec_patches = gem_draw_poly(gemfile,path_out = True)
    
#     line = line_draw(fig,ax).connect()
#     line.set_solid_capstyle('round')
#     line_verts = np.stack(line.get_data()).T
#     poly_verts = np.concatenate((line_verts,np.flipud(line_verts)[1:]))
#     polyline = ax.add_patch(Polygon(poly_verts))
#     check = '1'
#     w = 1
#     # d = .2

#     poly_verts[:,0] = smooth(poly_verts[:,0],2)
#     poly_verts[:,1] = smooth(poly_verts[:,1],2)
#     og_verts = poly_verts.copy() 
#     og_L = np.sum(np.sqrt(np.sum((poly_verts[1:,:]-poly_verts[:-1,:])**2,
#                                      axis = 1)))
#     # pts_mm = 5
#     got_edge = np.zeros(len(poly_verts)).astype(bool)
#     # fx = interp1d(np.linspace(0,1,len(poly_verts)),poly_verts[:,0],kind = 'cubic')
#     # fy = interp1d(np.linspace(0,1,len(poly_verts)),poly_verts[:,1],kind = 'cubic')

#     # poly_verts = np.zeros((len(poly_verts)*20,2))
#     # poly_verts[:,0] = fx(np.linspace(0,1,len(poly_verts)))
#     # poly_verts[:,1] = fy(np.linspace(0,1,len(poly_verts)))
    

#     print('press any key to grow, q to quit')
#     while check != 'q':

#         L = np.zeros(len(poly_verts)) 
#         L[1:] = np.cumsum(np.sqrt(np.sum((poly_verts[1:,:]-poly_verts[:-1,:])**2,
#                                      axis = 1)))
#         fx = interp1d(L,poly_verts[:,0],kind = 'cubic')
#         fy = interp1d(L,poly_verts[:,1],kind = 'cubic')
#         f_got = interp1d(L,got_edge,kind = 'linear')

#         poly_verts = np.zeros((int(L[-1]*pts_mm),2))
#         # poly_verts = np.zeros((len(og_verts)*w,2))
#         poly_verts[:,0] = fx(np.linspace(min(L),max(L),len(poly_verts)))
#         poly_verts[:,1] = fy(np.linspace(min(L),max(L),len(poly_verts)))
#         got_edge = f_got(np.linspace(min(L),max(L),len(poly_verts))).round().astype(bool)

#         rec_verts = np.zeros((len(poly_verts)+2,poly_verts.shape[1]))
#         rec_verts[1:-1,:] = poly_verts[:,:]
#         rec_verts[0,:] = poly_verts[-2,:]
#         rec_verts[-1,:] = poly_verts[1,:]

#         # rec_verts[:,0] = smooth(rec_verts[:,0],2)
#         # rec_verts[:,1] = smooth(rec_verts[:,1],2)

#         # poly_delt = (rec_verts[2:,:]-rec_verts[:-2,:])
#         poly_delt = ((rec_verts[2:,:]-rec_verts[:-2,:])+
#                      (rec_verts[1:-1,:]-rec_verts[:-2,:])+
#                      (rec_verts[2:,:]-rec_verts[1:-1,:]))/3

#         ang = np.arctan2(poly_delt[:,0],poly_delt[:,1])
#         dx = np.cos(ang)*d
#         dy = np.sin(ang)*d
        
#         for n in range(len(poly_verts)):
#             if got_edge[n] == False:
#                 got_edge[n] = np.logical_or(\
#                                 img[((poly_verts[n,0] + dx[n])*pxls_mm).round().astype(int),
#                                 ((poly_verts[n,1] - dy[n])*pxls_mm).round().astype(int)],
#                                 img[((poly_verts[n,0])*pxls_mm).round().astype(int),
#                                 ((poly_verts[n,1])*pxls_mm).round().astype(int)])
        
#         poly_verts[~got_edge,0] += -dx[~got_edge]
#         poly_verts[~got_edge,1] += dy[~got_edge]


#         polyline.set_xy(poly_verts)
#         # ax.add_patch(Polygon(poly_verts))
#         # line.set_markersize(w)
#         w+=1
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         check = input()

        
#     print(w)
#     # ax.add_patch(Polygon(np.concatenate((np.stack(line.get_data()).T,
#     #                                              np.flipud(np.stack(line.get_data()).T)))))
#     # # line_grad = np.zeros(img.shape)
#     # line.set_clip_on(True)
#     # # for part_name in elec_patches:
#     # # for piece in elec_patches[1]:
#     # line.set_clip_path(elec_patches[0][0])
#     garb,unique_edge = np.unique(np.round(poly_verts[got_edge]*pts_mm),
#                                  axis = 0,return_index = True) 

#     return(poly_verts[got_edge][unique_edge],ang[got_edge][unique_edge])
#     # for pt_n in range(len(line)-1):

def find_surface(gemfile,img = [], d = .2,pts_mm = 5,edge_buff = .2):
    #==============================================================================
    # Function find surface: identifies electrode surface and normal direction
    # input:
    # output:
    # Author: Jonathan Bower, University of New Hampshire 2020
    # * need to update so the gemdraw image isn't inverted**
    #==============================================================================
    def smooth(arr,itterations=1):
        for i in range(itterations):
            smoot=np.zeros(len(arr))
            smoot[0]=np.nanmean([arr[0],arr[1]])
            smoot[1:-1]=np.nanmean(np.stack([arr[0:-2],arr[1:-1],arr[2:]]),axis = 0)
            smoot[-1]=np.nanmean([arr[-1],arr[-2]])
            arr=smoot
        return(arr)


    from scipy.interpolate import interp1d
    if img == []:
        img = gem_draw(gemfile,plot = False).T
    canvas_shape,pxls_mm = get_canvas(gemfile)

    fig,ax,elec_patches = gem_draw_poly(gemfile,path_out = True)
    
    line = line_draw(fig,ax).connect()
    line.set_solid_capstyle('round')
    line_verts = np.stack(line.get_data()).T
    poly_verts = np.concatenate((line_verts,np.flipud(line_verts)[1:]))
    polyline = ax.add_patch(Polygon(poly_verts))
    check = '1'
    w = 1
    # d = .2

    poly_verts[:,0] = smooth(poly_verts[:,0],2)
    poly_verts[:,1] = smooth(poly_verts[:,1],2)
    og_verts = poly_verts.copy() 
    og_L = np.sum(np.sqrt(np.sum((poly_verts[1:,:]-poly_verts[:-1,:])**2,
                                     axis = 1)))
    pts_mm = 5
    got_edge = np.zeros(len(poly_verts)).astype(bool)
    # fx = interp1d(np.linspace(0,1,len(poly_verts)),poly_verts[:,0],kind = 'cubic')
    # fy = interp1d(np.linspace(0,1,len(poly_verts)),poly_verts[:,1],kind = 'cubic')

    # poly_verts = np.zeros((len(poly_verts)*20,2))
    # poly_verts[:,0] = fx(np.linspace(0,1,len(poly_verts)))
    # poly_verts[:,1] = fy(np.linspace(0,1,len(poly_verts)))
    

    print('press any key to grow, q to quit')
    while check != 'q':

        L = np.zeros(len(poly_verts)) 
        L[1:] = np.cumsum(np.sqrt(np.sum((poly_verts[1:,:]-poly_verts[:-1,:])**2,
                                     axis = 1)))
        fx = interp1d(L,poly_verts[:,0],kind = 'cubic')
        fy = interp1d(L,poly_verts[:,1],kind = 'cubic')
        f_got = interp1d(L,got_edge,kind = 'linear')

        poly_verts = np.zeros((int(L[-1]*pts_mm),2))
        # poly_verts = np.zeros((len(og_verts)*w,2))
        poly_verts[:,0] = fx(np.linspace(min(L),max(L),len(poly_verts)))
        poly_verts[:,1] = fy(np.linspace(min(L),max(L),len(poly_verts)))
        got_edge = f_got(np.linspace(min(L),max(L),len(poly_verts))).round().astype(bool)

        rec_verts = np.zeros((len(poly_verts)+2,poly_verts.shape[1]))
        rec_verts[1:-1,:] = poly_verts[:,:]
        rec_verts[0,:] = poly_verts[-2,:]
        rec_verts[-1,:] = poly_verts[1,:]

        # rec_verts[:,0] = smooth(rec_verts[:,0],2)
        # rec_verts[:,1] = smooth(rec_verts[:,1],2)

        # poly_delt = (rec_verts[2:,:]-rec_verts[:-2,:])
        poly_delt = ((rec_verts[2:,:]-rec_verts[:-2,:])+
                     (rec_verts[1:-1,:]-rec_verts[:-2,:])+
                     (rec_verts[2:,:]-rec_verts[1:-1,:]))/3

        ang = np.arctan2(poly_delt[:,1],poly_delt[:,0])
        dx = np.sin(ang)*d
        dy = np.cos(ang)*d
        
        e_dx = np.sin(ang)*edge_buff
        e_dy = np.cos(ang)*edge_buff
        # for n in range(len(poly_verts)):
        #     if got_edge[n] == False:
        #         got_edge[n] = np.logical_or(\
        #                         img[((poly_verts[n,1] + dy[n])*10).round().astype(int),
        #                         ((poly_verts[n,0] - dx[n])*10).round().astype(int)],
        #                         img[((poly_verts[n,1])*10).round().astype(int),
        #                         ((poly_verts[n,0])*10).round().astype(int)])
        
        got_edge = np.logical_or(got_edge,
                                 np.logical_or(\
                                img[((poly_verts[:,1] + e_dy)*pxls_mm).round().astype(int),
                                ((poly_verts[:,0] - e_dx)*pxls_mm).round().astype(int)],
                                img[((poly_verts[:,1])*pxls_mm).round().astype(int),
                                ((poly_verts[:,0])*pxls_mm).round().astype(int)]))

        poly_verts[~got_edge,0] += -dx[~got_edge]
        poly_verts[~got_edge,1] += dy[~got_edge]


        polyline.set_xy(poly_verts)
        # ax.add_patch(Polygon(poly_verts))
        # line.set_markersize(w)
        w+=1
        fig.canvas.draw()
        fig.canvas.flush_events()
        check = input()

        
    print(w)
    # ax.add_patch(Polygon(np.concatenate((np.stack(line.get_data()).T,
    #                                              np.flipud(np.stack(line.get_data()).T)))))
    # # line_grad = np.zeros(img.shape)
    # line.set_clip_on(True)
    # # for part_name in elec_patches:
    # # for piece in elec_patches[1]:
    # line.set_clip_path(elec_patches[0][0])
    L = np.zeros(len(poly_verts)) 
    L[1:] = np.cumsum(np.sqrt(np.sum((poly_verts[1:,:]-poly_verts[:-1,:])**2,
                                 axis = 1)))
    garb,unique_edge = np.unique(np.round(poly_verts[got_edge]*pts_mm),
                                 axis = 0,return_index = True)
    edge_verts = poly_verts[got_edge][unique_edge][np.argsort(L[got_edge][unique_edge])]

    # get the normal direction of the surface points 
    edge_dx,edge_dy = get_gem_edge_norm(gemfile,edge_verts,pxls_mm,img.T, show = False)
    edge_ang = np.arctan2(edge_dy,edge_dx)

    # Add nanvalues to discontinuous points aroudn the surface helps with interpolation
    L = np.zeros(len(edge_verts)) 
    L[1:] = np.cumsum(np.sqrt(np.sum((edge_verts[1:,:]-edge_verts[:-1,:])**2,
                                 axis = 1)))
    dl = np.zeros(len(L))
    dl[1:] = L[1:]-L[:-1]
    dis = np.argwhere(dl>d*pts_mm).flatten()
    edge_verts = np.insert(edge_verts,dis,np.nan,axis = 0)
    edge_ang = np.insert(edge_ang,dis,np.nan)
    
    ax.plot(edge_verts[:,0],edge_verts[:,1])
    return(edge_verts,edge_ang)

def get_gem_edge_norm(gemfil,verts,pxls_mm,gem_img=[],show = True,grad_num = 5):
    if gem_img == []:
        img = gem_draw(gemfil,plot = False)
    else:
        img = gem_img
    
    imgg = (img).astype(int)
    for i in range(grad_num):
        dx,dy = np.gradient(imgg)
        imgg = imgg + np.sqrt(dx**2+dy**2)
        imgg = imgg/np.max(imgg)
        imgg[img] = 1

    dx,dy = np.gradient(imgg)
    dxh = np.nan_to_num(-dx/np.sqrt(dx**2+dy**2))*10
    dyh = np.nan_to_num(-dy/np.sqrt(dx**2+dy**2))*10
    
    dx_v = dxh[(verts*pxls_mm).round(0).astype(int)[:,0],
                (verts*pxls_mm).round(0).astype(int)[:,1]]
    dy_v = dyh[(verts*pxls_mm).round(0).astype(int)[:,0],
                (verts*pxls_mm).round(0).astype(int)[:,1]]
    if show ==True:
        fig,ax = gem_draw_poly(gemfil)
        ax.quiver(verts[:,0],verts[:,1],dx_v,dy_v,scale_units = 'xy')
    return(dx_v,dy_v)


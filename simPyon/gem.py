import numpy as np
from matplotlib import widgets
from matplotlib.patches import Polygon

        
def within(line , brac_open, brac_close):
    return line[line.find(brac_open)+len(brac_open):line.find(brac_close)]

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

def get_pa_info(gem_file):
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
            info = within(line,'(',')').split(',')
            canvas_info = {'Lx':int(info[0]),
                           'Ly':int(info[1]),
                           'Lz':int(info[2]),
                           'symmetry':info[3].strip().lower(),
                           'mirroring': info[4].strip()[0].lower(),
                           'base':('y' if info[4].strip()[0].lower() == 'x' else 'x'),
                           'pxls_mm':float(info[7])
                           }
    for line in file_lines: 
        if 'SET_PA_LOCATION' in line:
            pa_loc = [float(val) for val in within(line,'[',']').split(',')]
            canvas_info['pa_offset_position'] = np.array(pa_loc)
    return(canvas_info)

def grow_line(line,fig,ax, d = .2,pts_mm = 5,edge_buff = .2):
    from scipy.interpolate import interp1d
    def smooth(arr,itterations=1):
        for i in range(itterations):
            smoot=np.zeros(len(arr))
            smoot[0]=np.nanmean([arr[0],arr[1]])
            smoot[1:-1]=np.nanmean(np.stack([arr[0:-2],arr[1:-1],arr[2:]]),axis = 0)
            smoot[-1]=np.nanmean([arr[-1],arr[-2]])
            arr=smoot
        return(arr)
    line.set_solid_capstyle('round')
    line_verts = np.stack(line.get_data()).T
    poly_verts = np.concatenate((line_verts,np.flipud(line_verts)[1:]))
    polyline = ax.add_patch(Polygon(poly_verts))
    check = '1'
    w = 1

    poly_verts[:,0] = smooth(poly_verts[:,0],2)
    poly_verts[:,1] = smooth(poly_verts[:,1],2)
    og_verts = poly_verts.copy() 
    og_L = np.sum(np.sqrt(np.sum((poly_verts[1:,:]-poly_verts[:-1,:])**2,
                                     axis = 1)))
    got_edge = np.zeros(len(poly_verts)).astype(bool)

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

        poly_delt = ((rec_verts[2:,:]-rec_verts[:-2,:])+
                     (rec_verts[1:-1,:]-rec_verts[:-2,:])+
                     (rec_verts[2:,:]-rec_verts[1:-1,:]))/3

        ang = np.arctan2(poly_delt[:,1],poly_delt[:,0])
        dx = np.sin(ang)*d
        dy = np.cos(ang)*d
        
        e_dx = np.sin(ang)*edge_buff
        e_dy = np.cos(ang)*edge_buff
        
        # got_edge = np.logical_or(got_edge,
        #                          np.logical_or(\
        #                         img[((poly_verts[:,1] + e_dy)*pxls_mm).round().astype(int),
        #                         ((poly_verts[:,0] - e_dx)*pxls_mm).round().astype(int)],
        #                         img[((poly_verts[:,1])*pxls_mm).round().astype(int),
        #                         ((poly_verts[:,0])*pxls_mm).round().astype(int)]))

        poly_verts[:,0] += -dx
        poly_verts[:,1] += dy

        over = polyline.get_path().contains_points(poly_verts)
        poly_verts = poly_verts[~over]
        # got_edge = got_edge[np.logical_or(got_edge,~over)]
        polyline.set_xy(poly_verts)
        w+=1
        fig.canvas.draw()
        fig.canvas.flush_events()
        # fig.ginput(1)
        # check = input('press any key to grow, q to quit')
    return(poly_verts)

def find_surface(gemfile,line,img = [], d = .2,pts_mm = 5,edge_buff = .2):
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

    # fig,ax,elec_patches = gem_draw_poly(gemfile,path_out = True)
    from .poly_gem import draw
    
    fig,ax = draw(gemfile)
    fig.canvas.draw()

    # lin_drw = meat.line_draw(fig,ax)
    # lin_drw.connect()
    # fig.canvas.draw()
    # fig.canvas.draw()
    # plt.ginput(2)
    # input('Hit a key when done drawing')
    # line = lin_drw.lines[-1]

    line.set_solid_capstyle('round')
    line_verts = np.stack(line.get_data()).T
    poly_verts = np.concatenate((line_verts,np.flipud(line_verts)[1:]))
    polyline = ax.add_patch(Polygon(poly_verts))
    check = '1'
    w = 1

    poly_verts[:,0] = smooth(poly_verts[:,0],2)
    poly_verts[:,1] = smooth(poly_verts[:,1],2)
    og_verts = poly_verts.copy() 
    og_L = np.sum(np.sqrt(np.sum((poly_verts[1:,:]-poly_verts[:-1,:])**2,
                                     axis = 1)))
    got_edge = np.zeros(len(poly_verts)).astype(bool)

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

        poly_delt = ((rec_verts[2:,:]-rec_verts[:-2,:])+
                     (rec_verts[1:-1,:]-rec_verts[:-2,:])+
                     (rec_verts[2:,:]-rec_verts[1:-1,:]))/3

        ang = np.arctan2(poly_delt[:,1],poly_delt[:,0])
        dx = np.sin(ang)*d
        dy = np.cos(ang)*d
        
        e_dx = np.sin(ang)*edge_buff
        e_dy = np.cos(ang)*edge_buff
        
        got_edge = np.logical_or(got_edge,
                                 np.logical_or(\
                                img[((poly_verts[:,1] + e_dy)*pxls_mm).round().astype(int),
                                ((poly_verts[:,0] - e_dx)*pxls_mm).round().astype(int)],
                                img[((poly_verts[:,1])*pxls_mm).round().astype(int),
                                ((poly_verts[:,0])*pxls_mm).round().astype(int)]))

        poly_verts[~got_edge,0] += -dx[~got_edge]
        poly_verts[~got_edge,1] += dy[~got_edge]

        over = polyline.get_path().contains_points(poly_verts)
        poly_verts = poly_verts[np.logical_or(got_edge,~over)]
        got_edge = got_edge[np.logical_or(got_edge,~over)]
        polyline.set_xy(poly_verts)
        w+=1
        fig.canvas.draw()
        fig.canvas.flush_events()
        check = input('press any key to grow, q to quit')

        
    print(w)

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
    
    ax.plot(edge_verts[:,0],edge_verts[:,1],'.')
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



def command_nest(gemfil):
    fil = open(gemfil,'r')
    lines = fil.readlines()
    # clean lines and remove comments
    clean_lines =[]
    for line in lines:
        l = line.strip()
        if l and l[0]!=';':
            clean_lines.append(l.split(';')[0].strip())

    # identify start of electrode declaration
    for i in range(len(clean_lines)):
        if '{' in clean_lines[i].lower():
            break

    # concatenate lines into one mega line
    tot_line = ''
    for l in clean_lines[i:]:
        tot_line +=l.replace(' ','')

    # split lines by electrode declaration
    electrode_list = tot_line.lower().split('electrode')[1:]
    # subdivide draw commands by electrode, fill, and drawtype
    electrodes = {}
    loc_0 = np.zeros(2)
    for elec in electrode_list:
        if '(' in elec:
            op_split = elec.split('fill{')[1:]
            op_head = elec.split('fill{')[0]
            if 'locate(' in op_head:
                op_head_split = op_head.split('locate(')
                loc_1 = np.fromstring(op_head_split[1][op_head_split[1].find('locate(')+1:op_head_split[1].find(')')],sep = ',')[:2]
                e_num = int(elec[op_head_split[0].find('(')+1:op_head_split[0].find(')')])
            else:
                loc_1 = np.zeros(2)+loc_0
                e_num = int(elec[op_head.find('(')+1:op_head.find(')')])
            # print(loc_0)
            print(loc_1)

            if e_num not in electrodes:
                electrodes[e_num] = []
            for op in op_split:
                # print(op.count('{')-op.count('}'))
                bsp_1 = op.split('}')[:-1]
                fill = {}
                fill['dtype'] = []
                fill['polys'] = []
                fill['locate'] = []
                for b in bsp_1:
                    if 'in' in b:
                        dtype_split = b.split('{')
                        poly_split = dtype_split[1].split(')')
                        sep_poly = {}
                        sep_poly['shape'] = []
                        sep_poly['verts'] = []
                        for poly in poly_split:
                            if poly:
                                vert_split=  poly.split('(')
                                nam = vert_split[0]
                                verts = np.fromstring(vert_split[1],sep = ',')
                                sep_poly['shape'].append(nam)
                                sep_poly['verts'].append(verts)
                        fill['dtype'].append(dtype_split[0])
                        fill['polys'].append(sep_poly)
                electrodes[e_num].append(fill)

    return(electrodes)




def get_elec_nums_gem(gem_fil):
    '''
    Get the electrode numbers and names from a gemfile and store them in 
    elec_nums and elec_dict. 
    
    Parameters
    ----------
    gem_fil: string
        gemfile to take the names and values from. Default uses the GEM file stored
        in self.gemfil
    '''

    lines = open(gem_fil).readlines()
    elec_num = []
    elec_dict = {}
    for line in lines:
        if line != '':
            if line.lower()[:line.find(';')].find('electrode') != -1:
                num = int(line[line.find('(') + 1:line.find(')')])
                if num not in elec_num:
                    elec_num.append(num)
                    elec_dict[num] = line[line.find(
                        ';'):].strip(';').strip()
    return(elec_num,elec_dict)
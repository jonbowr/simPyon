from shapely.geometry import LineString as ls
from shapely.geometry import MultiLineString as ms
from shapely.geometry.multipolygon import MultiPolygon as mp
from shapely.geometry.polygon import Polygon as pg
from scipy.interpolate import interp1d,RectBivariateSpline,splrep,interp2d
from scipy.ndimage import gaussian_filter as gf
import time
from matplotlib import cm
import numpy as np

from ipywidgets import FloatProgress,IntProgress
from IPython.display import display


def plot_lines(lines,color= cm.jet(np.random.rand()),ax = []):
    for lin in (lines if type(lines) == ms or type(lines) == list else [lines]):
        lin_xy = np.array(lin.xy).T
        ax.plot(lin_xy[:,0],lin_xy[:,1],color = color)

def get_xy(lines):
    lin_xy = []
    for lin in (lines if type(lines) == ms or type(lines) == list else [lines]):
        lin_xy.append(np.array(lin.xy).T)
    return(lin_xy)


def spline_combine(l_splines):
    from shapely.ops import linemerge
    xy = np.zeros((len(l_splines),2))
    elecs = np.zeros((len(l_splines),2))
    itype = np.zeros(len(l_splines)).astype(str)
    lins = []
    for l,n in zip(l_splines,range(len(l_splines))):
        xy[n] = l.l()
        elecs[n] = l.elec
        itype[n] = l.itype
        lins.append(l.line)

    srt = np.argsort(xy[:,0])
    # ee = np.zeros(len(l_splines)+1)
    # ee[:-1] = elecs[srt][:,0]
    # ee[-1] = elecs[srt][-1,1]
    ee = elecs[srt]
    return(line_spline(linemerge(lins),itype = itype[srt],elec = ee))


class line_spline:
    def __init__(self, line_string, itype = '', elec = np.zeros(2)*np.nan):
        self.line = line_string
        self.itype = itype
        self.elec = elec
        # self.int_lines = []
        # self.int_pts = []
        # self.int_pos = []

    def __call__(self):
        return(self.line)

    def xy(self):
        return(np.array(self.line.xy).T)

    def l(self):
        xy = self.xy()
        log_dir = np.argwhere(xy[0,:]!=xy[1,:])
        return(xy[:,(log_dir if log_dir else 0)].flatten())

    def loc(self):
        xy = self.xy()
        log_dir = xy[0,:]==xy[1,:]
        return(xy[:,log_dir].flatten()[0])

    def get_f(self,bound = 0):
        from scipy.interpolate import interp1d
        y = np.concatenate([self.elec[:,0],self.elec[-1,-1]])
        return(interp1d(self.l(),np.nan_to_num(y),kind = 'linear'))
    # def ls_intersection(self,line_spline):
    #     if self not in line_spline.int_lines and self.line.intersects(line_spline.line):
    #         self.int_lines.append(line_spline)
    #         line_spline.int_lines.append(self)

    #         int_pt = self.line.intersection(line_spline.line)

    #         for l in [self,line_spline]:
    #             # l.int_pts.append(np.array(int_pt.xy).flatten())
    #             l.int_pos.append(l.line.project(int_pt))
    #         l.int_pts.append(l.line.interpolate(l.line.project(int_pt)))

class poly_group:

    def __init__(self,poly_dict):
        self.p_dict = poly_dict

    def __call__(self):
        if type(list(self.p_dict.values())[0]) ==mp:
            return(mp(sum((list(t) for t in self.p_dict.values()),[])))
        if type(list(self.p_dict.values())[0]) ==ms:
            return(ms(sum((list(t) for t in self.p_dict.values()),[])))

    def __getitem__(self,item):
        return(self.p_dict[item])

    def __iter__(self):
        return(iter(self.p_dict))

class line_mesh:
    def __init__(self,h,v,gpolys):
        long_coords = [((t,min(v),0),(t,max(v),0)) for t in h]
        lat_coords = [((min(h),t,0),(max(h),t,0)) for t in v]
        # lins ={'long':{'coords':[ls(t) for t in long_coords]},'lat':{'coords':[ls(t) for t in lat_coords]}}
        lins = {'v':[{'coords':line_spline(ls(t),itype = 'guide')} for t in long_coords],
                'h':[{'coords':line_spline(ls(t),itype = 'guide')} for t in lat_coords]}
        self.h = h
        self.v = v
        self.polys = gpolys
        self.lins = {}
        for crd_nam,lines in lins.items():
            self.lins[crd_nam] = []
            for lin in lines:
                lin['lin'] = []
                # lin['out'] = []
                crds_xy = lin['coords'].xy()
                lin['loc'] = lin['coords'].loc()
                for nam,pol in gpolys.items():
                    lin_int = lin['coords'].line.intersection(pol)
                    if lin_int:
                        if type(lin_int)==ms:
                            for l in lin_int:
                                lin['lin'].append(line_spline(l,itype = 'in',elec = [nam]*2))
                        elif type(lin_int)==ls: 
                            lin['lin'].append(line_spline(lin_int,itype = 'in',elec = [nam]*2))
                tot_ln = []
                tot_elec = []
                for intrs in lin['lin']: 
                    if intrs.line:
                        tot_ln.append(intrs.xy())
                        tot_elec.append(intrs.elec)
                # lin['els'] = []
                if len(tot_ln) != 0:
                    tot_ln = np.concatenate(tot_ln)
                    tot_elec = np.stack(tot_elec)
                    tot_srt = np.argsort(tot_ln[:,(0 if crd_nam == 'h' else 1)])

                    cpld_crds = np.insert(crds_xy,1,tot_ln[tot_srt],axis = 0)
                    cpld_elec = np.zeros(len(cpld_crds))*np.nan
                    cpld_elec[1:-1] = tot_elec.flatten()
                    new_ln = []
                    n = 0
                    for ep,el in zip(cpld_crds.reshape(-1,2,2),cpld_elec.reshape(-1,2)):
                        l = line_spline(ls(np.concatenate([ep,el.reshape(2,1)],axis = 1)),
                                           itype = 'out',elec = el)
                        new_ln.append(ep)
                        if type(l.line)==ls:
                            lin['lin'].append(l)
                        n+=1
                else:
                    l = line_spline(ls(np.concatenate([crds_xy,np.zeros((2,1))*np.nan],axis = 1)),
                                    itype = 'out',elec = np.zeros(2)*np.nan)
                    if type(l.line)==ls:
                        lin['lin'].append(l)

                self.lins[crd_nam].append(spline_combine(lin['lin']))



    def calc_coords(self):
        self.coords = {}
        for directs, vals in self.lins.items():
            self.coords[directs] = []
            for lins in vals:
                pts = []
                el = []
                reg = []
                for l in lins['out']:
                    pt = l.l()
                    if len(pt) !=0:
                        pts.append(pt)
                        el.append(l.elec)
                        reg.append(l.itype)
                if pts:
                    srt = np.argsort(np.stack(pts)[:,0])
                    self.coords[directs].append({'loc':lins['loc'],'pts':np.stack(pts)[srt],'el':np.stack(el)[srt],'type':np.array(reg)[srt]})
                else:
                    self.coords[directs].append({'loc':lins['loc'],'pts':np.array([]),'el':np.array([]),'type' : np.array([])})

    def get_v_eq(self):
        from scipy.interpolate import interp1d
        class f_ch:
            def __init__(self,f1,f2):
                self.f1 = f1
                self.f2 = f2
            def __call__(self,x):
                return(np.sqrt((self.f1(x)*np.flipud(self.f2(np.flipud(x))))))
        eq = {}
        for nams,locs in self.coords.items():
            eq[nams] = []
            for vals in locs:
                if len(vals['pts'])!=0:
                    t =vals['pts']
                    et = (vals['el']+1)*10
                    et[np.isnan(et)] = 0
                    tt = vals['type']
                    et[tt == 'out'] = np.nan
                    f1 = interp1d(t.flatten(),et.flatten(),kind = 'linear',assume_sorted = True)
                    # f2 = interp1d(np.flipud(t.flatten()),np.flipud(et.flatten()),kind = 'linear')


                    # eq[nams].append(f_ch(f1,f2))
                    eq[nams].append(f1)
                    # pts[nams].append(t.flatten())
                    # pts['y' if nams=='x' else 'x'].append(np.array([vals['loc']]*len(t.flatten())))
                    # pts['z'].append(et.flatten())
        self.eq = eq
        # for nam in pts:
        #     pts[nam] = np.concatenate(pts[nam])
        # return(pts)

    def get_xyz(self):
        from scipy.interpolate import interp1d
        pts = {'x':[],'y':[],'z':[]}
        for nams,locs in self.coords.items():
            for vals in locs:
                if len(vals['pts'])!=0:
                    t =vals['pts']
                    et = (vals['el']+1)*10
                    tt = vals['type']
                    et[tt == 'out'] = 0
                    pts[nams].append(t.flatten())
                    pts['y' if nams=='x' else 'x'].append(np.array([vals['loc']]*len(t.flatten())))
                    pts['z'].append(et.flatten())
        for nam in pts:
            pts[nam] = np.concatenate(pts[nam])
        return(pts)


    def calc_intersections(self):
        for lh in self.lins['h']:
            for l in lh['out']:
                pick_l = np.argwhere(np.logical_and(self.h>min(l.l()),
                                                    self.h<max(l.l()))).flatten().astype(int)
                for p in pick_l:
                    for ho in self.lins['v'][p]['out']:
                        l.ls_intersection(ho)


class line_mesh2:
    def __init__(self,h,v,glines):
        long_coords = [((t,min(v)),(t,max(v))) for t in h]
        lat_coords = [((min(h),t),(max(h),t)) for t in v]
        # lins ={'long':{'coords':[ls(t) for t in long_coords]},'lat':{'coords':[ls(t) for t in lat_coords]}}
        lins = {'v':[ls(t) for t in long_coords],
                'h':[ls(t) for t in lat_coords]}
        # self.lins = lins

        ints = {}
        for crd,lin in lins.items():
            ints[crd] = []
            for l in lin:
                lint = glines().intersection(l)
                coords = np.zeros((2,3))-1
                coords[:,:2] = np.array(l)
                if lint:
                    if type(lint) == ls:
                        ints[crd].append(ls([coords[0]]+list(lint.coords)+[coords[1]]))
                    elif type(lint)==ms:
                        # pass
                        ints[crd].append(ls(coords))
                        # ints[crd].append(ls([coords[0]]+[np.array(l.coords) for l in lint]+[coords[1]]))
                    else:
                        ints[crd].append(ls([coords[0]]+list(lint)+[coords[1]]))
                else:
                    ints[crd].append(ls(coords))
        self.lins = ints
        self.coords = {}
        self.coords['h'] = h
        self.coords['v'] = v
        self.poly_lines = glines
        self.f_idx = interp1d(h,np.arange(len(h)),kind = 'nearest')
        self.f_idy = interp1d(v,np.arange(len(v)),kind = 'nearest') 

    def __crd_ind__(self,crd):
        return(0 if crd =='h' else 1)

    def  calc_f(self,volt_d):       
        from scipy.interpolate import interp1d
        self.f = {}
        for crd,lins in self.lins.items():
            self.f[crd] = []
            for l in lins:
                ll = np.array(l)
                le = np.zeros(len(ll))
                for n in range(len(ll)):
                    if ll[n,2]>0 and ll[n,2] in volt_d:
                        le[n] = volt_d[int(ll[n,2])]

                self.f[crd].append(interp1d(ll[:,self.__crd_ind__(crd)],le,
                                            kind = 'linear',assume_sorted = True))

    def  calc_f_elec(self,conv = 10**-3):       
        import time as t
        f_e = {}

        for crd,lins in self.lins.items():
            # f_e[crd] = {}
            e_dict = {}
            for e in self.poly_lines:
                e_dict[e] = []
                for l in lins:
                    ll = np.array(l)
                    le = np.zeros(len(ll))

                    # print(np.array(l))
                        # for n in range(len(ll)):
                            # if ll[n,2]>0 and ll[n,2] in volt_d:
                            #     le[n] = volt_d[int(ll[n,2])]
                    le[ll[:,2]==e] = 1
                    e_dict[e].append(interp1d(ll[:,self.__crd_ind__(crd)],le,
                                            kind = 'linear',
                                            assume_sorted = True)(self.coords[crd]))
                e_dict[e] = np.stack(e_dict[e])
            f_e[crd]=e_dict
        int_bar = IntProgress(min=0, max=len(e_dict))
        int_bar.value = 0
        frac_bar = FloatProgress(min=0, max=1) # instantiate the bar
        display(int_bar) # display the bar
        display(frac_bar)

        self.f_e = {}
        self.f_cng = {}

        ts = t.time()
        filt = self.filt
        for e in f_e['v']:
            im = (f_e['h'][e]+f_e['v'][e].T)/2
            cng  = []
            cng_av = 1
            for gw in [1]:
                while cng_av >conv:
                    sm = gf(im,gw,mode = ['reflect','nearest'])
                    # sm = smooth_im(im)
                    cng_av = np.max(abs(sm[filt]-im[filt]))
                    cng.append(cng_av)
                    im[filt] = sm[filt]
                    frac_bar.value = conv/cng_av
                cng_av = 1
            int_bar.value+=1
            self.f_e[e] = im
            self.f_cng[e] = np.array(cng)

        print(t.time()-ts)

    def get_group_pts(self):
        f_e = {}
        for crd,lins in self.lins.items():
            e_dict = {}
            for e in self.poly_lines:
                e_dict[e] = []
                for l in lins:
                    ll = np.array(l)
                    le = np.zeros(len(ll))
                    le[ll[:,2]==e] = 1
                    e_dict[e].append(np.array([ll[:,self.__crd_ind__(crd)],le]).T)
            f_e[crd]=e_dict
        return(f_e)


    def  calc_f_lines(self,conv = 10**-3):       
        import time as t

        steps = [8,4,2,1]
        # ts = t.time()
        pt_dict = self.get_group_pts()

        int_bar = IntProgress(min=0, max=len(steps))
        int_bar.value = 0
        frac_bar = FloatProgress(min=0, max=1) # instantiate the bar
        display(int_bar) # display the bar
        display(frac_bar)

        x = self.coords['h']
        y = self.coords['v']

        xin = np.array([])
        yin = np.array([])
        for skippe in steps:
            frac_bar.value = 0
            pt_dict = self.interp_elec(pt_dict,x[::skippe],y[::skippe],frac_bar,s = skippe)
            int_bar.value+=1
        self.interp_pts = pt_dict
        dictor = {}
        for crd,dir_e in pt_dict.items():
            e_dict = {}
            for e,e_pts in dir_e.items():
                e_dict[e] = []
                for pts in e_pts:
                    e_dict[e].append(interp1d(pts[:,0],pts[:,1],assume_sorted = True,
                                              kind = 'linear')(x if crd == 'h' else y))
                e_dict[e] = np.stack(e_dict[e])
            dictor[crd] = e_dict


        self.f_e = {}
        filt = self.filt
        for e in dictor['v']:
            im=((dictor['h'][e]+dictor['v'][e].T)/2)
            # cng  = []
            cng_av = 1
            # for gw in [4,2,1]:
            while cng_av >conv:
                sm = gf(im,1,mode = ['mirror','nearest'])
                # sm = smooth_im(im)
                cng_av = np.max(abs(sm[filt]-im[filt]))
                # cng.append(cng_av)
                im[filt] = sm[filt]
                frac_bar.value = conv/cng_av
            #     cng_av = 1
            # int_bar.value+=1
            # im[~filt] = np.nan
            # im[~filt] = np.round(im[~filt],0)
            self.f_e[e] = im
            # self.f_cng[e] = np.array(cng)
        # print(t.time()-ts)

    def interp_elec(self,pt_dict,x,y,barp,conv = 10**-3,s = 1):

        f_e = {}
        # ts = t.time()
        filt = self.filt


        # take lin points, inerpolate them stack them to make h and v mats
        dictor = {}
        for crd,dir_e in pt_dict.items():
            e_dict = {}
            for e,e_pts in dir_e.items():
                e_dict[e] = []
                for pts in e_pts:
                    e_dict[e].append(interp1d(pts[:,0],pts[:,1])(x if crd == 'h' else y))
                e_dict[e] = np.stack(e_dict[e])
            dictor[crd] = e_dict

        filt = self.get_elec_filt(x,y)
        pxls_mm = len(x)/max(x)
        int_pts = {'h':{},'v':{}}
        for e in dictor['v']:
            im = (dictor['h'][e][self.f_idy(y).astype(int),:]+(dictor['v'][e][self.f_idx(x).astype(int),:]).T)/2
            # im[~filt] = np.round(im[~filt],0)
            cng  = []
            cng_av = 1
            for gg in [1]:
                while cng_av >conv:
                    sm = gf(im,gg,mode = ['mirror','nearest'])
                    # sm = smooth_im(im)
                    cng_av = np.max(abs(sm[filt]-im[filt]))
                    cng.append(cng_av)
                    im[filt] = sm[filt]
                    barp.value = conv/cng_av

            fxy = interp2d(x,y,im)
            for direc,old_pts in pt_dict.items():
                # int_pts[direc] = {}
                l = self.coords[('v'if direc == 'h' else 'h')]
                # for e,pts in old_pts.items():
                int_pts[direc][e] = []
                for val,pt in zip(l,old_pts[e]):
                    if direc == 'h':
                        # log = np.ones(len(x)).astype(bool)
                        # sml_pix = np.round(val*pxls_mm,0).astype(int)
                        # if sml_pix < len(y):
                        #     log = filt[sml_pix,:]
                        # else:
                        #     log = filt[-1,:]
                        log = self.filt[self.f_idy(val).astype(int),self.f_idx(x).astype(int)]
                        new_pts = np.stack([x[log],fxy(x[log],val).flatten()]).T
                    elif direc == 'v':
                        # log = np.ones(len(y)).astype(bool)
                        # sml_pix = np.round(val*pxls_mm,0).astype(int)
                        # if sml_pix < len(x):
                        #     log = filt[:,sml_pix]
                        # else:
                        #     log = filt[:,-1]
                        log = self.filt[self.f_idy(y).astype(int),self.f_idx(val).astype(int)]
                        new_pts = np.stack([y[log],fxy(val,y[log]).flatten()]).T
                    new_pts = new_pts[np.argsort(new_pts[:,0])]


                    t_pts = np.concatenate([pt[log_unique(pt[:,0],new_pts[:,0])],new_pts],axis = 0)
                    srt = np.argsort(t_pts[:,0])
                    int_pts[direc][e].append(t_pts[srt])
        return(int_pts)


    def calc_elec_filt(self):
        from skimage.draw import polygon2mask,polygon_perimeter
        x = self.coords['h']
        y = self.coords['v']
        f_filt = np.ones([len(x),len(y)]).astype(bool)
        pxls_mm = len(x)/max(x)
        for g in self.poly_lines():
            pol_ind = np.round(np.array(g)[:,:2]*pxls_mm,0)
            f_filt[polygon2mask([len(x),len(y)],pol_ind)] = False
            # pc,pr = polygon_perimeter(pol_ind[:,0],pol_ind[:,1])
            # f_filt[pc,pr] = False
        self.filt = f_filt.T
        return(f_filt.T)

    def get_elec_filt(self,x,y):
        from skimage.draw import polygon2mask,polygon_perimeter
        f_filt = np.ones([len(x),len(y)]).astype(bool)
        pxls_mm = len(x)/max(x)
        for g in self.poly_lines():
            pol_ind = np.round(np.array(g)[:,:2]*pxls_mm,0)
            f_filt[polygon2mask([len(x),len(y)],pol_ind)] = False
            # pc,pr = polygon_perimeter(pol_ind[:,0],pol_ind[:,1])
            # f_filt[pc,pr] = False
        # self.filt = f_filt.T
        return(f_filt.T)

    def get_v_array(self):
        x = self.coords['h']
        y = self.coords['v']
        im = np.zeros([len(x),len(y)])
        n= 1
        for crd,fs in self.f.items():
            ig = []
            for f in fs:
                ig.append(f(self.coords[crd]))
            thing = np.stack(ig)
            if crd == 'v':
                im = im+thing
            else:
                im=im+thing.T
            n+=1
        return((im/n).T)


    def coords_edge(self):
        dx = self.coords['h'][1]-self.coords['h'][0]
        dy = self.coords['v'][1]-self.coords['v'][0]
        x = np.zeros(len(self.coords['h'])+1)
        y = np.zeros(len(self.coords['v'])+1)
        x[:-1] = self.coords['h'] - dx/2
        x[-1] = self.coords['h'][-1]+dx/2
        y[:-1] = self.coords['v'] - dy/2
        y[-1] = self.coords['v'][-1]+dy/2
        return(x,y)


def smooth_im(image):
    nm = np.zeros(image.shape)
    nm[1:-1,1:-1] = (image[2:,1:-1]+image[:-2,1:-1]+image[1:-1,2:]+image[1:-1,:-2])/4
    nm[:,[0,-1]] = image[:,[0,-1]]
    nm[-1,:] = image[-1,:]
    nm[0,:] = nm[1,:]
    return(nm)
    # np[0,1:-1] = (image[2:,1:-1]+image[:-2,1:-1]+image[1:-1,2:]+image[1:-1,:-2])/4
    

def log_unique(A,B):
    sidx = B.argsort()
    idx = np.searchsorted(B,A,sorter=sidx)
    idx[idx==len(B)] = 0
    return(B[sidx[idx]] != A)

# class mesh_line:
#     def __init__(self,h,v,glines):
#         long_coords = [((t,min(v)),(t,max(v))) for t in h]
#         lat_coords = [((min(h),t),(max(h),t)) for t in v]
#         lins ={'h':{'coords':[ls(t) for t in long_coords]},
#                     'v':{'coords':[ls(t) for t in lat_coords]}}
#         for direct,vals in lins.items():
#             lins[direct]['lin'] = []
#             for l in vals:
#                 for g in glines.values():
#                     int_pts = l.intersection(g)
#                     if int_pts:
#                         for p in int_pts:
#                             l.line.interpolate(l.line.project(int_pt))

#         self.lins = lins

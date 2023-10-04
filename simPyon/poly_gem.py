from shapely import geometry as geo
from descartes import PolygonPatch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
plt.ion()

class gem_poly:
    
    def polygon(self,verts):
        if len(verts) > 4:
            return(geo.Polygon(verts.reshape(-1,2)+self.origin))
        else:
            return()
    
    def circle(self,verts):
        return(geo.Point(verts[:2]+self.origin).buffer(verts[2]))
    
    def box(self,verts):
        v = verts.reshape(-1,2)+self.origin
        log_check = v[0,:]==v[1,:]
        v[1,log_check] = v[1,log_check]+(np.random.rand()-.5)*max(verts)/10**3
        return(geo.box(min(v[:,0]),min(v[:,1]),max(v[:,0]),max(v[:,1])))
    
    def __init__(self, origin = np.array([0,0])):
        self.func_dict = {'box':self.box,
                          'clipbox':self.box,
                         'polyline':self.polygon,
                         'circle':self.circle}

        self.origin = origin
    def get_poly(self,dtype,verts):
        if dtype:
            return(self.func_dict[dtype.lower()](verts))
        else:
            return()

def get_verts(gemfil,clip_canvas = True):
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
        if 'electrode' in clean_lines[i].lower():
            break

    # concatenate lines into one mega line
    tot_line = ''
    for l in clean_lines[i:]:
        tot_line +=l.replace(' ','')

    # split lines by electrode declaration
    electrode_list = tot_line.lower().split('electrode')[1:]

    if clip_canvas:
        from .gem import get_pa_info
        pa_inf = get_pa_info(gemfil)
        pa_off = pa_inf['pa_offset_position']
        pax = pa_inf['Lx']*pa_inf['pxls_mm']
        pay = pa_inf['Ly']*pa_inf['pxls_mm']
        canvas_box = np.array([pa_off[0],pa_off[1],pax+pa_off[0],pay+pa_off[1]])

    # subdivide draw commands by electrode, fill, and drawtype
    electrodes = {}
    for elec in electrode_list:
        if '(' in elec:
            elec = elec.rsplit('}',2)[0]
            op_split = elec.split('fill{')[1:]
            e_num = int(elec[elec.find('(')+1:elec.find(')')])
            if e_num not in electrodes:
                electrodes[e_num] = []
            for op in op_split:
                bsp_1 = op.split('}')[:-1]
                fill = {}
                fill['dtype'] = []
                fill['polys'] = []
                locate = np.array([0,0,0,1])
                for b in bsp_1:
                    if b:
                        if 'locate' in b:
                            locate = np.fromstring(b[b.find('(')+1:b.find(')')],sep = ',')
                            b = b.split('{',1)[1]
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
                                    verts = (verts.reshape(-1,2)+locate[:2]).flatten()
                                    sep_poly['shape'].append(nam)
                                    sep_poly['verts'].append(verts)
                            if clip_canvas:
                                sep_poly['shape'].append('box')
                                sep_poly['verts'].append(canvas_box)
                            fill['dtype'].append(dtype_split[0])
                            fill['polys'].append(sep_poly)        
                    else:
                        locate = np.array([0,0,0,1])

                electrodes[e_num].append(fill)

    return(electrodes)


def verts_to_polys(electrodes,origin = np.array([0,0])):
    gd = gem_poly(origin)
    poly_polys = {}
    for elec,groups in electrodes.items():
        group_polys = []
        group_excludes = []
        for g in groups:
            inpolys = []
            outpolys = []
            for dtype,polys in zip(g['dtype'],g['polys']):
                poly = gd.get_poly(polys['shape'][0],polys['verts'][0])
                if poly:
                    if len(list(polys.values())[0])>1:
                        if polys['shape'][1].lower() == 'box':
                            clip = gd.get_poly(polys['shape'][1],polys['verts'][1])
                            poly = poly.symmetric_difference(poly.difference(clip))        
                    if dtype == 'within':
                        inpolys.append(poly)
                    elif dtype == 'notin':
                        outpolys.append(poly)
                else:
                    pass
            group_polys.append(inpolys)
            group_excludes.append(outpolys)
        poly_polys[elec] = {'in': group_polys,
                             'notin':group_excludes}
    return(poly_polys)

def poly_meld(poly_polys):
    final_polys = {}
    for elec,polys in poly_polys.items():
        elec_polys = []
        for in_polys,notin_polys in zip(polys['in'],polys['notin']):
            for in_poly in in_polys:
                if in_poly:
                    for notin_poly in notin_polys:
                        if in_poly.intersects(notin_poly) == True:
                            try:
                                in_poly = in_poly.difference(notin_poly)
                                in_poly = in_poly.simplify()
                            except:
                                pass
                                # print('Warning: Poly Draw Failed on electrode %d'%elec)
                    if in_poly.geom_type.lower()=='geometrycollection':
                        for p in in_poly.geoms:
                            if p.geom_type.lower() == 'polygon':
                                elec_polys.append(p)
                    else:
                        elec_polys.append(in_poly)
        final_polys[elec] = elec_polys
    return(final_polys)

def poly_unify(poly_polys):
    final_polys = {}
    for elec,polys in poly_polys.items():
        final_polys[elec] = []
        for in_polys,notin_polys in zip(polys['in'],polys['notin']):
            elec_polys = []
            for in_poly in in_polys:
                if in_poly:
                    for notin_poly in notin_polys:
                        if in_poly.intersects(notin_poly) == True:
                            try:
                                in_poly = in_poly.difference(notin_poly)
                                in_poly = in_poly.simplify()
                            except:
                                pass
                                # print('Warning: Poly Draw Failed on electrode %d'%elec)
                    if in_poly.geom_type.lower()=='geometrycollection' or type(in_poly) == geo.MultiPolygon:
                        for p in in_poly.geoms:
                            if p.geom_type.lower() == 'polygon':
                                elec_polys.append(p)
                        # elec_polys.append([p for p in in_poly])
                    else:
                        elec_polys.append(in_poly)
            final_polys[elec].append(geo.MultiPolygon(elec_polys))
    return(final_polys)

def poly_draw(final_polys,canvas = [],
              fig = [],ax = [],
              show_mirror= False,
              mirror_ax = None,
              origin = np.zeros(2),
              cmap = cm.rainbow,
              sub_cols = False):
    from shapely.affinity import scale,translate
    if not fig:
        fig,ax = plt.subplots()
    elecs = list(final_polys.keys())

    e_tot = 0
    for num,elec in final_polys.items():
        e_tot+=len(elec)
    n = 0
    for num,elec in final_polys.items():
        for poly in elec:
            if poly:
                if canvas:
                    poly = poly.difference(poly.difference(gem_poly().get_poly('box',np.array([0,0,canvas[0],canvas[1]]))))
                if show_mirror ==True:
                    if mirror_ax.lower()=='y':
                        Q1 = scale(poly, yfact = -1, origin = (0,0))
                    elif mirror_ax.lower()=='x':
                        Q1 = scale(poly, xfact = -1, origin = (0,0))
                    Q1 = translate(Q1,-origin[0],-origin[1])
                    tpatch = PolygonPatch(Q1)
                    tpatch.set_color(cmap(num/max(elecs)))
                    tpatch.set_linewidth(0)
                    ax.add_patch(tpatch)
                poly = translate(poly,-origin[0],-origin[1])
                patch = PolygonPatch(poly)
                xy = patch.get_verts()
                if sub_cols == True:
                    patch.set_color(cmap(n/e_tot))
                    n+=1
                else:
                    patch.set_color(cmap(num/max(elecs)))
                patch.set_linewidth(0)
                ax.add_patch(patch)

    ax.autoscale(enable = True)
    ax.set_aspect('equal')
    return(fig,ax)


def draw(gemfil,canvas = [],mirror_ax = None,fig = [],ax = [],
         origin = np.zeros(2),cmap = cm.viridis,show_mirror = False):
    electrodes = get_verts(gemfil)
    poly_polys =verts_to_polys(electrodes) 
    final_polys = poly_meld(poly_polys)
    fig,ax = poly_draw(final_polys,canvas = canvas,fig = fig,
                        ax = ax, mirror_ax = mirror_ax,
                        origin = origin,cmap = cmap,
                        show_mirror = show_mirror)
    return(fig,ax)

def clean_verts(gemfil,canvas = [],mirror_ax = None,fig = [],ax = [],origin = np.zeros(2)):
    electrodes = get_verts(gemfil)
    poly_polys =verts_to_polys(electrodes) 
    final_polys = poly_meld(poly_polys)

    from shapely.affinity import scale,translate
    if not fig:
        plt.ioff()
        fig,ax = plt.subplots()
        plt.ion()
    elecs = list(final_polys.keys())
    verts = {}
    for num,elec in final_polys.items():
        verts[num] = []
        for poly in elec:
            if poly:
                if canvas:
                    poly = poly.difference(poly.difference(gem_poly().get_poly('box',np.array([0,0,canvas[0],canvas[1]]))))
                if mirror_ax:
                    if mirror_ax.lower()=='y':
                        Q1 = scale(poly, yfact = -1, origin = (0,0))
                    elif mirror_ax.lower()=='x':
                        Q1 = scale(poly, xfact = -1, origin = (0,0))
                    Q1 = translate(Q1,-origin[0],-origin[1])
                    tpatch = PolygonPatch(Q1)
                    tpatch.set_color(cmap(num/max(elecs)))
                    tpatch.set_linewidth(0)
                    ax.add_patch(tpatch)
                poly = translate(poly,-origin[0],-origin[1])
                verts[num].append(PolygonPatch(poly).get_verts())
        # verts[num] = np.concatenate(verts[num])
    return(verts)


def get_polys(gemfil):
    electrodes = get_verts(gemfil)
    poly_polys =verts_to_polys(electrodes) 
    final_polys = poly_meld(poly_polys)
    return(final_polys)
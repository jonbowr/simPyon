from . import poly_gem as pg
from shapely.ops import unary_union,nearest_points
import numpy as np
from matplotlib import cm

class geo:

    def __init__(self,gemfil):
        self.gemfil = gemfil
        if isinstance(gemfil,str):
            self.gem_verts = pg.get_verts(self.gemfil)
            self.verts = pg.clean_verts(self.gemfil)
        elif isinstance(gemfil,list):
            self.gem_verts = {}
            self.verts = {}
            for gm in gemfil:
                gverts = pg.get_verts(gm)
                verts = pg.clean_verts(gm)
                for num in gverts:
                    if num not in self.gem_verts:
                        self.gem_verts[num] = []
                        self.verts[num] = []
                    self.gem_verts[num] += gverts[num]
                    self.verts[num] += verts[num] 
                # self.gem_verts.update()
                    # self.verts.update()

    def _repr_pretty_(self, p, cycle):
        from IPython.display import display
        display(self.get_single_poly())


    def get_xy(self):
        xy = []
        for elec in self.gem_verts.values():
            for e_part in elec:
                for polys in e_part['polys']:
                    for v in polys['verts']:
                        xy.append(v.reshape(-1,2))
        return(xy)


    def get_gx(self):
        x = []

        for elec in self.gem_verts.values():
            for e_part in elec:
                for polys in e_part['polys']:
                    for v in polys['verts']:
                        x.append(v[::2])
        return(x)

    def get_gy(self):
        x = []

        for elec in self.gem_verts.values():
            for e_part in elec:
                for polys in e_part['polys']:
                    for v in polys['verts']:
                        x.append(v[1::2])
        return(x)

    def get_x(self):
        x = []
        for elec in self.verts.values():
            for part in elec:
                x.append(part[:,0])
        return(x)

    def get_y(self):
        y = []
        for elec in self.verts.values():
            for part in elec:
                y.append(part[:,1])
        return(y)

    def get_polys(self): 
        # generates electrode num dict, with each entry containing list of individual polygons
        return(pg.poly_meld(pg.verts_to_polys(self.gem_verts)))

    def get_grouped_polys(self):
        # generates electrode num dict, with each entry containing Multipolygon patch
        poly_group = {}
        polys = self.get_polys()

        from shapely.geometry.multipolygon import MultiPolygon as mp
        for nam,pols in polys.items():
            g_geo = unary_union(pols)
            poly_group[nam] = (g_geo if type(g_geo) == mp \
                               else mp([g_geo]))
        return(poly_group)

    def get_single_poly(self):
        # generates single multipolygon patch with all geo polygons
        return(unary_union([g for p in self.get_polys().values() for g in p]))

    def get_subgrouped_polys(self):
        # generates electrode based dictionary of list of multipolygons grouped by definition in gemfil
        return(pg.poly_unify(pg.verts_to_polys(self.gem_verts)))

    def get_all_polys(self):
        # generates single multipolygon patch with all geo polygons not sure what is diff from get_single poly

        from shapely.geometry.multipolygon import MultiPolygon as mp
        return(mp([thing for stuff in self.get_grouped_polys().values() for thing in stuff]))
        
    def get_subgroup_xy(self):
        xy = {}
        for num,elec in self.gem_verts.items():
            xy[num] = []
            for e_part in elec:
                ee = []
                for polys in e_part['polys']:
                    for v in polys['verts']:
                        ee.append(v.reshape(-1,2))
                xy[num].append(np.concatenate(ee,axis = 0))
        return(xy)


    def get_grouped_lines(self):
        from shapely.geometry.multilinestring import MultiLineString as ms
        from shapely.geometry.linestring import LineString as ls
        g_lines = {}
        gpolys = self.get_grouped_polys()
        for num,g in gpolys.items():
            g_lines[num] = []
            bounds = g.boundary
            for b in bounds:
                xy = np.array(b.xy).T
                g_lines[num].append(ls(xy))
            g_lines[num] = ms(g_lines[num])
        return(g_lines)

    def get_grouped_lines_3d(self):
        from shapely.geometry.multilinestring import MultiLineString as ms
        from shapely.geometry.linestring import LineString as ls
        g_lines = {}
        gpolys = self.get_grouped_polys()
        for num,g in gpolys.items():
            g_lines[num] = []
            bounds = g.boundary
            for b in bounds:
                xy = np.array(b.xy).T
                v = np.array([num]*len(xy)).reshape(-1,1)
                g_lines[num].append(ls(np.concatenate([xy,v],axis = 1)))
            g_lines[num] = ms(g_lines[num])
        return(g_lines)

    def get_total_lines_3d(self):
        from shapely.geometry.multilinestring import MultiLineString as ms
        from shapely.geometry.linestring import LineString as ls
        g_lines = []
        gpolys = self.get_grouped_polys()
        for num,g in gpolys.items():
            bounds = g.boundary
            for b in bounds:
                xy = np.array(b.xy).T
                v = np.array([num]*len(xy)).reshape(-1,1)
                # g_lines.append(np.concatenate([xy,v],axis = 1))
                g_lines.append(ls(np.concatenate([xy,v],axis = 1)))
        g_lines = ms(g_lines)
        return(g_lines)

    def get_total_verts_3d(self):
        from shapely.geometry.multilinestring import MultiLineString as ms
        from shapely.geometry.linestring import LineString as ls
        g_lines = []
        gpolys = self.get_grouped_polys()
        for num,g in gpolys.items():
            bounds = g.boundary
            for b in bounds:
                xy = np.array(b.xy).T
                v = np.array([num]*len(xy)).reshape(-1,1)
                g_lines.append(np.concatenate([xy,v],axis = 1))
        return(g_lines)

    def get_total_verts_3d(self):
        from shapely.geometry.multilinestring import MultiLineString as ms
        from shapely.geometry.linestring import LineString as ls
        g_lines = []
        gpolys = self.get_grouped_polys()
        for num,g in gpolys.items():
            bounds = g.boundary
            for b in bounds:
                xy = np.array(b.xy).T
                v = np.array([num]*len(xy)).reshape(-1,1)
                g_lines.append(np.concatenate([xy,v],axis = 1))
        return(g_lines)

    def get_labels(self):
        # if gem_fil == []:
        for gem_fil in self.gemfil:
            # gem_fil = self.gemfil
            lines = open(gem_fil).readlines()
            elect_labels = {}
            for line in lines:
                if line != '':
                    if line.lower()[:line.find(';')].find('electrode') != -1:
                        num = int(line[line.find('(') + 1:line.find(')')])
                        if num not in elect_labels:
                            # self.elec_num += [num]
                            elect_labels[num] = []
                        elect_labels[num].append(line[line.find(
                            ';'):].strip(';').strip())
        return(elect_labels)

    def get_normal(self,xy_pts,buffer = .05):
        from shapely.geometry import MultiPoint
        pol = self.get_single_poly().buffer(buffer).boundary
        pts = MultiPoint(xy_pts)
        verts = np.array([[pr.x,pr.y] for pr in [nearest_points(pol,pt)[0] for pt in pts.geoms]])
        return(verts)
        diff = verts-xy_pts
        return(np.arctan2(diff[:,1],diff[:,0])*180/np.pi)

    def ddraw(self,label = False):
        from matplotlib import cm
        fig,ax = pg.poly_draw(self.get_subgrouped_polys(),sub_cols = True,cmap = cm.viridis)
        if label == True:
            from .fig_measure.label import label

            ax.label = label(fig,ax,[item for sublist in self.get_labels().values() for item in sublist],
                          [np.mean(item,axis = 0) for sublist in self.get_subgroup_xy().values() for item in sublist],
                          alpha = .8)

            # ax.label = label(fig,ax,[item for sublist in self.get_labels().values() for item in sublist],
            #               [np.mean(item,axis = 0) for sublist in self.get_subgroup_xy().values() for item in sublist])
            ax.label.connect()

        return(fig,ax)


    def draw(self,canvas = [],
              fig = [],ax = [],
              show_mirror= False,
              mirror_ax = None,
              origin = np.zeros(2),
              cmap = cm.viridis,
              sub_cols = False,
              show_verts = False):
        # from matplotlib import cm
        if not ax:
            from matplotlib import pyplot as plt
            fig,ax = plt.subplots()
        fig,ax1 = pg.poly_draw(self.get_grouped_polys(),
                              canvas = canvas,
                              fig = fig,ax = ax, mirror_ax = mirror_ax,
                            origin = origin,cmap = cmap,show_mirror = show_mirror)
        if show_verts == True:
            ax1.vpts = ax1.plot(np.concatenate(self.get_x())-origin[0],
                     np.concatenate(self.get_y())-origin[1],'.')[0]

        return(fig,ax)
    # def get_patches(self):
    #     patches = {}
    #     polys = self.get_polys()

    #     plt.ioff()
    #     fig,ax = plt.subplots()
    #     plt.ion()
    #     for nam,pols in polys.items():
    #         tpatch = PolygonPatch(unary_union(pols))
    #         tpatch.set_linewidth(0)
    #         patches[nam] = tpatch

    #     return(patches)
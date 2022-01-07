from . import poly_gem as pg
from shapely.ops import unary_union
import numpy as np
# from descartes import PolygonPatch
# from matplotlib import pyplot as plt

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

    # def __repr__(self):
        # return(self.get_single_poly().__repr__())
# 

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
        return(pg.poly_meld(pg.verts_to_polys(self.gem_verts)))

    def get_grouped_polys(self):
        poly_group = {}
        polys = self.get_polys()

        from shapely.geometry.multipolygon import MultiPolygon as mp
        for nam,pols in polys.items():
            g_geo = unary_union(pols)
            poly_group[nam] = (g_geo if type(g_geo) == mp \
                               else mp([g_geo]))
        return(poly_group)

    def get_single_poly(self):
        return(unary_union([g for p in self.get_polys().values() for g in p]))

    def get_subgrouped_polys(self):
        return(pg.poly_unify(pg.verts_to_polys(self.gem_verts)))

    def get_all_polys(self):
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


    def draw(self):

        from matplotlib import cm
        fig,ax = pg.poly_draw(self.get_grouped_polys(),sub_cols = True,cmap = cm.viridis)
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
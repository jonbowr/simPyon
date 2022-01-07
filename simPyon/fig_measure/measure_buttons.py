from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from . import measure_tools as mb
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


class fig_buttons(object):

    def __init__(self,fig,ax,verts = []):
        self.fig = fig
        self.ax = ax
        self.meat = {'mark':mb.mark(self.fig,self.ax).connect(),
                    'measure':mb.measure(self.fig,self.ax).connect(),
                    'draw':mb.line_draw(self.fig,self.ax).connect(),
                    'show_pts':mb.show_pts(self.fig,self.ax).connect()}
        for m in self.meat.values():
            m.active = False
        
    def mark(self, event):
        for m in self.meat.values():
            m.active = False
        self.current = self.meat['mark']
        self.current.active = True

    def show_pts(self, event):
        for m in self.meat.values():
            m.active = False
        self.current = self.meat['show_pts']
        self.current.active = True
    
    def measure(self, event):
        for m in self.meat.values():
            m.active = False
        self.current = self.meat['measure']
        self.current.active = True

    def draw(self,event):
        for m in self.meat.values():
            m.active = False
        self.current = self.meat['draw']
        self.current.active = True
        
    def clear(self,event):
        # print(self.current)
        self.current.clear()
            
    def tot_clear(self,event):
        for m in self.meat.values():
            m.tot_clear()


def measure_buttons(fig,ax, verts = []):
    callback = fig_buttons(fig,ax,verts = verts)
    

    buttonz = { 'Mark':[[],#fig.add_axes([0.01, 0.9, 0.1, 0.075]),
                        callback.mark],
                'Annotate Pts.':[[],#fig.add_axes([0.01, 0.9, 0.1, 0.075]),
                        callback.show_pts],
                'Measure':[[],#fig.add_axes([0.12, 0.9, 0.1, 0.075]),
                        callback.measure],
                'Draw':[[],#fig.add_axes([0.23, 0.9, 0.1, 0.075]),
                        callback.draw],
                'Clear':[[],#fig.add_axes([0.34, 0.9, 0.1, 0.075]),
                        callback.tot_clear],
                'Undo':[[],#fig.add_axes([0.45, 0.9, 0.1, 0.075]),
                        callback.clear]}
    shif = 0

    h = [Size.Fixed(.1),Size.Fixed(10)]
    v = [Size.Fixed(.1),Size.Fixed(.4)]
    divider = Divider(fig, (0.01, 0.9, 0.9, 0.075), h, v, aspect=False)
    bt_ax = Axes(fig, divider.get_position())
    bt_ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
    bt_ax.axis('off')
    fig.add_axes(bt_ax)
    for nam,but in buttonz.items():
        # buttonz[nam][0] = bt_ax.inset_axes([.01+shif/10,0,.09,.95])
        # buttonz[nam][0] = fig.add_axes([0.01+shif/10, 0.9, 0.1, 0.075])
        buttonz[nam][0] = fig.add_axes(bt_ax.inset_axes([shif/10,0,.09,.95]))
        buttonz[nam].append(Button(buttonz[nam][0],nam))
        buttonz[nam][2].on_clicked(but[1])
        shif+=1
    fig.buttonz = buttonz
    # fig.butt_ax = bt_ax
    return(fig,ax)
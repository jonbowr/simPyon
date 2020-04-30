from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
from . import measure_tools as mb
from matplotlib.widgets import Button

class fig_buttons(object):

    def __init__(self,fig,ax):
        self.fig = fig
        self.ax = ax
        self.meat = {'mark':mb.mark(self.fig,self.ax).connect(),
                    'measure':mb.measure(self.fig,self.ax).connect(),
                    'draw':mb.line_draw(self.fig,self.ax).connect()}
        for m in self.meat.values():
            m.active = False
        
    def mark(self, event):
        for m in self.meat.values():
            m.active = False
        self.current = self.meat['mark']
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
        print(self.current)
        self.current.clear()
            
    def tot_clear(self,event):
        for m in self.meat.values():
            m.tot_clear()


def measure_buttons(fig,ax):
    callback = fig_buttons(fig,ax)
    

    buttonz = { 'Mark':[fig.add_axes([0.01, 0.9, 0.1, 0.075]),
                        callback.mark],
                'Measure':[fig.add_axes([0.12, 0.9, 0.1, 0.075]),
                        callback.measure],
                'Draw':[fig.add_axes([0.23, 0.9, 0.1, 0.075]),
                        callback.draw],
                'Clear':[fig.add_axes([0.34, 0.9, 0.1, 0.075]),
                        callback.tot_clear],
                'Undo':[fig.add_axes([0.45, 0.9, 0.1, 0.075]),
                        callback.clear]}

    for nam,but in buttonz.items():
        buttonz[nam].append(Button(but[0],nam))
        buttonz[nam][2].on_clicked(but[1])

    fig.buttonz = buttonz
    return(fig,ax)
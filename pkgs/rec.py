import numpy as np
from tempfile import NamedTemporaryFile as ntf
import os

class data_rec:
    def __init__(self):
        self.what = {
                    "Ion N":True,
                    "Events":False,
                    "TOF":True,
                    "Mass":False,
                    "Charge":False,
                    "IN-E":False,
                    "IN-B":False,
                    "X":True,
                    "Y":True,
                    "Z":True,
                    "V":False,
                    "Azm":True,
                    "Elv":True,
                    "Vx":True,
                    "Vy":True,
                    "Vz":True,
                    "A":False,
                    "Ax":False,
                    "Ay":False,
                    "Az":False,
                    "Ve":False,
                    "Grad Ve":False,
                    "dVedx":False,
                    "dVedy":False,
                    "dVedz":False,
                    "B":False,
                    "Bx":False,
                    "By":False,
                    "Bz":False,
                    "KE":True,
                    "Ke error":False,
                    "CWF":False
                    }
    
        self.when = {
                    "Start":True,
                    "Step":False,
                    "Splat":True,
                    "All":False,
                    "Entering":False,
                    "Crossing":False,
                    "Reversals":False,
                    "Cross X":False,
                    "Cross Y":False,
                    "Cross Z":False
                    }

        self.rec_base = ['0000','0000','0000','0000','0000','0000',
                        '0000','0000','0000','0000','0000','0100',
                        '0100','0100','0100','0020','2020','2020',
                        '2020','2020','2020','2020','2020','2020',
                        '2020','2020','2020','2020','2020','2020',
                        '2020','2020','2020','2020','2020','2020',
                        '2020','2020','2020','2020','2020','2020',
                        '2020','2020','2000','0100','0000','2c00',
                        '0200','0000','0000','0000','0000','0000',
                        '0000','0000']
        self.file = None

    def data_head(self):
        head = []
        for nam,use in self.what.items():
            if use == True:
                head.append(nam)
        return(head)

    def print_recfil(self,dir = '.'):
        if self.file is not None:
            self.close()

        self.file = ntf(mode = 'w',
                   suffix = '.rec',
                   dir='.',
                   delete = False)
        # self.file = open(tmpr.name,'w')
        # print(self.file.name)
        rec_tot = []
        for use in (list(self.what.values())+list(self.when.values())):
            # print(use)
            if use == True:
                rec_tot.append('0100')
            else:
                rec_tot.append('0000')

        rec_tot+=self.rec_base
        str_tot = ''
        i = 1
        import codecs
        for thing in rec_tot:
            print(codecs.decode(thing,'hex'))
            self.file.write(str(codecs.decode(thing,'hex')))
            # str_tot += thing+('\n' if i%8 == 0 else' ')
            # i += 1
        # # print(str_tot)
        # from codecs import encode
        # return(str_tot.decode('hex'))
        # self.file.write(encode(str_tot,'hex'))
        self.file.close()

    def close(self):
        self.file.close()
        os.unlink(self.file.name)
        self.file = None

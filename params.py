
''' Storing Ship Parameters'''


class fsparam():
    def __init__(self):
        self.U0 = 7.97319           # Design Speed
        self.Lpp = 320              # Length of the Ship
        self.B = 58                 # Beam 
        self.d = 20.8               # Draft
        self.disp = 312600          # Volume Displaced
        self.Cb = 0.810             # Block Coefficient
        self.Dp = 9.86              # Diameter of propeller
        self.Hr = 15.8              # Rudder span length
        self.Ar = 112.5             # Area of Rudder
        self.rho = 1025              # Density of water
        self.rgy = 0.25*self.Lpp     # Radius of gyration
        self.xG =  11.2              # Centre of gravity
        self.m = self.disp*self.rho 
        self.Iz = self.m*(self.rgy)**2
        self.mx = 0.022
        self.my = 0.233
        self.Jz = 0.011
        self.Np = 1.77835763
        self.rdr = 1.76
    

class l7param():
    def __init__(self):
        self.U0 = 1.17953736           # Design Speed
        self.Lpp = 7                # Length of the Ship
        self.B = 1.27                 # Beam 
        self.d = 0.46               # Draft
        self.disp = 3.27          # Volume Displaced
        self.Cb = 0.810             # Block Coefficient
        self.Dp = 0.216              # Diameter of propeller
        self.Hr = 0.345              # Rudder span length
        self.Ar = 0.0539             # Area of Rudder
        self.rho = 1025              # Density of water
        self.rgy = 0.25*self.Lpp     # Radius of gyration
        self.xG =  0.25              # Centre of gravity
        self.m = self.disp*self.rho 
        self.Iz = self.m*(self.rgy)**2
        self.mx = 0.022
        self.my = 0.233
        self.Jz = 0.011
        self.Np = 12
        self.rdr = 10

class l3param():
    def __init__(self):
        self.U0 = 0.76                # Design Speed
        self.Lpp = 2.902              # Length of the Ship
        self.B = 0.527                # Beam 
        self.d = 0.189                # Draft
        self.disp = 0.235             # Volume Displaced
        self.Cb = 0.810               # Block Coefficient
        self.Dp = 0.090               # Diameter of propeller
        self.Hr = 0.144               # Rudder span length
        self.Ar = 0.00928             # Area of Rudder
        self.rho = 1025               # Density of water
        self.rgy = 0.25*self.Lpp      # Radius of gyration
        self.xG =  0.102              # Centre of gravity
        self.m = self.disp*self.rho 
        self.Iz = self.m*(self.rgy)**2
        self.mx = 0.022
        self.my = 0.233
        self.Jz = 0.011
        self.Np = 17.95
        self.rdr = 15.6







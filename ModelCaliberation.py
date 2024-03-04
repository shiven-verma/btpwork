

class expCal():
    def __init__(self):
        self.Xvv =  -0.040
        self.Xvr = 0.002
        self.Xrr = 0.011
        self.Xvvvv = 0.771

        self.Yv = -0.315
        self.Yvvv = -1.607
        self.Yr = 0.083
        self.Yrrr = 0.008
        self.Yvvr = 0.379
        self.Yvrr = -0.391

        self.Nv = -0.137
        self.Nr = -0.049
        self.Nvvv =-0.030
        self.Nvvr = -0.294
        self.Nvrr = 0.055
        self.Nrrr = -0.013

        self.res = 0.022000091196616663

        self.x_p = -0.5
        self.tp = 0.220
        self.k0 = 0.2931
        self.k1 = -0.2753
        self.k2 = -0.1385
        self.wp0 = 0.35     # 0.40 for models
        self.wpm0 = 0.4

        self.tr = 0.387
        self.ah = 0.312
        self.xh = -0.464
        self.xr = -0.5
        self.f_alpha = 2.747
        self.epsilon = 1.09
        self.kappa = 0.5
        self.lr = -0.710

        self.Nrc = (self.xr+self.ah*self.xh)
        self.Yrc = (1+self.ah)
        self.Xrc = (1-self.tr)




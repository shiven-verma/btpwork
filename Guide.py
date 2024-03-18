import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,sqrt,pi,exp,dist





class Guide():
    def __init__(self,spl,aginf,t0):
        self.p0 = aginf
        self.spline = spl
        self.t0 = t0
        self.ydd = self.spline.__call__([t0],1)[0]
        self.xdd = 1
        self.Xt0 = np.arctan2(self.ydd,self.xdd)
        self.pd0 = np.array([[t0],[self.spline.__call__([t0],0)][0]])

    def transform(self,theta):
        return np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])

    def initial(self):
        eps0 = self.transform(self.Xt0).T@(self.p0-self.pd0)
        return [eps0[0][0],eps0[1][0],self.t0]

    def liveguide(self,refstate,inp,h):
        s = refstate[0]
        e = refstate[1]
        t = refstate[2]
        psi = inp[0]
        x = inp[1]
        y = inp[2]
        #eps = np.array([[s],[e]])

        ydd = self.spline.__call__([t],1)[0]
        xdd = 1
        Xt = np.arctan2(ydd,xdd)  # angle between frame and global frame
        
        eps = np.array([[s],[e]])
        gamma = 5
        delta = 5
        Xr = np.arctan(-e/delta)

        U = 0.3*np.sqrt(delta**2+e**2)
        U = 1
        # print("speed:",U)
        psi_des = Xt + Xr
        Upp = U*cos(Xr) + gamma*s #speed of the frame
        t_dot = Upp/sqrt(ydd**2+xdd**2)

        RpX = self.transform(psi)
        RpXt = self.transform(Xt)
        
        p_dot = RpX@np.array([[U],[0]])
        pd_dot = RpXt@np.array([[Upp],[0]])

        ydd_dot = self.spline.__call__([t],2)[0]
        Xt_dot = ydd_dot*xdd*t_dot/(xdd**2+ydd**2)
        Spt = np.array([[0,-Xt_dot],[Xt_dot,0]])
        
        #eps_dot = Spt.T@eps + self.transform(psi-Xt)@np.array([[U],[0]]) - np.array([[Upp],[0]])
        p = np.array([[x],[y]])
        pd = np.array([[t],[self.spline.__call__([t],0)][0]])
        eps_dot = (RpXt@Spt).T@(p-pd) + RpXt.T@(p_dot-pd_dot) 

        
        
        s_dot = eps_dot[0][0]
        e_dot = eps_dot[1][0]

        newref = np.zeros(3)
        newref[0] = s_dot*h + s
        newref[1] = e_dot*h + e
        newref[2] = t_dot*h + t
        # guidesig = [np.sqrt(p_dot[0]**2+p_dot[1]**2)[0],psi_des]
        guidesig = [U,psi_des]
        # print("error:",eps)
        # print(guidesig," : Guide")

        return newref,guidesig
    

import numpy as np
from math import cos,sin,sqrt,atan
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

class Serret_Frenet_Guidance:
    def __init__(self,spline):
        self.spline = spline
        self.kappa = 0.1
        self.gamma = 3.0
        self.delta = 1
        

    def transform(self,theta):
        return np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
        

    def guidance_command(self,stvar):
        # t = stvar[0]
        # psi = stvar[3]
        # p = np.array([[stvar[1]],[stvar[2]]])
        # U = stvar[4]

        t,px,py,psi,U = stvar[:]
        p = np.array([[px],[py]])
        beta = 0
        
        X = psi + beta
        
        ydd = self.spline.__call__(t,1).item(0)
        xdd = 1
        Xt = atan(ydd/xdd)
        
        pd = np.array([[t],[self.spline.__call__(t,0).item(0)]])
        eps = self.transform(Xt).T@(p-pd)
        s,e = eps[:]
        

        Xr = atan(-e/self.delta)
        Upp = U*cos(X-Xt)+self.gamma*s
        t_dot = Upp/sqrt(xdd**2+ydd**2)
        
        Xd = Xt + Xr
        Ud = self.kappa*sqrt(self.delta**2+e**2)
        Ud = 0.51
        xd = pd[0][0]
        yd = pd[1][0]

        return [Xd,Ud,t_dot[0]]

class Agent():
    def __init__(self,init):
        self.x0 = init[0]
        self.y0 = init[1]
        self.psi0 = init[2]

    def initial(self):
        return [self.x0,self.y0,self.psi0]

    def dynamics(self,states,dstate,h):
        self.x = states[0]
        self.y = states[1]

        self.psid = dstate[1]
        self.ud = dstate[0]

        stder = np.zeros(2)
        stder[0] = self.ud*np.cos(self.psid)
        stder[1] = self.ud*np.sin(self.psid)
        stnew = np.array([self.psid, stder[0]*h+self.x, stder[1]*h+self.y])
        return stnew
    
    def simulation(self,spline,time,xspl,initlist):
        t = min(xspl)
        n = time.shape[0]
        h = time[1]-time[0]
        sol = np.zeros([3,n])
        SFG = Serret_Frenet_Guidance(spline)
        x,y,psi,u = initlist[[3,4,5,0]]
        var0 = [t,x,y,psi,u]
        i = 0
        while i<n:
            guide = SFG.guidance_command(var0)
            Xd,Ud,td = guide[:]
            sol[:,i] = x,y,Xd
            statenew = self.dynamics([x,y],[Ud,Xd],h)
            psi,x,y = statenew[:]
            t = td*h + t
            var0 = [t,x,y,psi,Ud]
            i+=1
            if abs(t-max(xspl))<0.5:
                sol = sol[:,:i]
                break

       
        return sol


# x0 = -3
# y0 = -1
# psi0 = 0.0

# A = Agent([x0,y0,psi0])


# # xspl = np.array([1,5,9,17,27,39,53,68,76,88])*2
# # yspl = np.array([2,1,1.7,2.7,3.4,4,5.4,4.3,3.6,3.3])*2

# xspl = np.array([1, 3.5, 5, 7, 8.7, 9.5,10,12.2])*3
# yspl = [2, 2.5, 2.4, 1.6, 1.7, 2,1.8,1.2]



# spline = CubicSpline(xspl,yspl,bc_type="clamped")

# x_ex = np.linspace(min(xspl),max(xspl),926)
# y_ex = spline(x_ex)


# T = 45
# # t = min(x_ex)
# time = np.linspace(0,T,500)

# X0 = [1.00,0,0,x0,y0,psi0,0]

# Refr1 = A.simulation(spline,time,xspl,np.array(X0))
# # print(Refr1[0,:])
# plt.plot(Refr1[0,:],Refr1[1,:])
# plt.plot(x_ex,y_ex)
# plt.show()
        
        
        
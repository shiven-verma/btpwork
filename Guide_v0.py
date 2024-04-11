import numpy as np
import matplotlib.pyplot as plt
from math import sin,cos,sqrt,pi,exp,dist
from scipy.interpolate import CubicSpline





class Guide():
    def __init__(self,spl,aginf,t0):
        aginf = np.array([[aginf[0]],[aginf[1]]])
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
        guidesig = [np.sqrt(p_dot[0]**2+p_dot[1]**2)[0],psi_des]
        guidesig = [U,psi_des]
        # print("error:",eps)
        # print(guidesig," : Guide")

        return newref,guidesig
    


class Agent():
    def __init__(self,init):
        self.x0 = init[0]
        self.y0 = init[1]
        self.psi0 = init[2]

    def initial(self):
        return [self.psi0,self.x0,self.y0]

    def dynamics(self,states,dstate,h):
        self.psi = states[0]
        #self.u = states[1]
        self.x = states[1]
        self.y = states[2]

        self.psid = dstate[1]
        self.ud = dstate[0]

        psi_dot = 3*(self.psid-self.psi)
        stder = np.zeros(3)
        
        stder[0] = psi_dot
        #stder[1] = u_dot
        stder[1] = self.ud*np.cos(self.psi)
        stder[2] = self.ud*np.sin(self.psi)

        stnew = np.array([stder[0]*h + self.psi, stder[1]*h+self.x, stder[2]*h+self.y])
        
        return stnew
    
    def simulation(self,spline,time,x,y,Ph,t0 = 0):
        RefTime = np.linspace(0,time,time*Ph)
        h = RefTime[1]-RefTime[0]
        n = RefTime.shape[0]
        X0 = self.initial()
        G = Guide(spline,X0[1:],t0)
        G0 = G.initial()
        ginit = G0.copy()
        sol = np.zeros([n,3])
        error = np.zeros([3,n])
        xinit = X0.copy()
        i = 0
        while i<n:
            sol[i,:] = xinit[:]
            refstate,guide = G.liveguide(ginit,xinit,h)
            new_state = self.dynamics(xinit,guide,h)
            xinit[:] = new_state[:]
            ginit[:] = refstate[:]
            error[:,i] = refstate
            if dist(xinit[1:],[x[-1],y[-1]]) < 0.5:
                sol = sol[0:i+1,:]
                error = error[:,0:i+1]
                RefTime = RefTime[0:i+1]
                break
            i+=1
        return sol





# x = [0,5,9,17,27,39,53,68,76,88]
# y = [0,1,1.3,2.7,6,4,5.4,3.0,3.6,4.0]

# spline = CubicSpline(x,y,bc_type="clamped")

# x_ex = np.linspace(min(x),max(x),200)
# y_ex = spline(x_ex)

# A = Agent([-5,-2,0])
# sol = A.simulation(spline,100,x,y)
# xn = sol[:,0]
# print(sol[:,0][::100].shape[0])

# plt.figure(figsize=(12,10))
# plt.plot(x_ex,y_ex,label="Spline")
# plt.plot(sol[:,1],sol[:,2],label="agent")

# ax = plt.gca()
# ax.set_aspect('equal', adjustable="box")
# plt.legend()
# plt.show()
    



    

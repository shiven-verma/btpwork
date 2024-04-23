#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as cd
from scipy.interpolate import CubicSpline

from MMG3 import simulation
from NMPC import controller,ship
from params import fsparam
from Guide import Serret_Frenet_Guidance,Agent


p = fsparam()
# Np = p.Np
NP = 30
NC = 1
Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
C = controller(NP,NC,Q)
S = ship()



x0 = -5
y0 = 4
psi0 = 0.0

A = Agent([x0,y0,psi0])


xspl = np.array([0,5,9,17,27,39,53,68,76,88])*2
yspl = np.array([0,1,1.7,2.7,3.4,4,5.4,4.3,3.6,3.3])*3

# xspl = np.insert(xspl0,0,x0)
# yspl = np.insert(yspl0,0,y0)

spline = CubicSpline(xspl,yspl,bc_type="clamped")

x_ex = np.linspace(min(xspl),max(xspl),926)
y_ex = spline(x_ex)

# pathtime = 560
Ph = 10

T = 400

time = np.linspace(0,T,T*10)

X0 = [0.50,0,0,x0,y0,psi0,0]

Refr1 = A.simulation(spline,time,x_ex,np.array(X0))
print(spline,time.shape,x_ex.shape,np.array(X0))
Refr1 = Refr1[:,::Ph]
plt.plot(Refr1[0,:],Refr1[1,:])
plt.show()



psi = spline.__call__(x_ex,1)





# ReferenceWindow = x_ex[::Ph].shape[0] 
ReferenceWindow = Refr1.shape[1]
SimumlationWindow = ReferenceWindow-C.P
print(SimumlationWindow,ReferenceWindow,Refr1.shape)


xref = x_ex[::Ph]
yref = y_ex[::Ph]
psiref = psi[::Ph]

print(x_ex[::Ph].shape[0])



# xref = np.linspace(0,ReferenceWindow-1,ReferenceWindow)
# yref = (xref/20)**2
# xrefd = xref - past(xref,1)
# yrefd = yref - past(yref,1)

# psiref = np.arctan2(yrefd,xrefd)





uopt = np.ones((SimumlationWindow,C.P))

ContMag = 1
tdiscrete = np.linspace(0,C.P-1,C.P)
tcontinuous = np.linspace(0,C.P-1,C.P*ContMag)
time = tdiscrete

# time = cd.SX(np.linspace(0,19,20))

# Xref = cd.SX(np.array([xref,yref,psiref]).T)
Xref = cd.SX(Refr1.T)
# print(Xref.shape)

# Xref = cd.vertcat(Xref,cd.SX(np.zeros([1,3])))


def circle(a,b,r):
    theta = np.linspace(0,2*np.pi,100)

    x = a + r*np.cos(theta)
    y = b + r*np.sin(theta)

    xo = a+r*0.3*np.cos(theta)
    yo = b+r*0.3*np.sin(theta)

    plt.plot(x,y,"--r",xo,yo,"b")



X0 = [0.50,0,0,x0,y0,psi0,0]

xini = X0.copy()
simdict = {}

t = xspl[0]
i = 0
while i<SimumlationWindow:
    refr = Xref[i:i+C.P,:]
    uoptd = C.nlpsolve(refr,xini,time)
    print(refr.shape,xini,time)
    uopt[i,:] = np.array(uoptd)[:,0]
    uoptm = uopt[i,:]
    uoptcont = uoptm[(tcontinuous//1).astype(int)]
    y = simulation(xini,uoptcont,tcontinuous)
    var = f"pred_{i}"
    simdict[var] = y
    xini[:] = y[:,1*ContMag]
    i+=1


# Need a piece of code which if given position gives guidance command and based on that gives next P reference points. 

# i = 0
# initx = X0.copy()
# while i<SimumlationWindow:
#     var = f"pred_{i}"
#     uoptml = uopt[i,:]
#     ucont = uoptml[(tcontinuous//1).astype(int)]
#     # value = simulation(initx,uoptml,time)
#     value = simulation(initx,ucont,tcontinuous)
#     initx = value[:,1*ContMag]
#     # print(value,"initx--")
#     simdict[var] = value    
#     i+=1


plt.figure(figsize=(14,10))
plt.plot(x_ex,y_ex,"--",label="reference")
plt.plot(xspl,yspl,'r*',label='waypoints')
# plt.plot(Refr1[0,:],Refr1[1,:])
j = 0
while j<SimumlationWindow:
    if j== SimumlationWindow-1:
        plt.plot(simdict[f"pred_{j}"][3,:],simdict[f"pred_{j}"][4,:], "-")
    else:
        plt.plot(simdict[f"pred_{j}"][3,:][[0,1]],simdict[f"pred_{j}"][4,:][[0,1]], "-", )
        # plt.plot(simdict[f"pred_{j}"][3,:],simdict[f"pred_{j}"][4,:], "--", )
    j+=1


# circle(18,3.5,1.5)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend()
plt.title("MPC with MMG obstacle")
plt.xlabel("y")
plt.ylabel("x")
# plt.savefig("mpcobs_mmg_pred.png")
print(Refr1.shape)
plt.show()
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as cd
from scipy.interpolate import CubicSpline

from MMG3 import simulation
from NMPC import controller,ship
from params import fsparam
from Guide_v0 import Agent


p = fsparam()
Np = p.Np
C = controller()
S = ship()

def past(ar,n):
    li = []
    li[:] = ar[:]
    li = [0]*n + li[:-n]
    return np.array(li)


xspl = np.array([0,5,9,17,27,39,53,68,76,88])*2
yspl = np.array([0,1,1.3,2.7,6,4,5.4,3.0,3.6,4.0])*2

spline = CubicSpline(xspl,yspl,bc_type="clamped")

x_ex = np.linspace(min(xspl),max(xspl),200)
y_ex = spline(x_ex)


x0 = -7
y0 = -2
psi0 = 0.0

pathtime = 560
Ph = 1200

A = Agent([x0,y0,psi0])
sol = A.simulation(spline,pathtime,xspl,yspl,Ph)



ReferenceWindow = sol[::Ph].shape[0] 
SimumlationWindow = ReferenceWindow-C.P

xref = sol[:,1][::Ph]
yref = sol[:,2][::Ph]
psiref = sol[:,0][::Ph]


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

Xref = cd.SX(np.array([xref,yref,psiref]).T)
print(Xref.shape)

# Xref = cd.vertcat(Xref,cd.SX(np.zeros([1,3])))


def circle(a,b,r):
    theta = np.linspace(0,2*np.pi,100)

    x = a + r*np.cos(theta)
    y = b + r*np.sin(theta)

    xo = a+r*0.3*np.cos(theta)
    yo = b+r*0.3*np.sin(theta)

    plt.plot(x,y,"--r",xo,yo,"b")




X0 = [1,0,0,x0,y0,psi0,0]
xini = X0.copy()
i = 0
while i<SimumlationWindow:
    # u = np.zeros(10)
    refr = Xref[i:i+C.P,:]
    # print(refr,"---ref")
    uoptd = C.nlpsolve(refr,xini,time)
    uopt[i,:] = np.array(uoptd)[:,0]
    uoptm = uopt[i,:]
    uoptcont = uoptm[(tcontinuous//1).astype(int)]
    y = simulation(xini,uoptcont,tcontinuous)
    # y = simulation(xini,uoptm,time)
    xini[:] = y[:,1*ContMag]
    i+=1


i = 0
simdict = {}
initx = X0.copy()
while i<SimumlationWindow:
    var = f"pred_{i}"
    uoptml = uopt[i,:]
    ucont = uoptml[(tcontinuous//1).astype(int)]
    # value = simulation(initx,uoptml,time)
    value = simulation(initx,ucont,tcontinuous)
    initx = value[:,1*ContMag]
    # print(value,"initx--")
    simdict[var] = value    
    i+=1


plt.figure(figsize=(14,10))
plt.plot(x_ex,y_ex,"--",label="reference")
plt.plot(xspl,yspl,'r*',label='waypoints')
j = 0
while j<SimumlationWindow:
    if j== SimumlationWindow-1+1:
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
plt.show()



    
    



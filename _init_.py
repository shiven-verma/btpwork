#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as cd
from scipy.interpolate import CubicSpline

from MMG3 import simulation
from NMPC import controller,ship
from params import fsparam
from Guide import Guide


p = fsparam()
Np = p.Np
G = Guide()

def past(ar,n):
    li = []
    li[:] = ar[:]
    li = [0]*n + li[:-n]
    return np.array(li)

C = controller()
S = ship()

ReferenceWindow = 80 
SimumlationWindow = ReferenceWindow-C.P


xref = np.linspace(0,ReferenceWindow-1,ReferenceWindow)
yref = (xref/20)**2
xrefd = xref - past(xref,1)
yrefd = yref - past(yref,1)

psiref = np.arctan2(yrefd,xrefd)

x = [0,1,7,9,17,29,41,58]
y = [0,1,1.3,2.7,6,4,5.4,3.0]

spline = CubicSpline(x,y,bc_type="clamped")

x_ex = np.linspace(min(x),max(x),200)
y_ex = spline(x_ex)




uopt = np.ones((SimumlationWindow,C.P))

ContMag = 1
tdiscrete = np.linspace(0,C.P-1,C.P)
tcontinuous = np.linspace(0,C.P-1,C.P*ContMag)
time = tdiscrete

# time = cd.SX(np.linspace(0,19,20))

Xref = cd.SX(np.array([xref,yref,psiref]).T)

def circle(a,b,r):
    theta = np.linspace(0,2*np.pi,100)

    x = a + r*np.cos(theta)
    y = b + r*np.sin(theta)

    xo = a+r*0.3*np.cos(theta)
    yo = b+r*0.3*np.sin(theta)

    plt.plot(x,y,"--r",xo,yo,"b")

# Ship takes [x,y,psi,u,v,r,delta]
# MMG takes [u,v,r,x,y,psi,delta,Np]

pos = []

X0 = [1,0,0,0,0,0,0]
xini = X0.copy()
i = 0
while i<SimumlationWindow:
    # u = np.zeros(10)
    refr = Xref[i:i+C.P,:]
    uoptd = C.nlpsolve(refr,xini,time)
    uopt[i,:] = np.array(uoptd)[:,0]
    uoptm = uopt[i,:]
    uoptcont = uoptm[(tcontinuous//1).astype(int)]
    print(uoptcont)
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
plt.plot(xref,yref,"--",label="reference")
j = 0
while j<SimumlationWindow:
    if j== SimumlationWindow-1+1:
        plt.plot(simdict[f"pred_{j}"][3,:],simdict[f"pred_{j}"][4,:], "-")
    else:
        plt.plot(simdict[f"pred_{j}"][3,:][[0,1]],simdict[f"pred_{j}"][4,:][[0,1]], "-*", )
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



    
    



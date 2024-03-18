#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as cd
from scipy.interpolate import CubicSpline

from MMG3 import simulation
from NMPC import controller,ship
from params import fsparam

p = fsparam()
Np = p.Np

def past(ar,n):
    li = []
    li[:] = ar[:]
    li = [0]*n + li[:-n]
    return np.array(li)

C = controller()
S = ship()

ReferenceWindow = 50 
SimumlationWindow = ReferenceWindow-C.P


xref = np.linspace(0,ReferenceWindow-1,ReferenceWindow)
yref = (xref/10)**2
xrefd = xref - past(xref,1)
yrefd = yref - past(yref,1)

psiref = np.arctan2(yrefd,xrefd)

x = [0,1,7,9,17,29,41,58]
y = [0,1,1.3,2.7,6,4,5.4,3.0]

spline = CubicSpline(x,y,bc_type="clamped")

x_ex = np.linspace(min(x),max(x),200)
y_ex = spline(x_ex)


uopt = np.ones((SimumlationWindow,C.P))

tdiscrete = np.linspace(0,C.P-1,C.P)
tcontinuous = np.linspace(0,C.P-1,C.P*100)
time = tdiscrete

# time = cd.SX(np.linspace(0,19,20))

Xref = cd.SX(np.array([xref,yref,psiref]).T)



# Ship takes [x,y,psi,u,v,r,delta]
# MMG takes [u,v,r,x,y,psi,delta,Np]

pos = []

X0 = [1,0,0,0,0,0,0]
xini = X0.copy()
i = 0
while i<SimumlationWindow:
    # u = np.zeros(10)
    refr = Xref[i:i+10,:]
    uoptd = C.nlpsolve(refr,xini,time)
    uopt[i,:] = np.array(uoptd)[:,0]
    uoptm = uopt[i,:]
    uoptcont = uoptm[((tcontinuous-0.01)//1).astype(int)]
    y = S.simulation(xini,uoptcont,tcontinuous)
    xini[:] = y[:,100]
    i+=1


i = 0
simdict = {}
initx = X0.copy()
while i<SimumlationWindow:
    var = f"pred_{i}"
    uoptml = uopt[i,:]
    ucont = uoptml[((tcontinuous-0.01)//1).astype(int)]
    value = S.simulation(initx,ucont,tcontinuous)
    initx = value[:,100]
    # print(value,"initx--")
    simdict[var] = value    
    i+=1


plt.figure(figsize=(14,10))
plt.plot(xref,yref,"--",label="reference")
j = 0
while j<SimumlationWindow:
    if j== SimumlationWindow-1:
        plt.plot(simdict[f"pred_{j}"][3,:],simdict[f"pred_{j}"][4,:], "-", label = f"iter{j+1}")
    else:
        # plt.plot(simdict[f"pred_{j}"][3,:][[0,100]],simdict[f"pred_{j}"][4,:][[0,100]], "-*", label = f"iter{j+1}")
        plt.plot(simdict[f"pred_{j}"][3,:],simdict[f"pred_{j}"][4,:], "--", label = f"iter{j+1}")
    j+=1


# circle(10,0.8,1.5)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.legend()
plt.title("MPC with MMG")
# plt.savefig("mpcobs_mmg_pred.png")
plt.show()



    
    



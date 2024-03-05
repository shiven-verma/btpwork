#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as cd

from MMG import _mmgder
from NMPC import controller,ship

def past(ar,n):
    li = []
    li[:] = ar[:]
    li = [0]*n + li[:-n]
    return np.array(li)

Rf = 40 
xref = np.linspace(0,39,40)
yref = (xref/10)**2
xrefd = xref - past(xref,1)
yrefd = yref - past(yref,1)

psiref = np.arctan2(yrefd,xrefd)

X0 = [0,0,0,1,0,0,0]
xini = X0.copy()
uopt = np.ones((Rf-10,10))
i = 0

time = cd.SX(np.linspace(0,19,20))

Xref = cd.SX(np.array([xref,yref,psiref]).T)

C = controller()
S = ship()

pos = []

while i<Rf-10:
    # u = np.zeros(10)
    refr = Xref[i:i+10,:]
    uoptd = C.nlpsolve(refr,xini,time)
    uopt[i,:] = np.array(uoptd)[:,0]
    uoptm = uopt[i,:]
    y = S.simulation(xini,uoptm,1)
    xini[:] = y[:,1]
    i+=1


simn = Rf-10
i = 0
simdict = {}
initx = X0.copy()
while i<simn:
    var = f"pred_{i}"
    value = S.simulation(initx,uopt[i,:],1)
    initx = value[:,1]
    # print(value,"initx--")
    simdict[var] = value    
    i+=1


plt.figure(figsize=(14,10))
plt.plot(xref,yref,label="reference")
j = 0
while j<simn:
    if j== simn-1:
        plt.plot(simdict[f"pred_{j}"][0,:],simdict[f"pred_{j}"][1,:], "-o", label = f"iter{j+1}")
    else:
        # plt.plot(simdict[f"pred_{j}"][0,:][0:2],simdict[f"pred_{j}"][1,:][0:2], "-o", label = f"iter{j+1}")
        plt.plot(simdict[f"pred_{j}"][0,:],simdict[f"pred_{j}"][1,:], "-o", label = f"iter{j+1}")
    j+=1

# circle(10,0.8,1.5)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.legend()
plt.title("Model Predictive Control Optimum control")
plt.show()
# plt.savefig("mpcobs_optcon.png")


    
    



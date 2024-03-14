#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import casadi as cd

from MMG import _mmgder
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

ReferenceWindow = 40 
SimumlationWindow = ReferenceWindow-C.P


xref = np.linspace(0,39,40)
yref = (xref/10)**2
xrefd = xref - past(xref,1)
yrefd = yref - past(yref,1)

psiref = np.arctan2(yrefd,xrefd)


uopt = np.ones((SimumlationWindow,10))



# Ship takes [x,y,psi,u,v,r,delta]
# MMG takes [u,v,r,x,y,psi,delta,Np]

pos = []

# X0 = [0,0,0,1,0,0,0]
# xini = X0.copy()
# i = 0
# while i<SimumlationWindow:
#     # u = np.zeros(10)
#     refr = Xref[i:i+10,:]
#     uoptd = C.nlpsolve(refr,xini,time)
#     uopt[i,:] = np.array(uoptd)[:,0]
#     uoptm = uopt[i,:]
#     y = S.simulation(xini,uoptm,1)
#     xini[:] = y[:,1]
#     i+=1


# i = 0
# simdict = {}
# initx = X0.copy()
# while i<SimumlationWindow:
#     var = f"pred_{i}"
#     value = S.simulation(initx,uopt[i,:],1)
#     initx = value[:,1]
#     # print(value,"initx--")
#     simdict[var] = value    
#     i+=1



def ChangeInitCon(ar):
    ax = np.zeros(8)
    ax[0] = ar[3]
    ax[1] = ar[4]
    ax[2] = ar[5]
    ax[3] = ar[0]
    ax[4] = ar[1]
    ax[5] = ar[2]
    ax[6] = ar[6]
    ax[7] = Np

    return ax

def InvertInitCom(ax):
    ar = np.zeros(7)
    ar[0] = ax[3]
    ar[1] = ax[4]
    ar[2] = ax[5]
    ar[3] = ax[0]
    ar[4] = ax[1]
    ar[5] = ax[2]
    ar[6] = ax[6]

    return ar

    



i = 0
X0 = [1,0,0,0,0,0,0]
xini = X0.copy()

tdiscrete = np.linspace(0,C.P,C.P)
tcontinuous = np.linspace(0,C.P,C.P*100)

# time = cd.SX(np.linspace(0,19,20))
time = tdiscrete
Xref = cd.SX(np.array([xref,yref,psiref]).T)


while i<SimumlationWindow:
    refr = Xref[i:i+10,:]
    uoptd = C.nlpsolve(refr,xini,time)
    uopt[i,:] = np.array(uoptd)[:,0]
    uoptm = uopt[i,:]
    y = _mmgder(time,xini,uoptm[0])
    print(y[6],"******delta")
    # y = S.simulation(xini,uoptm,1)
    xini[:] = y[:]
    i+=1



j = 0
simdict = {}
initx = X0.copy()
while j<SimumlationWindow:
    var = f"pred_{j}"
    y = _mmgder(time,initx,uopt[j,0])
    # y = S.simulation(xini,uoptm,1)
    value = y
    # value = S.simulation(initx,uopt[j,:],1)
    initx = value[:]
    # print(initx[],"movement")
    # print(value,"initx--")
    simdict[var] = value    
    j+=1



plt.figure(figsize=(14,10))
plt.plot(xref,yref,label="reference")
j = 0
while j<SimumlationWindow:
    if j == SimumlationWindow-1:
        plt.plot(simdict[f"pred_{j}"][0],simdict[f"pred_{j}"][1], "-o", label = f"iter{j+1}")
    else:
        # plt.plot(simdict[f"pred_{j}"][0,:][0:2],simdict[f"pred_{j}"][1,:][0:2], "-o", label = f"iter{j+1}")
        plt.plot(simdict[f"pred_{j}"][0],simdict[f"pred_{j}"][1], "-o", label = f"iter{j+1}")
    j+=1

# circle(10,0.8,1.5)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.legend()
plt.title("Model Predictive Control Optimum control")
plt.show()
# plt.savefig("mpcobs_optcon.png")


    
    



#!/usr/bin/env python3

import numpy as np
from math import sin, cos, pi, sqrt, exp
import matplotlib.pyplot as plt

######### Ship Parameters ##########
########## Table 1 ##########
U0 = 7.97319          # Design Speed
Lpp = 320          # Length of the Ship
B = 58             # Beam 
d = 20.8           # Draft
disp = 312600      # Volume Displaced
Cb = 0.810         # Block Coefficient
Dp = 9.86          # Diameter of propeller
Hr = 15.8          # Rudder span length
Ar = 112.5         # Area of Rudder
rho = 1025         # Density of water
rgy = 0.25*Lpp     # Radius of gyration
xG =  11.2     # Centre of gravity


######## Mass and Moment of inertia ###########
m_d = rho*disp
Iz_d = m_d*(rgy)**2

mass_ndm_c = 0.5*rho*d*Lpp**2 
I_ndm_c = 0.5*rho*d*Lpp**4 

m = m_d #/mass_ndm_c                    # Non-Dim mass
Iz = Iz_d #/I_ndm_c                     # Non-Dim MOI
mx,my,Jz = (0.022,0.233,0.011)          # Added Mass
mx = mx*mass_ndm_c
my = my*mass_ndm_c
Jz = Jz*I_ndm_c

####### Hull Parameters ##########
####### Table 3 #########
Xvv =  -0.040
Xvr = 0.002
Xrr = 0.011
Xvvvv = 0.771

Yv = -0.315
Yvvv = -1.607
Yr = 0.083
Yrrr = 0.008
Yvvr = 0.379
Yvrr = -0.391

Nv = -0.137
Nr = -0.049
Nvvv =-0.030
Nvvr = -0.294
Nvrr = 0.055
Nrrr = -0.013

res = 0.022000091196616663

############# Propeller Parameters ###########
######### Table 3 ##########
x_p = -0.5
tp = 0.220
k0 = 0.2931
k1 = -0.2753
k2 = -0.1385
Np_d = 1.776977
wp0 = 0.35

############## Rudder Parameters ###############
############## Table 3 #################
tr = 0.387
ah = 0.312
xh = -0.464*Lpp
xr = -0.5*Lpp
f_alpha = 2.747
epsilon = 1.09
kappa = 0.5
eta = Dp/Hr
lr = -0.710

Nrc = (xr+ah*xh)
Yrc = (1+ah)
Xrc = (1-tr)

############### Mass Matrix #################
########### Equation 4 ##############
M = np.array([[(m+mx),0,0],[0,(m+my),m*xG],[0,m*xG,(Jz+((xG**2)*m)+Iz)]])     # Mass Matrix
Minv = np.linalg.inv(M)                                                       # Inverse of Mass Matrix

############# Rudder Constraints ############
mindel = -10
maxdel = 10
mindel_r = mindel*pi/180
maxdel_r = maxdel*pi/180


def _mmgder(t,var,dc):
    
    ###### States are non-dimensional #####
    u = var[0]                # Surge velocity
    v = var[1]                # Lateral sway velocity
    r = var[2]                # Yaw rate
    x = var[3]
    y = var[4]
    psi = var[5]              # Yaw Angle
    delta = var[6]            # Rudder Angle 
    Np = var[7]               # Propeller Speed
    
    
    Ures = sqrt(u**2 + v**2)     # Resulatant Velocity
    beta = -np.arctan(v/u)
    
    U = Ures
    
    delta_c = dc
    delta_c = np.clip(delta_c,mindel,maxdel)             # Rudder Angle Commanded 
    delta_c = delta_c*pi/180                             # Degree to Radian
    delta = np.clip(delta,mindel_r,maxdel_r)             # Constrains on Delta

        

    ### Non-dimesnional constants ###
    Fndmc = (0.5)*rho*Lpp*d*(U**2)
    Mndmc = (0.5)*rho*Lpp*Lpp*d*(U**2)
    
    
    
    ############ Hull Forces ##############
    ################ Equation 7 #############
    up = u/U                        # up is u prime and non dimensional
    vm = v/U                        # vm is non-dimensional
    rp = r*Lpp/U                    # rp is non-dimensional
    Uresp = np.sqrt(up**2+vm**2)
    
    Xhnd = -res + Xvv*(vm**2) + Xvr*vm*rp + Xrr*(rp**2) + Xvvvv*vm**4
    Yhnd = Yv*vm + Yr*rp + Yvvv*vm**3 + Yvvr*rp*vm**2 + Yvrr*vm*rp**2 + Yrrr*rp**3
    Nhnd = Nv*vm + Nr*rp + Nvvv*vm**3 + Nvvr*rp*vm**2 + Nvrr*vm*rp**2 + Nrrr*rp**3

    Xh = Xhnd*Fndmc          
    Yh = Yhnd*Fndmc
    Nh = Nhnd*Mndmc
    F_hull = np.array([[Xh],[Yh],[Nh]])
    
    
    ################## Propeller Forces ###########################
    
    beta_p = beta - (x_p*rp)                                        # Eq. 15
    
    C1 = 2.0                                                       # Table 3
    C2 = 0.5*np.heaviside(beta_p,0)+1.1                            # Table 3
    
    wp  =  1 - (1+(1-exp(-C1*abs(beta_p)))*(C2-1))*(1-wp0)         # Wake Coeff. Eq. 16
    Jp = u*(1-wp)/(Np*Dp)                                        # Propeller Advanced ratio Eq. 11
    KT = k2*(Jp**2) + k1*Jp + k0                                   # Eq. 10
    
    Thrust = rho*(Np**2)*(Dp**4)*KT                                # Dimensional Thrust Eq. 9
    Xp = (1-tp)*Thrust                                             # Surge Force Eq. 8
    F_prop = np.array([[Xp],[0],[0]])
    
    
    
    ################# Rudder Force ##############################
    
    beta_r = beta - lr*rp                                          # Equation 24
    gamma_r = 0.395 if beta_r<0 else 0.640                               # Table 3
    vr = Ures*gamma_r*beta_r                                       # Equation 23
    uProp = (1 - wp)*u                                             # Equation 25
    uR1 = sqrt(1 + 8*KT/(pi*Jp**2))                                #
    uR2 = (1 + kappa*(uR1 -1))**2
    ur  = epsilon * uProp * sqrt((eta*uR2) + (1-eta))              #
    Urs = vr**2 + ur**2                                            
    Ur = sqrt(Urs)                                                 # Equation 20
    
    alpha_r = delta - np.arctan(vr/ur)                             # Equation 21   
    
    F_normal =   (0.5) * rho * Ar * (Urs) * f_alpha * sin(alpha_r)   # Equation 19
       
    
    Xrud = -Xrc*F_normal*sin(delta)
    Yrud = -Yrc*F_normal*cos(delta)
    Nrud = -Nrc*F_normal*cos(delta)
    
    F_rudder = np.array([[Xrud],[Yrud],[Nrud]])

    
    ############## Solving x_dot = A_inv*b(x) #################
    F = F_prop + F_hull + F_rudder                                             # Equation 5
    eom = np.array([[-(m+my)*v*r-xG*m*(r**2)],
                              [(m+mx)*u*r],
                              [m*xG*u*r]] )                                    # Equation 4

    b = F - eom
    vd = Minv@b
    x_dot = u*cos(psi)-v*sin(psi)                                             # Transformation matrix
    y_dot = u*sin(psi)+v*cos(psi)
    psi_dot = r
    delta_dot = np.sign(delta_c-delta)*1.76*pi/180 

    
    
    
    der = np.zeros(8)

    der[0] = vd[0][0]
    der[1] = vd[1][0]
    der[2] = vd[2][0]
    der[3] = x_dot
    der[4] = y_dot
    der[5] = psi_dot
    der[6] = delta_dot
    der[7] = 0

    var[0] = der[0]*t + u
    var[1] = der[1]*t + v
    var[2] = der[2]*t + r
    var[3] = der[3]*t + x
    var[4] = der[4]*t + y
    var[5] = der[5]*t + psi
    var[6] = der[6]*t + delta
    var[7] = Np
    

    return var




X0 = [7.97319,0,0,0,0,0,0,1.77835763]
funcp = lambda t,x:_mmgder(t,x,-35)
funcs = lambda t,x:_mmgder(t,x,35)
Tmax = 1600
tspan = [0,Tmax]
t_eval = np.arange(0,Tmax,0.1)
dt = 0.1
total_iteration = int(Tmax/dt)

i = 0
states = np.zeros([total_iteration,8])
delta_com = []
dc = 10
while i < total_iteration:
    var = _mmgder(dt,X0,dc)
    X0[:] = var[:]
    states[i,:] = var[:]

    if var[5]*180/pi>=abs(dc):
        dc = -10
    if var[5]*180/pi<=-abs(dc):
        dc = 10
    delta_com.append(dc)
    i+=1

psi = states[:,5]*180/pi
delta = states[:,6]*180/pi
plt.plot(t_eval,psi,label="psi")
plt.plot(t_eval,delta,label="delta")
plt.plot(t_eval,delta_com,label="del_c")
plt.title("Zigzag Maneuver [10/10]")
plt.xlabel("time(s)")
plt.ylabel("Degrees")
plt.legend()
plt.show()

r = states[:,2]
u = states[:,0]
h = dt
psi_n = psi[2:15997]
psi_1 = psi[1:15996]
psi_2 = psi[0:15995]

y = psi_n - psi_1
y_1 = psi_1 - psi_2
Y = y - y_1
delta_2 = delta[0:15995]


x = np.zeros([4,15995])
x[0,:] = delta_2*h**2 
x[1,:] = h**2
x[2,:] = h*y_1
x[3,:] = y_1**3/h

A = np.linalg.inv(x@x.T)@x@Y

T = -1/A[2]
K = A[0]*T
c = -A[3]*T
b = A[1]*T

print("parameters are: \
      T:",T,  \
      "K:",K, \
      "Bias:",b, \
      "Nonlinear coeff:",c     
      )



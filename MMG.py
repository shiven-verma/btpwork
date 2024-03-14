#!/usr/bin/env python3

import numpy as np
from math import exp,sqrt,pi,sin,cos

from ModelCaliberation import expCal
from params import fsparam

p = fsparam()
c = expCal()

# Np = p.Np

massndm = 0.5*p.rho*p.d*p.Lpp**2 
Indm = 0.5*p.rho*p.d*p.Lpp**4
mx = p.mx*massndm
my = p.my*massndm
Jz = p.Jz*Indm

wp0 = c.wp0 if p.Lpp>300 else c.wpm0

M = np.array([[(p.m+mx),0,0],[0,(p.m+my),p.m*p.xG],[0,p.m*p.xG,(Jz+((p.xG**2)*p.m)+p.Iz)]])           # Mass Matrix
Minv = np.linalg.inv(M)                                                                               # Inverse of Mass Matrix


mindel = -35
maxdel = 35
mindel_r = mindel*pi/180
maxdel_r = maxdel*pi/180


def _mmgder(t,var,dc):
    
    ###### States are non-dimensional #####
    u = var[0]                # Surge velocity
    v = var[1]                # Lateral sway velocity
    r = var[2]                # Yaw rate        
    psi = var[5]              # Yaw Angle
    delta = var[6]            # Rudder Angle 
    Np = p.Np               # Propeller Speed
    
    
    Ures = sqrt(u**2 + v**2)     # Resulatant Velocity
    beta = -np.arctan(v/u)
    
    U = Ures
    
    delta_c = dc
    delta_c = np.clip(delta_c,mindel,maxdel)             # Rudder Angle Commanded 
    delta_c = delta_c*pi/180                             # Degree to Radian
    delta = np.clip(delta,mindel_r,maxdel_r)             # Constrains on Delta

    

    ### Non-dimesnional constants ###
    Fndmc = (0.5)*p.rho*p.Lpp*p.d*(U**2)
    Mndmc = (0.5)*p.rho*p.Lpp*p.Lpp*p.d*(U**2)
    
    
    
    ############ Hull Forces ##############
    ################ Equation 7 #############
    up = u/U                        # up is u prime and non dimensional
    vm = v/U                        # vm is non-dimensional
    rp = r*p.Lpp/U                  # rp is non-dimensional
    
    Xhnd = -c.res + c.Xvv*(vm**2) + c.Xvr*vm*rp + c.Xrr*(rp**2) + c.Xvvvv*vm**4
    Yhnd = c.Yv*vm + c.Yr*rp + c.Yvvv*vm**3 + c.Yvvr*rp*vm**2 + c.Yvrr*vm*rp**2 + c.Yrrr*rp**3
    Nhnd = c.Nv*vm + c.Nr*rp + c.Nvvv*vm**3 + c.Nvvr*rp*vm**2 + c.Nvrr*vm*rp**2 + c.Nrrr*rp**3

    Xh = Xhnd*Fndmc          
    Yh = Yhnd*Fndmc
    Nh = Nhnd*Mndmc
    F_hull = np.array([[Xh],[Yh],[Nh]])
    
    
    ################## Propeller Forces ###########################
    
    beta_p = beta - (c.x_p*rp)                                        # Eq. 15
    
    C1 = 2.0                                                       # Table 3
    C2 = 0.5*np.heaviside(beta_p,0)+1.1                            # Table 3
    
    wp  =  1 - (1+(1-exp(-C1*abs(beta_p)))*(C2-1))*(1-wp0)         # Wake Coeff. Eq. 16
    Jp = u*(1-wp)/(Np*p.Dp)                                        # Propeller Advanced ratio Eq. 11
    KT = c.k2*(Jp**2) + c.k1*Jp + c.k0                                   # Eq. 10
    
    Thrust = p.rho*(Np**2)*(p.Dp**4)*KT                                # Dimensional Thrust Eq. 9
    Xp = (1-c.tp)*Thrust                                             # Surge Force Eq. 8
    F_prop = np.array([[Xp],[0],[0]])
    
    
    
    ################# Rudder Force ##############################
    
    beta_r = beta - c.lr*rp                                          # Equation 24
    gamma_r = 0.395 if beta_r<0 else 0.640                               # Table 3
    vr = Ures*gamma_r*beta_r                                       # Equation 23
    eta = p.Dp/p.Hr
    uProp = (1 - wp)*u                                             # Equation 25
    uR1 = sqrt(1 + 8*KT/(pi*Jp**2))                                #
    uR2 = (1 + c.kappa*(uR1 -1))**2
    ur  = c.epsilon * uProp * sqrt((eta*uR2) + (1-eta))              #
    Urs = vr**2 + ur**2                                            
    Ur = sqrt(Urs)                                                 # Equation 20
    
    alpha_r = delta - np.arctan(vr/ur)                             # Equation 21   
    
    F_normal =   (0.5) * p.rho * p.Ar * (Urs) * c.f_alpha * sin(alpha_r)   # Equation 19
       
    
    Xrud = -c.Xrc*F_normal*sin(delta)
    Yrud = -c.Yrc*F_normal*cos(delta)
    Nrud = -c.Nrc*F_normal*p.Lpp*cos(delta)
    
    F_rudder = np.array([[Xrud],[Yrud],[Nrud]])

    
    ############## Solving x_dot = A_inv*b(x) #################
    F = F_prop + F_hull + F_rudder                                             # Equation 5
    eom = np.array([[-(p.m+my)*v*r-p.xG*p.m*(r**2)],
                              [(p.m+mx)*u*r],
                              [p.m*p.xG*u*r]] )                                    # Equation 4

    b = F - eom
    vd = Minv@b
    x_dot = u*cos(psi)-v*sin(psi)                                             # Transformation matrix
    y_dot = u*sin(psi)+v*cos(psi)
    psi_dot = r
    delta_dot = np.sign(delta_c-delta)*1.76*pi/180 
    delta_dot = 0.9*(delta_c-delta)
    
    
    
    X_dot = np.zeros(7)
    
    X_dot[0] = vd[0]
    X_dot[1] = vd[1]
    X_dot[2] = vd[2]
    X_dot[3] = x_dot
    X_dot[4] = y_dot
    X_dot[5] = psi_dot
    X_dot[6] = delta_dot
    # X_dot[7] = 0
    
    h = t[1] - t[0]
    return X_dot*h + var




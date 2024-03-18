import numpy as np
from math import sin, cos, pi, sqrt, exp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

######### Ship Parameters ##########
########## Table 1 ##########
U0 = 0.76          # Design Speed
Lpp = 2.902          # Length of the Ship
B = 0.527             # Beam 
d = 0.189           # Draft
disp = 0.235          # Volume Displaced
Cb = 0.810         # Block Coefficient
Dp = 0.090          # Diameter of propeller
Hr = 0.144           # Rudder span length
Ar = 0.00928         # Area of Rudder
rho = 1025         # Density of water
rgy = 0.25*Lpp     # Radius of gyration
xG =  0.011     # Centre of gravity


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
Np_d = 17.95
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
mindel = -35
maxdel = 35
mindel_r = mindel*pi/180
maxdel_r = maxdel*pi/180


def _mmgder(var,dc):
    
    ###### States are non-dimensional #####
    u = var[0]                # Surge velocity
    v = var[1]                # Lateral sway velocity
    r = var[2]                # Yaw rate        
    psi = var[5]              # Yaw Angle
    delta = var[6]            # Rudder Angle 
    # Np = var[7]               # Propeller Speed
    Np = Np_d
    
    
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
    # print("Xh:",Xh)
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
    # print(Xp)
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
    # print("F:",F)
    eom = np.array([[-(m+my)*v*r-xG*m*(r**2)],
                              [(m+mx)*u*r],
                              [m*xG*u*r]] )                                    # Equation 4

    b = F - eom
    vd = Minv@b
    # print("Vd:",vd)
    x_dot = u*cos(psi)-v*sin(psi)                                             # Transformation matrix
    y_dot = u*sin(psi)+v*cos(psi)
    psi_dot = r
    delta_dot = np.sign(delta_c-delta)*25.76*pi/180 
    
    
    
    der = np.zeros(7)
    
    der[0] = vd[0]
    der[1] = vd[1]
    der[2] = vd[2]
    der[3] = x_dot
    der[4] = y_dot
    der[5] = psi_dot
    der[6] = delta_dot
    # der[7] = 0
    
    return der

# X0 = [U0,0,0,0,0,0,0,17.95]
# funcp = lambda t,x:_mmgder(t,x,-35)
# funcs = lambda t,x:_mmgder(t,x,35)
# Tmax = 160
# tspan = [0,Tmax]
# t_eval = np.arange(0,Tmax,0.01)


# sol_port = solve_ivp(funcp,tspan,X0,t_eval=t_eval)
# sol_stbd = solve_ivp(funcs,tspan,X0,t_eval=t_eval)

def simulation(X0,control,t):
    h = t[1]-t[0]
    n = control.shape[0]
    sol = np.zeros([7,n])
    i = 0
    xinit = X0.copy()
    while i<n:
        sol[:,i] = xinit[:]
        xd = _mmgder(xinit,control[i])
        xup = xd*h + xinit
        xinit[:] = xup
        i+=1
    return sol
    



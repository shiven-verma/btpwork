# Do it for 4 states
import numpy as np
import casadi as cd
import matplotlib.pyplot as plt


#MPC Parameters

M = 20    # Control Horizon
P = 20     # Prediction Horizon
Q = np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])     # Cost Weight
Rf = 55   # Reference 
delT = 1  # Discrete Time stamp


tcontinuous = np.linspace(0, P*delT, 1000) 
tpredict = np.arange(0, P*delT, delT)  
tref = np.linspace(0,Rf,Rf)


def extend(u):
    return np.concatenate([u, np.repeat(u[-1], P-M)])

#Ship Parameters
#U0 = 1  # m/s

def model(X,u,h,U):
    psi = X[0]
    r = X[1]
    x = X[2]
    y = X[3]
    
    delta = u
    
    bias = 0.0
    K = 0.7
    T = 4
    c = 0.1

    x_d = np.zeros(4)
    
    x_d[0] = r
    x_d[1] = (K*delta + bias -c*r**3 - r)/T   # Non-linear
    x_d[2] = U*np.cos(psi)
    x_d[3] = U*np.sin(psi)

    x_n = np.zeros(4)

    x_n[0] = x_d[0]*h + psi
    x_n[1] = x_d[1]*h + r
    x_n[2] = U*np.cos(psi)*h + x
    x_n[3] = U*np.sin(psi)*h + y
    
    #x_new_1 is psi
    #x_new_2 is yaw rate r

    return x_n

# u - input/delta/rudder angle
# U0 - fixed surge velocity
# X - state
# x - position in x coordinate

def simulation(u,xm0,U,t=tpredict):
    h = t[1]-t[0]
    n = t.shape[0]
    y = np.zeros([4,n])
    i = 0
    x_init = []
    x_init[:] = xm0[:]
    while i<n:
        y[:,i] = x_init[:]
        x_up = model(x_init,u[i],h,U)
        x_init[:] = x_up[:]
        #y[:,i]= x_up[:]
        i = i+1
    return y


# x0 = [0,0,0,0]
# u = [0.610]*30
# u = np.array(u)
# ucont = u[((tcontinuous-0.01)//delT).astype(int)]
# sol = simulation(ucont,x0,tcontinuous)
def circle(a,b,r):
    theta = np.linspace(0,2*np.pi,100)

    x = a + r*np.cos(theta)
    y = b + r*np.sin(theta)

    xo = a+r*0.3*np.cos(theta)
    yo = b+r*0.3*np.sin(theta)

    plt.plot(x,y,"--r",xo,yo,"b")

# # Combined Cost Function
def stolist(N,li1,li2):
    i = 0
    emptli = []
    while i<N:
        emptli.append(li1[i])
        i = i+1
    for x in li2:
        emptli.append(x)
    return emptli 


def cost(xref,u,x0,Us,t=tpredict):
    opt_var = u.shape[0]
    h = t[1] - t[0]
    uv = cd.SX.sym("uv",opt_var)
    u_P = np.zeros(extend(u).shape[0]-u.shape[0])
    xvar = stolist(u.shape[0],uv,u_P)
    xref = cd.SX(xref)
    u0 = u[:]
    xini = x0.copy()
    
    # Put Prediction and Model here
    j = 0
    ye = cd.SX(1,4)
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    while j<extend(u).shape[0]:
        psi = xini[0]
        r = xini[1]
        x = xini[2]
        y = xini[3]
        
        bias = 0.01
        K = 0.5
        T = 4
        c = 0.1

        U = Us
        
        psi_new = psi + r*h
        r_new = r + ((K*xvar[j]+ bias -0*c*r**3 - r)/T)*h
        x_new = x + U*np.cos(psi)
        y_new = y + U*np.sin(psi)

        xini[0] = psi_new 
        xini[1] = r_new
        xini[2] = x_new
        xini[3] = y_new

        xn = cd.SX(1,4)
        xn[0,0] = psi_new
        xn[0,1] = r_new
        xn[0,2] = x_new
        xn[0,3] = y_new
        ye = cd.vertcat(ye,xn)

        
        temp1.append(psi_new)
        temp2.append(r_new)
        temp3.append(x_new)
        temp4.append(y_new)
        
        j = j+1
        
    #### Model is working fine, it gives 20 values for both the states, but we need to optimize only 10
    
    temp = [temp1,temp2,temp3,temp4]
    ye = ye[1:,:]
    print(ye.shape)
    #print(cd.sum2(ye[:,0]-xref[:,0]))
    
    xpre = ye[:,2]
    ypre = ye[:,3]
    r = 1.5
    a = [10,20,15]
    b = [0.8,5,2.5]
    
    f = cd.sum1(0.002*cd.sum2((xref[:,0]-ye[:,0])**2) + cd.sum2((xref[:,2]-ye[:,2])**2) + cd.sum2((xref[:,3]-ye[:,3])**2)) + cd.sum1(uv) #+ 1.11*cd.sum1(1/((xpre-20)**2*(ypre-2.5)**2)) #
    # print(f.shape," :f")
    #f = 0.002*sum((xref[:,0]-temp1)**2) + sum((xref[:,2]-temp3)**2) + sum((xref[:,3]-temp4)**2)
    lbx = [-0.610]*u.shape[0]
    ubx = [ 0.610]*u.shape[0]
    p = [10,1,0.5]
    # cx = np.array([10]*opt_var)
    # cy = np.array([1]*opt_var)
    # r = np.array([0.5]*opt_var)
    #g = [(temp3-10)**2 + (temp4-1)**2]

    
    k = [r**2 - a[0]**2 - b[0]**2, r**2 - a[1]**2 - b[1]**2,  r**2 - a[2]**2 - b[2]**2]

    
    # g = [xpre**2 + ypre**2 - 2*a*xpre - 2*b*ypre,xpre**2]
    # print(g," :g")
    g = cd.vertcat(xpre**2 + ypre**2 - 2*a[0]*xpre - 2*b[0]*ypre, xpre**2 + ypre**2 - 2*a[1]*xpre - 2*b[1]*ypre, xpre**2 + ypre**2 - 2*a[2]*xpre - 2*b[2]*ypre )
    lbg = cd.vertcat(np.ones(P)*k[0],np.ones(P)*k[1],np.ones(P)*k[2])
    # lbg = [k,0.00]
    ubg = [cd.inf,cd.inf,cd.inf]
    print()
    nlp = {"x":uv,"f":f,"g":g}
    solver = cd.nlpsol('solver', 'ipopt', nlp)
    solved = solver(x0 = u0,lbx = lbx,ubx=ubx,lbg=lbg)
    print(solved['x'],solved['f'],"cost")
    return solved['x']



def past(ar,n):
    li = []
    li[:] = ar[:]
    li = [0]*n + li[:-n]
    return np.array(li)

xref = tref.copy()
curve = (xref/10)**2
yref = curve.copy()
xrefd = xref - past(xref,1)
yrefd = yref - past(yref,1)
psiref = np.arctan2(yrefd,xrefd)
rref = np.zeros(Rf)

Xref = np.array([psiref,rref,xref,yref]).T
x0 = [0,0,0,0]
x_ini = x0.copy()
uopt = np.ones([Rf-P,M])
k = 0

U0 = 1
while k<(Rf-P):
    u = np.zeros(M)
    refr = Xref[k:k+P]
    Us = U0
    uoptd = cost(refr,u,x_ini,Us)
    uopt[k,:] = np.array(uoptd)[:,0]
    uoptm = uopt[k,:]
    y = simulation(extend(uoptm),x_ini,Us)
    x_ini[:] = y[:,1]
    k+=1
    

simn = Rf-P
i = 0
simdict = {}
xm0 = x0.copy()
initx = xm0
while i<simn:
    var = f"pred_{i}"
    value = simulation(extend(uopt[i,:]),initx,U0)
    initx = value[:,1]
    simdict[var] = value    
    i+=1

plt.figure(figsize=(16,12))
plt.plot(yref,xref,label="reference")
j = 0
while j<simn:
    if j== simn-1:
        plt.plot(simdict[f"pred_{j}"][3,:],simdict[f"pred_{j}"][2,:], "-o", label = f"iter{j+1}")
    else:
        plt.plot(simdict[f"pred_{j}"][3,:][0:2],simdict[f"pred_{j}"][2,:][0:2], "-o", label = f"iter{j+1}")
        # plt.plot(simdict[f"pred_{j}"][3,:], simdict[f"pred_{j}"][2,:], "-o", label = f"iter{j+1}")
    j+=1

circle(0.8,10,1.5)
circle(5,20,1.5)
circle(2.5,15,1.5)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.legend()
plt.title("MPC Path predictions(with control cost)")
plt.xlabel("Y")
plt.ylabel("X")
plt.show()
# plt.savefig("mpcobs_cont.png")
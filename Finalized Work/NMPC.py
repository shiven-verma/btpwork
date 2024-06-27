# !/usr/bin/env python3

import numpy as np
import casadi as cd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from math import sin,cos,sqrt,pi,exp

              
class controller():
    def __init__(self,NP,NC,Q,obst_r):
        self.P = NP
        self.C = NC
        self.Q = Q
        self.r = obst_r

    def prediction_model(self,states,control_inp,h):
        u = states[0]
        v = states[1]
        r = states[2]
        x = states[3]
        y = states[4]
        psi = states[5]
        delta = states[6]

        T = 14
        K = 0.061
        a = -68.05
        b = 0.0022*0

        Kp = 0.9
        state_derivative = cd.SX(7,1)

        state_derivative[0] = 0 
        state_derivative[1] = 0 
        state_derivative[2] = (K*delta + b -a*r**3 - r)/T   # Non-linear
        state_derivative[3] = u*cd.cos(psi) 
        state_derivative[4] = u*cd.sin(psi) 
        state_derivative[5] = r  
        state_derivative[6] = Kp*(control_inp-delta)*25.76*pi/180

        new_state = state_derivative*h + states

        return new_state
    
    def cost(self,ref,X0,t,obs_pos): #X0 initial states 
        h = t[1]-t[0] 
        self.control_variable = cd.SX.sym("control_variable",self.P)
        xinit = X0.copy()
        prediction = cd.SX(1,7)
        for i in range(self.P):
            newstate = self.prediction_model(xinit,self.control_variable[i],h)
            xinit = newstate
            prediction = cd.vertcat(prediction,newstate.T)
        prediction = prediction[1:,:]
        x_prediction = prediction[:,3]
        y_prediction = prediction[:,4]
        psi_prediction = prediction[:,5]


        [cx,cy] = obs_pos
        r = self.r
        self.k = []
        self.g = []
        self.lbg = []
        for i in range(len(r)):
            self.k.append(r[i]**2 - cx[i]**2 - cy[i]**2)
            self.g = cd.vertcat(self.g,x_prediction**2 + y_prediction**2 - 2*cx[i]*x_prediction - 2*cy[i]*y_prediction)
            self.lbg = cd.vertcat(self.lbg,np.ones(self.P)*self.k[i])
        
        if self.P > ref.shape[0]:
            f = cd.sum1((ref[-1,0]-x_prediction[-1])**2 + (ref[-1,1]-y_prediction[-1])**2 + (ref[-1,2]-psi_prediction[-1]**2)) + cd.sum1(self.control_variable**2)
        else:
            f = cd.sum1(5*cd.sum2((ref[:,0]-x_prediction[0:ref.shape[0]])**2) + 
                    5*cd.sum2((ref[:,1]-y_prediction[0:ref.shape[0]])**2) + 
                    2*cd.sum2((ref[:,2]-psi_prediction[0:ref.shape[0]]))**2) + cd.sum1(self.control_variable**2)

        return f
    

    def nlpsolve(self,ref,X,t,obs_pos):
        X0 = X.copy()
        u0 = np.zeros(self.P)
        lbx = [-0.610]*self.P
        ubx = [0.610]*self.P

        wli = [1/(i+1) for i in range(self.P)]
        weight = cd.DM(wli[::-1])

        
        f = self.cost(ref,X0,t,obs_pos)
        g = self.g
        lbg = self.lbg
        ubg = [cd.inf,cd.inf,cd.inf]

        nlp = {"x":self.control_variable,"f":f, "g":g}
        solver = cd.nlpsol('solver','ipopt',nlp)
        solved = solver(x0 = u0,lbx = lbx,ubx=ubx,lbg=lbg)
        # print(solved['x'])
        return solved['x']
    


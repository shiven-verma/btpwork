#!/usr/bin/env python3

import numpy as np
import casadi as cd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from math import sin,cos,sqrt,pi,exp



class ship():
    def __init__(self):
        self.T = 4 # Time constant
        self.K = 0.5# rudder constant
        self.a = 0.1 # nonlinear term
        self.b = 0.01 # bias term


    def model(self,state,input):
        x = state[0]
        y = state[1]
        psi = state[2]
        u = state[3]
        v = state[4]
        r = state[5]
        delta = state[6]

        Kp = 0.9
        input = np.clip(input,-0.610,0.610)
        stateder = np.zeros(7)

        stateder[0] = u*cos(psi)
        stateder[1] = u*sin(psi)
        stateder[2] = r
        stateder[3] = 0
        stateder[4] = 0
        stateder[5] = (self.K*delta + self.b -self.a*r**3 - r)/self.T   # Non-linear
        stateder[6] = Kp*(input-delta)

        return stateder
    
    def simulation(self,X0,control,t):
        # if len(control) == len(t):
        #     raise TypeError("Dimensions does not match")
        # h = t[1]-t[0]
        h = t
        n = control.shape[0]
        sol = np.zeros([7,n])
        i = 0
        xinit = X0.copy()
        while i<n:
            sol[:,i] = xinit[:]
            xd = self.model(xinit,control[i])
            xup = xd*h + xinit
            xinit[:] = xup
            i+=1
        return sol
        

       
class controller():
    def __init__(self):
        self.P = 10
        self.C = 1
        self.Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])


    def prediction_model(self,states,control_inp,h):
        x = states[0]
        y = states[1]
        psi = states[2]
        u = states[3]
        v = states[4]
        r = states[5]
        delta = states[6]

        T = 4
        K = 0.5
        a = 0.1
        b = 0.01

        Kp = 0.9
        #input = np.clip(control_inp,-0.610,0.610)
        stder = cd.SX(7,1)

        stder[0] = u*cd.cos(psi)
        stder[1] = u*cd.sin(psi)
        stder[2] = r
        stder[3] = 0
        stder[4] = 0
        stder[5] = (K*delta + b -a*r**3 - r)/T   # Non-linear
        stder[6] = Kp*(control_inp-delta)

        stnew = stder*h + states
        #print("stn: ",stnew)

        return stnew
    
    def cost(self,ref,X0,t): #X0 initial states , t =
        n = self.P
        h = t[1]-t[0] 
        self.control_variable = cd.SX.sym("control_variable",n)
        xinit = X0.copy()
        y = cd.SX(1,7)
        i = 0
        while i<n:
            newstate = self.prediction_model(xinit,self.control_variable[i],h)
            xinit = newstate
            # print(newstate.T.shape, y.shape)
            y = cd.vertcat(y,newstate.T)
            i+=1
        y = y[1:,:]
        x_pred = y[:,0]
        y_pred = y[:,1]
        psi_pred = y[:,2]
        print("Reference size: ",ref.shape)
        f = cd.sum1(cd.sum2((ref[:,0]-x_pred)**2) + cd.sum2((ref[:,1]-y_pred)**2) + cd.sum2((ref[:,2]-psi_pred))**2)
        return f
    

    def nlpsolve(self,ref,X,t):
        X0 = X.copy()
        u0 = np.zeros(10)
        lbx = [-0.610]*self.P
        ubx = [0.610]*self.P
        P = []
        g = [0]
        ubg = [0]
        lbg = [0]
        f = self.cost(ref,X0,t)

        nlp = {"x":self.control_variable,"f":f, "g":g}
        solver = cd.nlpsol('solver','ipopt',nlp)
        solved = solver(x0 = u0,lbx = lbx,ubx=ubx)
        print(solved['x'])
        return solved['x']
    


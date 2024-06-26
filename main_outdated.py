
#import libraries
from casadi import SX
import numpy as np
# import pygame
from math import sqrt,pi
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# import supporting functions
from MMG3 import simulation
from NMPC import controller
from Guide import Serret_Frenet_Guidance,Agent



class MainSimulation:
    def __init__(self,xpoints,ypoints,initial_state,T):
        self.x0 = initial_state[3]
        self.y0 = initial_state[4]
        self.psi0 = initial_state[5]
        self.X0 = initial_state.copy()
        self.xpoints = xpoints
        self.ypoints = ypoints
        self.T = T

    def generate_reference(self):
        self.spline = CubicSpline(self.xpoints,self.ypoints,bc_type="clamped")
        self.xspl = np.linspace(min(self.xpoints),max(self.xpoints),926)
        self.yspl = self.spline(self.xspl)
        self.path_step = 10
        time = np.linspace(0,self.T,self.T*self.path_step)
        A = Agent([self.x0,self.y0,self.psi0])
        self.Reference = A.simulation(self.spline,time,self.xspl,np.array(self.X0.copy()))
        # print(self.spline,time.shape,self.xspl.shape,np.array(self.X0.copy()))
        self.Reference = self.Reference[:,::self.path_step]
        ReferenceWindow = self.Reference.shape[1]
        self.SimulationWindow = ReferenceWindow 

        if ReferenceWindow>400:
            raise("error")
        
        
    def circle(self,obstacle_pos):
        cx = obstacle_pos[0]
        cy = obstacle_pos[1]
        radius = obstacle_pos[2]

        theta = np.linspace(0,2*pi,100)
        for i in range(len(cx)):
            x = cx[i] + radius[i]*np.cos(theta)
            y = cy[i] + radius[i]*np.sin(theta)
            xo = cx[i] + 0.3*radius[i]*np.cos(theta)
            yo = cy[i] + 0.3*radius[i]*np.sin(theta)
            plt.plot(x,y,"--r",xo,yo,"b")

    def obstacle_moves(self,x0,y0,psi,v):
        x = np.array(x0) + v*np.cos(psi)
        y = np.array(y0) + v*np.sin(psi)
        return [x,y]


    def RunTheShip(self,NC):
        self.SimulationDataDict = {}
        self.generate_reference()
        
        uopt = np.ones((self.SimulationWindow,NP))
        ContMag = 1
        tdiscrete = np.linspace(0,NP-1,NP)
        tcontinuous = np.linspace(0,NP-1,NP*ContMag)
        Xinit = self.X0.copy()
        flag = 0
        cte_list = []
        i = 0
        while True:
            print(self.Reference.shape[:],"--Refshape")
            Refer = self.Reference.T[i:i+NP*NC:NC,:]
            print(Refer.shape[:],"--Refshape")
            optimized_control_input = C.nlpsolve(Refer,Xinit,tcontinuous,self.obstacle_moves)
            # print(optimized_control_input)
            uoptimal = uopt[i,:] = np.array(optimized_control_input)[:,0]
            uoptimal_continuous = uoptimal[(tcontinuous//1).astype(int)]
            if i==self.SimulationWindow-1+1:
                flag = 1
            predicted_steps = simulation(Xinit,uoptimal_continuous,tcontinuous,flag)
            variable = f"pred_{i}"
            self.SimulationDataDict[variable] = predicted_steps
            Xinit[:] = predicted_steps[:,1*NC*ContMag]
            ###########
            # Dynamic NC and NP
            obs_dist = (np.array(obstacle[0])-Xinit[3])**2 + (np.array(obstacle[1])-Xinit[4])**2
            CTE = abs(Xinit[4]-Refer[0,1])
            if CTE>1:
                NC = 2
            else:
                NC = 1
            if Refer.shape[0]==1:
                break
            
            # if (obs_dist<NP*Xinit[0]).any():
            #     print("obs detected")
            

            cte_list.append(CTE)
            # CTE = abs(Xinit[1]-Refer[NC,1])
            # print(CTE)
            print(i,"iter out of", self.SimulationWindow-1)
            i = i+1*NC
        plt.plot(range(self.SimulationWindow-2),cte_list)

    def Visualiser(self):
        plt.figure(figsize=(16,12))
        plt.plot(self.xspl,self.yspl,"--",label="reference")
        plt.plot(self.xpoints,self.ypoints,"r*",label="ScorePoints")
        for i in range(self.SimulationWindow-2):
            if i== self.SimulationWindow-3:
                plt.plot(self.SimulationDataDict[f"pred_{i}"][3,:],self.SimulationDataDict[f"pred_{i}"][4,:], "-")
            else:
                plt.plot(self.SimulationDataDict[f"pred_{i}"][3,:][[0,1]],self.SimulationDataDict[f"pred_{i}"][4,:][[0,1]], "-", )
                # plt.plot(self.SimulationDataDict[f"pred_{i}"][3,:],self.SimulationDataDict[f"pred_{i}"][4,:], "--", )
        ax = plt.gca()
        ax.set_aspect('equal',adjustable='box')
        # plt.legend()
        plt.title('Model Predictive Control Simulation')
        plt.xlabel("y")
        plt.ylabel("x")



if __name__ == "__main__":
    xspl = np.array([0,5,9,17,27,39,53,64,74,86,100])
    yspl = np.array([-1,0.5,1.2,2.3,3.0,3.6,5.2,4.3,3.9,3.0,2.4])

    x0 = -5
    y0 = 0.5
    psi0 = 0.0
    U0 = 0.5

    initial_state = [U0,0,0,x0,y0,psi0,0]
    T = 400

    #Declare Controller
    NP = 30
    NC = 1
    Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    #obstacle [x,y,r]
    obstacle = [[50,134],[2.5*2,4*2],[1.5,2.5]]
    C = controller(NP,NC,Q,obstacle)

    main = MainSimulation(xspl,yspl,initial_state,T)
    main.RunTheShip(NC)
    main.Visualiser()
    main.circle(obstacle)
    # plt.savefig("OP30C2.png", bbox_inches='tight', pad_inches=0)
    plt.show()
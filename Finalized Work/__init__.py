
## Import Libraries
from casadi import SX
import numpy as np
from math import sqrt,pi
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time

## Import Supporting Functions
from MMG3 import simulation
from NMPC import controller
from Guide import Agent

class EnvironmentData():
    def __init__(self,data):
        self.ObstaclePositionArray = np.array([data[0],data[1]])
        self.ShipPositionArray = np.array([data[2],data[3]])
        self.Ref = data[4]
        self.OptTime = data[5]


class MainSimulation:
    def __init__(self,xpoints,ypoints,initial_state,T,obstacle):
        self.x0 = initial_state[3]
        self.y0 = initial_state[4]
        self.psi0 = initial_state[5]
        self.X0 = initial_state.copy()
        self.xpoints = xpoints
        self.ypoints = ypoints
        self.T = T
        self.obsx = obstacle[0]
        self.obsy = obstacle[1]
        self.radius = obstacle[2]

    def generate_reference(self):
        self.spline = CubicSpline(self.xpoints,self.ypoints,bc_type="clamped")
        self.xspl = np.linspace(min(self.xpoints),max(self.xpoints),926)
        self.yspl = self.spline(self.xspl)
        path_step = 10
        time = np.linspace(0,self.T,self.T*path_step)
        A = Agent([self.x0,self.y0,self.psi0])
        Reference = A.simulation(self.spline,time,self.xspl,np.array(self.X0.copy()))
        # print(self.spline,time.shape,self.xspl.shape,np.array(self.X0.copy()))
        self.Reference = Reference[:,::path_step]
        ReferenceWindow = self.Reference.shape[1]
        self.SimulationWindow = ReferenceWindow 
        
        
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
    
    def CrossTrackError(self,Reference,Pose):
        CTE = Pose[4]-Reference[0,1]
        return CTE

    def RunTheShip(self,NC):
        self.SimulationDataDict = {}
        self.generate_reference()
        EnvData = np.zeros([6,self.SimulationWindow])
        self.i_list = []
        
    
        uopt = np.ones((self.SimulationWindow,NP))
        
        tcontinuous = np.linspace(0,NP-1,NP)
        Xinit = self.X0
        flag = 0

        i = 0
        j = 0
        v = 0.2
        while i<self.SimulationWindow:
            obs_pos = self.obstacle_moves(self.obsx,self.obsy,np.pi,v)
            [self.obsx,self.obsy] = obs_pos[:]
            Refer = self.Reference.T[i:i+NP*NC:NC,:]

            ## Optimization and Calculating Time
            start_time = 0
            start_time = time.time()
            optimized_control_input = C.nlpsolve(Refer,Xinit,tcontinuous,obs_pos)
            end_time = time.time()

            uoptimal = uopt[i,:] = np.array(optimized_control_input)[:,0]
            # uoptimal_continuous = uoptimal[(tcontinuous//1).astype(int)]

            ## MMG simulation
            if i==self.SimulationWindow:
                flag = 1
            predicted_steps = simulation(Xinit,uoptimal,tcontinuous,flag)


            variable = f"pred_{i}"
            self.SimulationDataDict[variable] = predicted_steps
            Xinit[:] = predicted_steps[:,NC]


            ## Dynamic NC Condition
            CTE = self.CrossTrackError(Refer,Xinit)
            obs_dist = (self.obsx[0]-Xinit[3])**2 + (self.obsy[0]-Xinit[4])**2
            if (CTE*NP*2>obs_dist).any() or CTE>2:
                NC = 1
                j+=1
            else:
                NC = 1
            optime = end_time-start_time
            templist = [self.obsx[0],self.obsy[0],Xinit[3],Xinit[4],Refer[0,1],optime]
            # print(self.obsx[0])
            EnvData[:,i] = templist[:]
            self.i_list.append(i)
            print(i,"iter out of", self.SimulationWindow-1,"--",j)
            i = i+1*NC
        Env = EnvironmentData(EnvData)
        return Env
    
    def Visualiser(self):
        plt.figure(figsize=(16,12))
        plt.plot(self.xspl,self.yspl,"--",label="reference")
        plt.plot(self.xpoints,self.ypoints,"r*",label="ScorePoints")
        for i in self.i_list:
            if i == self.i_list[-1]:
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

    xspl = np.array([0,4,10,18,29,40,57,64,75,87,97])
    yspl = np.array([2,1.3,0.8,1.3,2.0,2.8,4.2,4.8,5.3,6.0,5.4])

    x0 = -4
    y0 = -1.5
    psi0 = 0.0
    U0 = 0.5

    initial_state = [U0,0,0,x0,y0,psi0,0]
    T = 400

    #Declare Controller
    NP = 45
    NC = 1
    Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    #obstacle [x,y,r]
    obstacle = [[74],[4.3],[1.5]]
    C = controller(NP,NC,Q,obstacle[2])

    main = MainSimulation(xspl,yspl,initial_state,T,obstacle)
    env = main.RunTheShip(NC)
    main.Visualiser()
    main.circle(obstacle)
    print(max(env.OptTime))
    # plt.savefig("OP30C2.png", bbox_inches='tight', pad_inches=0)
    plt.show()

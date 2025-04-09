import dis
from turtle import width
from casadi import *
# from pypoman import compute_polytope_halfspaces
import math as m
import numpy as np
import time

class dms_planner:
    '''
    基于优化的规划器：
    q: 状态量的权重
    r: 控制量的权重
    '''
    def __init__(self, dt = 0.1, Np=15):
        self.dt = dt 
        self.Np = Np
        # self.maxDeltaRate = 1.4 # 最大前轮转角速率rad/s
        # self.maxA = 4 # 最大加速度
        # self.minA = -6 # 最小减速度
        self.maxjerk = 48 # 最大jerk值
        self.maxDelta = 0.5 # 最大前轮转角 rad
        self.n_controls = 2 # 控制量个数 [a, delta_rate]
        self.n_states = 6 # 状态量个数 [s, l, theta, v, a, delta]
        self.mu = 0.85
        self.g0 = 9.8
        # self.q = q  # Weight matrix
        # self.r = r  #weight matrix
        self.w =[]
        self.w0 = []
        self.lbw = []
        self.ubw = []
        self.J = []
        self.g = []
        self.lbg = []
        self.ubg = []
        
        s = SX.sym('s')
        l = SX.sym('l')
        theta = SX.sym('theta')
        length = SX.sym('length')
        width = SX.sym("width")
        # 双圆的前圆心
        s_f = s + 1/4*length*cos(theta)
        l_f = l + 1/4*length*sin(theta)
        # 双圆的后圆心
        s_r = s - 1/4*length*cos(theta)
        l_r = l - 1/4*length*sin(theta)
        # 圆的半径
        r_circle = 0.5*sqrt((0.5*length)**2+width**2)
        self.generate_circle = Function('generate_circle', [s, l, theta, length, width], [s_f,l_f,s_r,l_r,r_circle], ['s', 'l', 'theta', 'length', 'width'], ['s_f','l_f','s_r','l_r','r_circle'])
        obs_s = SX.sym('obs_s')
        obs_l = SX.sym('obs_l')
        distance = (s-obs_s)**2+(l-obs_l)**2
        self.collision_function = Function('collision_function', [s, l, obs_s, obs_l], [distance], ['ego_s', 'ego_l', 'obs_s', 'obs_l'], ['distance'])

    def build_model(self,coeffi,L_car,q,r,maxA,minA,maxDeltaRate,) -> bool:
        """建立系统模型

        Args:
            coeffi (_type_): 曲率的拟合系数
            L_car (_type_): ego的车长
            q (_type_): 状态的惩罚矩阵
            r (_type_): 控制量的惩罚矩阵
            maxA (_type_): 最大加速度
            minA (_type_): 最小加速度
            maxDeltaRate (_type_): 最大前轮转角速率

        Returns:
            bool: _description_
        """
        self.maxDeltaRate = maxDeltaRate # 最大前轮转角速率rad/s
        self.maxA = maxA # 最大加速度
        self.minA = minA # 最小减速度
        dt =self.dt
        Np = self.Np
        n_controls = self.n_controls
        n_states = self.n_states

        # Declare model variables
        s = SX.sym('s')
        l = SX.sym('l')
        theta = SX.sym('theta')
        v = SX.sym('v')
        a = SX.sym("a")
        delta = SX.sym('delta')
        
        s_ref = SX.sym('s_ref')
        l_ref = SX.sym('l_ref')
        theta_ref = SX.sym('theta_ref')
        v_ref = SX.sym('v_ref')
        a_ref = SX.sym('a_ref')
        delta_ref = SX.sym('delta_ref')
        X_ref =vertcat(s_ref,l_ref,theta_ref,v_ref,a_ref,delta_ref)
        
        #control variable
        jerk = SX.sym('jerk')
        delta_rate = SX.sym('delta_rate')
        X_system = vertcat(s, l,theta,v,a, delta)
        U_system = vertcat(jerk, delta_rate)

        kr_s_fit =np.polyval(coeffi,s)
        kr_s_fit_function = Function('kr_s_fit_function', [s], [kr_s_fit])
        
        # 满足车辆动力学模型
        s_dot     = v*cos(theta)/(1-kr_s_fit_function(s)*l)
        l_dot     = v*sin(theta)
        theta_dot = tan(delta)*v/L_car -kr_s_fit_function(s)*(v*cos(theta)/(1-kr_s_fit_function(s)*l))
        v_dot     = a
        a_dot     = jerk
        delta_dot = delta_rate
        X_dot = vertcat(s_dot,l_dot,theta_dot,v_dot,a_dot,delta_dot)
        
        yaw_rate = tan(delta)*v/L_car
        self.yawrate = Function('yawrate', [X_system, U_system], [yaw_rate])

        # Objective term
        # q = self.q # 状态量权重
        # r = self.r # 控制量权重
        L = U_system.T@r@U_system +(X_system-X_ref).T@q@(X_system-X_ref)

        # Formulate discrete time dynamics
        if False:
            # CVODES from the SUNDIALS suite
            dae = {'x':X_system, 'p':U_system, 'ode':X_dot, 'quad':L}
            F = integrator('F', 'cvodes', dae, 0, dt)
        else:
            # Fixed step Runge-Kutta 4 integrator
            M = 2 # RK4 steps per interval
            DT = dt/M
            f = Function('f', [X_system, U_system,X_ref], [X_dot, L])
            X0 = SX.sym('X0', n_states)
            U_dec = SX.sym('U', n_controls)
            X = X0
            Q = 0
            for j in range(M):
                k1, k1_q = f(X, U_dec,X_ref)
                k2, k2_q = f(X + DT/2 * k1, U_dec,X_ref)
                k3, k3_q = f(X + DT/2 * k2, U_dec,X_ref)
                k4, k4_q = f(X + DT * k3, U_dec,X_ref)
                X = X + DT/6*(k1 +2*k2 +2*k3 +k4)
                Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
            self.F = Function('F', [X0, U_dec,X_ref], [X, Q],['x0','uin','x_ref'],['xf','qf'])
        return True
    
    def generate_constraint(self, state_ini, state_final_des, state_cons, obs_s_l, w_opt, d_safe, car_following_list, follow_distance, maxDelta):
        """建立优化问题的约束

        Args:
            state_ini (_type_): 初始状态
            state_final_des (_type_): 期望终端状态
            state_cons: 关键状态约束
            obs_s_l (_type_): 筛选的障碍物SL列表
            w_opt (_type_): 上一次ocp的解作为初值
            d_safe: 双圆之间的安全裕度
            car_following_list: 前后车SL长宽等信息的list
        """
        
        #obs collision constraints
        
        F = self.F
        Np = self.Np
        maxA = self.maxA
        minA = self.minA
        maxjerk = self.maxjerk
        maxDeltaRate = self.maxDeltaRate
        generate_circle = self.generate_circle
        collision_function = self.collision_function
        
        n_controls = self.n_controls
        n_states = self.n_states
        n_obs    = len(obs_s_l) # suppose NP*num_obs
        maxDelta = maxDelta

        # Start with an empty NLP
        w=[] # optimization variable
        w0 = w_opt # initial guess (last optimal solution)
        lbw = [] # lower bound
        ubw = [] # upper bound
        J = 0 # obj
        g =[] # constraint
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = SX.sym('X_0', n_states)
        
        w += [Xk]
        lbw += [state_ini[0], state_ini[1], state_ini[2], state_ini[3], state_ini[4], state_ini[5]]
        ubw += [state_ini[0], state_ini[1], state_ini[2], state_ini[3], state_ini[4], state_ini[5]]

        # Formulate the NLP
        for k in range(Np):
            # New NLP variable for the control
            Uk = SX.sym('U_' + str(k), n_controls)
            w   += [Uk]
            lbw += [-maxjerk, -maxDeltaRate]
            ubw += [maxjerk, maxDeltaRate]

            # 每一步的参考状态
            xref =[state_ini[0]+(state_final_des[0]-state_ini[0])/Np*(k+1),state_ini[1]+(state_final_des[1]-state_ini[1])/Np*(k+1),0,state_ini[3]+(state_final_des[3]-state_ini[3])/Np*(k+1),0,0]
            
            # Integrate till the end of the interval
            Fk = F(x0=Xk, uin=Uk,x_ref=xref)
            Xk_end = Fk['xf']
            J= J + Fk['qf']

            # New NLP variable for state at end of interval
            Xk = SX.sym('X_' + str(k+1), n_states)
            w   += [Xk]
            lbw += [state_ini[0], state_cons.bound_l_right, -state_cons.bound_theta, 0, minA, -maxDelta]
            ubw += [state_cons.bound_s, state_cons.bound_l_left, state_cons.bound_theta, state_cons.bound_v, maxA, maxDelta]
            
            # Add yawrate limit
            yawrate = self.yawrate(Xk, Uk)
            g += [yawrate*Xk[3]]
            lbg += [-self.mu*self.g0]
            ubg += [self.mu*self.g0] 
            
            # Add collision constraints 双圆碰撞约束
            for j in range(n_obs):
                ego_circle = generate_circle(s=Xk[0],l=Xk[1],theta=Xk[2],length=state_ini[6],width=state_ini[7])
                obs_circle = generate_circle(s=obs_s_l[j][k][0],l=obs_s_l[j][k][1],theta=obs_s_l[j][k][2],length=obs_s_l[j][k][3],width=obs_s_l[j][k][4])
                
                # 两两圆心之间距离
                distance1 = collision_function(ego_s=ego_circle['s_f'],ego_l=ego_circle['l_f'],obs_s=obs_circle['s_f'],obs_l=obs_circle['l_f']) # 前前
                distance2 = collision_function(ego_s=ego_circle['s_f'],ego_l=ego_circle['l_f'],obs_s=obs_circle['s_r'],obs_l=obs_circle['l_r']) # 前后
                distance3 = collision_function(ego_s=ego_circle['s_r'],ego_l=ego_circle['l_r'],obs_s=obs_circle['s_f'],obs_l=obs_circle['l_f']) # 后前
                distance4 = collision_function(ego_s=ego_circle['s_r'],ego_l=ego_circle['l_r'],obs_s=obs_circle['s_r'],obs_l=obs_circle['l_r']) # 后后
                
                dis_min = ego_circle['r_circle']+obs_circle['r_circle']
                dis_min = float(dis_min)+d_safe # 0.8m安全裕度
                g += [distance1['distance'],distance2['distance'],distance3['distance'],distance4['distance']]
                lbg += [dis_min**2,dis_min**2,dis_min**2,dis_min**2] 
                ubg += [np.inf,np.inf,np.inf,np.inf] 

            # Add collision constraints 跟车约束
            for j in range(len(car_following_list)):
                g += [(car_following_list[j][k][0]-Xk[0])**2]
                lbg += [((car_following_list[j][k][3]+state_ini[6])/2+follow_distance)**2]
                ubg += [np.inf]
            
            # Add equality constraint
            g   += [Xk_end-Xk] # continuity
            lbg += [0, 0, 0, 0, 0, 0]
            ubg += [0, 0, 0, 0, 0, 0]
            

        #"hold" terminal conditions , 暂时不定终端状态
        g   += [Xk_end]
        lbg += [state_ini[0], state_cons.bound_l_right, -state_cons.bound_theta, 0, minA, -maxDelta]
        ubg += [state_cons.bound_s, state_cons.bound_l_left, state_cons.bound_theta, state_cons.bound_v, maxA, maxDelta]
        
        self.w = w 
        self.w0 = w0
        self.lbw = lbw
        self.ubw = ubw
        self.J = J
        self.g = g
        self.lbg = lbg
        self.ubg = ubg
        
    def solve(self):
        # Create an NLP solver
        prob = {'f': self.J, 'x': vertcat(*self.w), 'g': vertcat(*self.g)}
        opts_setting = {
            'ipopt.max_iter': 4000,  # 最大迭代次数
            'ipopt.print_level': 0,  #  用于优化计算的日志输出详细级别，数字越高，越详细，0即为不输出相关信息
            'print_time': 0, # 不输出最后的求解时间
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6,
            'record_time': 1
        }
        solver = nlpsol('solver', 'ipopt', prob, opts_setting)
        # Solve the NLP
        sol = solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        sovle_states = solver.stats()
        self.time_cost = sovle_states['t_wall_total']
        w_opt = sol['x'].full().flatten()
        # print("OCP States:",sovle_states['return_status'],"Takes "+str(self.time_cost)+"s")

        S_opt = w_opt[0::(self.n_controls+self.n_states)]
        L_opt = w_opt[1::(self.n_controls+self.n_states)]
        Theta_opt = w_opt[2::(self.n_controls+self.n_states)]
        V_opt = w_opt[3::(self.n_controls+self.n_states)]
        A_opt = w_opt[4::(self.n_controls+self.n_states)]
        delta_opt = w_opt[5::(self.n_controls+self.n_states)]

        return S_opt,L_opt,Theta_opt,V_opt,A_opt,delta_opt,w_opt,sovle_states['return_status']
            
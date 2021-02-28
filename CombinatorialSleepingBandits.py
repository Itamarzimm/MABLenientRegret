from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt

###### Define the MAB settings :
#T = 1000000#10000000#50000000
k = 50
eps = 0.2
max_arm = 1-eps
###### Define the arms :
A = [c for c in range(k)]
A_EXP= [np.random.uniform(0,max_arm) for d in range(k)] ## Uniform Disdtibution
A_lower =  [0 for d in range(k)] #[A_EXP[d] for d in range(k)]
A_upper = [1 for d in range(k)] #[(1-((1-A_EXP[d]))) for d in range(k)]
A_DIST_range =[(min(A_EXP[d]-A_lower[d],A_upper[d]-A_EXP[d])) for d in range(k)]
A_star = max(A,key=lambda i :A_EXP[i])
A_star_val = max(A_EXP)
A_diff_from_best = (-(np.array(A_EXP)) + A_star).sum()

graph_type = 2
g_2_bound = []
g_2_LenientRegret_array = []
g_2_E_regret_array=[]
g_bound_k_2 =[]
if graph_type == 1 :
    T_Set = [100000]
else :
    T_Set = [(5000 * s) for s in range(1,100)]
for T in T_Set:
    #### UCB Alogrithem() :
    ## Initialization :
    UCB = np.zeros(k,dtype=np.float32)
    AVG = np.zeros(k,dtype=np.float32)
    couns = np.zeros(k,dtype=np.int32)
    Lenient_regret =  0.0
    E_regret=  0.0
    Lenient_regret_array = []
    E_regret_array=[]
    x_axis = []
    bound_k_2 = []
    bound_k_1 = []

    phase_count = 0
    ## Step
    for i in range(T):
        # Graph maintenance
        if i % 100 == 0 and i >0 :
            Lenient_regret_array.append(Lenient_regret)
            E_regret_array.append(E_regret)
            x_axis.append(i)
            bound_k_2.append(((k**2/2)*np.log(i))/(eps**2))
            bound_k_1.append(((k * np.log(i))) / (eps**2))
        # Define the sleeping actions
        num_of_actions_i = np.random.randint(2, k - 1)
        available_arms = np.random.choice(np.array(A), num_of_actions_i, replace=False) ## This is the set of available actions .

        ## Find the maximum in terms of UCB:
        max_UCB_idx = max(available_arms,key = lambda x:UCB[x])
        best_arm_value = np.random.uniform(A_EXP[max_UCB_idx]-A_DIST_range[max_UCB_idx],A_EXP[max_UCB_idx]+A_DIST_range[max_UCB_idx])

        ## Update the values of UCB,cnt,avg for the best action :
        cnt = couns[max_UCB_idx]
        rad = np.sqrt(((8 * np.log(T)) / (cnt+1)))
        AVG[max_UCB_idx] = ((AVG[max_UCB_idx] * (cnt)) + (best_arm_value)) / (cnt + 1)
        UCB[max_UCB_idx] = AVG[max_UCB_idx] + rad
        couns[max_UCB_idx] += 1

        ## Find the maximum in terms of mean:
        a_star_idx = max(available_arms,key = lambda x:A_EXP[x])
        a_star_val = A_EXP[a_star_idx]
        ## Calculate the regret :
        E_regret += (a_star_val - A_EXP[max_UCB_idx])
        if ((A_EXP[max_UCB_idx] + eps) < a_star_val):
            Lenient_regret += (A_star_val - A_EXP[max_UCB_idx])

    g_2_bound.append((32 * k * np.log(T)) / (eps**2))
    g_bound_k_2.append((32 * ((k**2)/2) * np.log(T)) / (eps**2))
    g_2_LenientRegret_array.append(Lenient_regret)
    g_2_E_regret_array.append(E_regret)

if graph_type == 1 :
    print('Lenient_regret: ' , Lenient_regret)
    print('regret : ' , E_regret)

    # Graphs:
    plt.yscale('log')
    plt.plot(x_axis,Lenient_regret_array, marker='*', c='r')
    plt.plot(x_axis,bound_k_1, marker='.', c='b')
    plt.plot(x_axis,bound_k_2, marker='.', c='y')
    plt.plot(x_axis,E_regret_array, marker='.', c='g')

    plt.xlabel('Number of steps(T)')
    plt.ylabel(' Regret')
    #plt.axvline(x=((k*(np.log(T)*32))/(eps**2)))
    #plt.ylim([0,T])
    plt.grid(True)
    plt.legend(('Empirical lenient regret ','Lenient regret upper bound 1  ','Lenient regret upper bound 2  ','Empirical regular regret  '))
    #plt.legend(('Empirical lenient regert of the algorithm' , 'Lenient regret upper bound 1  ','Empirical regular regret  '))
    Name = 'Regret in reality vs regret bound for eact T'
    plt.show()

if graph_type == 2 :
    x_s = T_Set
    # Graphs:
    plt.yscale('log')
    plt.plot(x_s,g_2_LenientRegret_array, marker='*', c='r')
    plt.plot(x_s, g_2_E_regret_array, marker='.', c='g')
    plt.plot(x_s,g_2_bound, marker='.', c='b')
    plt.plot(x_s,g_bound_k_2,  c='y')
    plt.xlabel('Number of steps(T)')
    plt.ylabel(' Regret')
    #plt.ylim
    plt.grid(True)
    plt.legend(('Empirical lenient regret ','Empirical regret','Improved lenient regret upper bound','Naive lenient regret upper bound'))
    Name = 'Regret in reality vs regret bound for eact T'
    plt.show()
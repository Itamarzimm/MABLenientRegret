import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.size'] = 12
###### Define the MAB settings :

#T = 2000000#10000000#50000000
k = 20
eps = 0.2#
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




## Our alogrithem(S.E) :
graph_type = 1
g_2_bound = []
g_2_LenientRegret_array = []
g_2_E_regret_array=[]
if graph_type == 1 :
    T_Set = [1000000]
else :
    T_Set = [(5000 * s) for s in range(1,100)]
for T in T_Set:
    # Initialization :
    A_tag = A.copy() ## This is the set of actions that have not been eliminated.
    AVG = np.zeros(k,dtype=np.float32)
    LenientRegret =  0.0 ##
    E_regret=  0.0
    LenientRegret_array = []
    E_regret_array=[]
    x_axis = []
    bound = []
    # Step
    i=0
    phase_count = 0
    while (i<T):
        # Graph maintenance
        if i % 1000 == 0 and i >0 :
            LenientRegret_array.append(LenientRegret)
            E_regret_array.append(E_regret)
            x_axis.append(i)

        if len(A_tag) ==0:
            print("empty SET!")

        # Start a New Phase of Elimination #
        for j in A_tag:
            i = i+1
            # Randomly select the j arm
            curr_reg = np.random.uniform(A_EXP[j]-A_DIST_range[j],A_EXP[j]+A_DIST_range[j])
            # Calculate the regret in this step
            E_regret +=  (A_star_val - A_EXP[j])
            if ((A_EXP[j]+eps)<A_star_val) :
                LenientRegret += (A_star_val - A_EXP[j])
            # Calculate the estimator for the arm
            AVG[j] = ((AVG[j]*phase_count)+(curr_reg))/(phase_count+1)

        max_avg = 0.0
        phase_count = phase_count + 1
        rad = np.sqrt(((8 * np.log(T)) / phase_count))
        # Find the estimator with the maximum value in UCB terms.
        for j in A_tag:
            ## Get MAX
            if AVG[j]> max_avg :
                max_avg = AVG[j]
            ## Find  max UCB
            max_ucb = max_avg - rad
        # Elimination according to "max_ucb"
        removelist = []
        for j in A_tag:
            ## Eliminate ARMS
            if ((AVG[j]+rad)< max_ucb):
                removelist.append(j)
        for j in removelist:
            A_tag.remove(j)

    g_2_bound.append((32*k*np.log(T))/eps)
    g_2_LenientRegret_array.append(LenientRegret)
    g_2_E_regret_array.append(E_regret)

if graph_type == 1:
    print('num of arms after T steps: ', len(A_tag))
    print('identify the best arm ? ', A_tag[0] == A_star)
    print('LenientRegret ' , LenientRegret)
    print('Eregret : ' , E_regret)

    # Graphs:
    plt.yscale('log')
    plt.plot(x_axis,LenientRegret_array, marker='*', c='r')
    #plt.plot(x_axis,bound, marker='.', c='b')
    plt.plot(x_axis,E_regret_array, marker='.', c='g')

    plt.xlabel('Number of steps(T)')
    plt.ylabel(' Regret')
    plt.axhline(y=((32*k*np.log(T))/eps), c='b')
    plt.axvline(x=((k*(np.log(T)*32))/(eps**2)), c='y')
    #plt.ylim([0,T])
    plt.grid(True)
    plt.legend(('Empirical lenient regret ','Empirical regret  '))
    Name = 'Regret in reality vs regret bound for eact T'
    plt.show()

else :
    x_s = T_Set
    # Graphs:
    #plt.yscale('log')
    plt.plot(x_s,g_2_LenientRegret_array, marker='*', c='r')
    plt.plot(x_s,g_2_bound, marker='.', c='b')
    plt.plot(x_s,g_2_E_regret_array, marker='.', c='g')
    plt.xlabel('Number of steps(T)')
    plt.ylabel(' Regret')
    #plt.ylim([0,T])
    plt.grid(True)
    plt.legend(('Empirical lenient regret ','Lenient regret upper bound  ','Empirical regret  '))
    Name = 'Regret in reality vs regret bound for eact T'
    plt.show()
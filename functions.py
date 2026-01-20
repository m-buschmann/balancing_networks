import numpy as np 
import matplotlib.pyplot as plt 
from itertools import product
from brian2 import * 

J_EE = 5
J_IE = -2.5
J_EI = 2
J_II = -0.45

def generate_connectivity_matrix(N_E=8000, N_I=2000, c=0.2, p=0):

    cm_EE = np.zeros((N_E, N_E))
    cm_EI = np.zeros((N_I, N_E))
    cm_IE = np.zeros((N_E, N_I))
    cm_II = np.zeros((N_I, N_I))

    # Exc2Exc
    for target in range(N_E):
        source = np.random.choice(int(N_E), size=int(c*N_E), replace=False)             
        cm_EE[source, target] = J_EE

    # Exc2Inh
    for target in range(N_I):
        source = np.random.choice(np.arange(0, N_E, 1), size=int(c*N_E), replace=False)
        cm_IE[source, target] = J_IE

    # Inh2Exc
    for target in range(N_E):
        source = np.random.choice(np.arange(0, N_I, 1), size=int(c*N_I), replace=False)
        cm_EI[source, target] = J_IE

    # Inh2Inh
    for target in range(N_I):
        source = np.random.choice(np.arange(0, N_I, 1), size=int(c*N_I), replace=False)
        cm_II[source, target] = J_II

    # Memory Patterns
    for k in range(p):
        if k == 1:
            for target in range(int(k*28-8), int((k+1)*28-8)):
                source = np.random.choice(np.arange(int(k*28-8), int((k+1)*28-8), 1), size=int(28), replace=False)             
                cm_EE[source, target] = J_EE*5        
        else:
            for target in range(int(k*28), int((k+1)*28)):
                source = np.random.choice(np.arange(int(k*28), int((k+1)*28), 1), size=int(28), replace=False)             
                cm_EE[source, target] = J_EE*5

    full_cm = np.block([
    [cm_EE, cm_IE],
    [cm_EI, cm_II]
    ])

    return {
            'cm_EE' : cm_EE,
            'cm_EI' : cm_EI,
            'cm_IE' : cm_IE,
            'cm_II' : cm_II,
            'full_cm' : full_cm
            }


# cm = generate_connectivity_matrix(N_E=8000, N_I=2000, p=3)
# full_cm = cm['full_cm']

# fig, ax = plt.subplots()

# im = ax.imshow( 
#                 full_cm, 
#                 cmap='viridis', 
#                 vmin=full_cm.min(),
#                 vmax=full_cm.max()
#             )
# fig.colorbar(im, ax=ax)

# ax.set_ylabel('Source Neuron Idx.')
# ax.set_xlabel('Target Neuron Idx.')

# plt.show()

def get_isi_cv(sp):
    N = len(sp)
    isi_cvs = np.empty(N)
    for i in range(len((sp))):
        isis = [sp[i][k+1] / ms - sp[i][k] / ms for k in range(len(sp[i]) - 1)]
        isi_cv = np.std(isis) / np.mean(isis)
        isi_cvs[i] = isi_cv

    return isi_cvs


def get_spiking_correlation(spiketrains, N1, N2):

    tau1 = 50 * ms
    tau2 = 4 * tau1
    dt = 0.1 * ms

    t_kernel = np.arange(-5*tau2, 5*tau2 + dt, dt)
    K_vals = (1/tau1 * np.exp(-np.abs(t_kernel)/tau1)
            - 1/tau2 * np.exp(-np.abs(t_kernel)/tau2))

    K_vals = np.asarray(K_vals)

    F = {}
    for i, train in spiketrains.items():
        F[i] = np.convolve(train, K_vals, mode="same")

    V = np.empty((N1, N2))
    for i, j in product(np.arange(0, N1, 1), np.arange(0, N2, 1)):
        V[i, j] = np.sum(F[i] * F[j])

    X = np.empty((N1, N2))
    for i, j in product(np.arange(0, N1, 1), np.arange(0, N2, 1)):
        X[i, j] = V[i, j] / np.sqrt(V[i, i] * V[i, j])

    return X
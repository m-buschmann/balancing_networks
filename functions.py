import numpy as np 
import matplotlib.pyplot as plt 
from itertools import product
from brian2 import * 


#### Simulation ####

def run_simulation(simtime, task, p_rc=0.05, simtime_2=50*10**3 * ms):


    # Cell
    tau_M = 20 * ms
    V_rest = -60 * mV
    theta = -50 * mV
    tau_ref = 5 * ms
    g_L = 10 * nS

    # Network
    N_E = 8000
    N_I = 2000

    # Synapses
    tau_E = 5 * ms
    tau_I = 10 * ms
    g_bar = 3 * nS
    V_E = 0 * mV
    V_I = -80 * mV
    gamma = 0*ms

    # STDP
    alpha = 0.12
    p0 = 5 * Hz
    eta = 10**-3
    tau_STDP = 20 * ms

    # Stimulus
    I_b = 200*pA

    # Neuron Model
    eqs = '''

    dV/dt = ((V_rest - V) + (I_b + g_E*(V_E - V) + g_I*(V_I - V)) * 1/g_L) / tau_M : volt (unless refractory)

    dg_E/dt = -g_E / tau_E : siemens
    dg_I/dt = -g_I / tau_I : siemens

    I_exc =  g_E*(V_E - V) : ampere
    I_inh =  g_I*(V_I - V) : ampere

    dx/dt = -x / tau_STDP : 1

    '''

    # Neuron Groups
    G_E = NeuronGroup(
                    N_E,
                    model=eqs, 
                    threshold='V>=theta', 
                    reset='V=V_rest; x+=1', 
                    refractory=tau_ref,
                    method='euler'
                    )

    G_I = NeuronGroup(
                    N_I, 
                    model=eqs, 
                    threshold='V>=theta', 
                    reset='V=V_rest; x+=1', 
                    refractory=tau_ref,
                    method='euler'
                    )

    #Synapses
    if task == 'a':
        pre_logic = 'g_I_post += w * 10 * g_bar'
        post_logic = ''
    else:
        pre_logic = 'g_I_post += w * 10 * g_bar; w += eta * (x_post - alpha)'
        post_logic = 'w += eta * x_pre'

    Syn_EE = Synapses(G_E, G_E, 'w : 1', on_pre='g_E_post += w*g_bar', delay=gamma)
    Syn_EE.connect(p=0.02)
    Syn_EE.w = 1

    Syn_IE = Synapses(G_E, G_I, on_pre='g_E_post += g_bar', delay=gamma)
    Syn_IE.connect(p=0.02)

    Syn_EI = Synapses(G_I, G_E, 'w : 1', on_pre=pre_logic, on_post=post_logic)
    Syn_EI.connect(p=0.02)
    Syn_EI.w = 0.1

    Syn_II = Synapses(G_I, G_I, on_pre='''g_I += 10*g_bar''', delay=gamma)
    Syn_II.connect(p=0.02)

    Poisson_E = PoissonInput(G_E, 'g_E', N=50, rate=2*Hz, weight=0.05*g_bar)
    Poisson_I = PoissonInput(G_I, 'g_E', N=50, rate=2*Hz, weight=0.05*g_bar)

    #Monitors
    SpikeMonE = SpikeMonitor(G_E[:800], record=True)
    SpikeMonI = SpikeMonitor(G_I[:200], record=True)
    CurrentMon = StateMonitor(G_E, ('I_inh', 'I_exc'), record=np.arange(1000, 1010, 1))
    StateMonSyn_EI = StateMonitor(Syn_EI, ('w'), record=Syn_EI[:10])
    RateMon = PopulationRateMonitor(G_E)
    
    monitors = [SpikeMonE, SpikeMonI, StateMonSyn_EI, RateMon, CurrentMon]

    if task == 'c':
        assembly = np.arange(0, 500)
        assembly_mon = StateMonitor(G_E, ('I_inh', 'I_exc'), record=range(10))
        monitors.append(assembly_mon)

    all_synapses = [Syn_EE, Syn_EI, Syn_II, Syn_IE]

    #Run
    net = Network()
    net.add(G_E, G_I, monitors, all_synapses, Poisson_I, Poisson_E)

    G_E.V = 'V_rest + rand() * (theta - V_rest)'
    G_I.V = 'V_rest + rand() * (theta - V_rest)'

    net.run(simtime)

    if task == 'c':
        assembly_start = 0
        assembly_end = 500
        assembly_indices = np.arange(assembly_start, assembly_end)

        w_assembly = 1.0  

        num_connections_modified = 0

        for i in assembly_indices:
            for j in assembly_indices:
                if i != j and np.random.rand() < p_rc:
                    existing = (Syn_EE.i == i) & (Syn_EE.j == j)
                    if np.any(existing):
                        idx = np.where(existing)[0][0]
                        Syn_EE.w[idx] *= 2.0
                    else:
                        Syn_EE.connect(i=i, j=j)
                        Syn_EE.w[-1] = w_assembly
        
        print('new assembly')

        net.run(simtime_2)
    
    
    spike_trains_E = {
        int(i): np.asarray(times / ms)
        for i, times in SpikeMonE.spike_trains().items()
    }

    spike_trains_I = {
        int(i): np.asarray(times / ms)
        for i, times in SpikeMonI.spike_trains().items()
    }

    result = {
            'SpikeMonE'         : SpikeMonE,
            'SpikeMonI'         : SpikeMonI,
            'CurrentMon'        : CurrentMon,
            'RateMonE'          : RateMon,
            'SynMon'            : StateMonSyn_EI,
            'spike_trains_E'    : SpikeMonE.spike_trains(),
            'spike_trains_I'    : SpikeMonI.spike_trains()
            }

    if task == 'c':
        result['assembly_mon'] = assembly_mon
    
    return result


##### Network Analysis #####

def get_isi_cv(N, spike_trains, t_min, t_max):
    
    cv_isi = np.zeros(N)

    for i, times in spike_trains.items():
        times = times[(times > t_min / ms) & (times < t_max / ms)]
        if len(times) >= 3:
            isis = np.diff(times)
            cv_isi[i] = np.std(isis) / np.mean(isis)
        else:
            cv_isi[i] = 0

        cv_valid = cv_isi[cv_isi != 0]

    return cv_valid


def kernel(t, tau1=50, tau2=200):
    return 1/tau1 * np.exp(-np.abs(t)/tau1) - 1/tau2 * np.exp(-np.abs(t)/tau2)


def get_spike_correlations(spike_times, t_min, t_max, dt=1, n_pairs=3800, tau1=50, tau2=200):
    N = len(spike_times)

    time = np.arange(t_min, t_max, dt)
    T = len(time)

    t_kernel = np.arange(-10*tau2, 10*tau2, dt)
    K = kernel(t_kernel, tau1, tau2)

    F = np.zeros((N, T))

    for i in range(N):
        signal = np.zeros(T)

        spikes_in_window = spike_times[i][(spike_times[i] >= t_min) & (spike_times[i] < t_max)]
        indices = np.round((spikes_in_window - t_min) / dt).astype(int)
        indices = indices[(indices >= 0) & (indices < T)]

        signal[indices] = 1.0

        conv = np.convolve(signal, K, mode='same') 
        start = len(K)//2
        F[i] = conv[start:start + T]

    rng = np.random.default_rng()
    i_idx, j_idx = np.triu_indices(N, k=1)
    sel = rng.choice(len(i_idx), size=n_pairs, replace=False)
    pairs = np.column_stack((i_idx[sel], j_idx[sel]))

    X = np.zeros(n_pairs)

    for k, (i, j) in enumerate(pairs):
        Fi = F[i]
        Fj = F[j]

        Vij = np.sum(Fi * Fj)
        Vii = np.sum(Fi * Fi)
        Vjj = np.sum(Fj * Fj)

        if Vii > 0 and Vjj > 0:
            X[k] = Vij / np.sqrt(Vii * Vjj)
        else:
            X[k] = np.nan

    return X
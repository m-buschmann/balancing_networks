import numpy as np 
import matplotlib.pyplot as plt 
from itertools import product
from brian2 import * 


#### Simulation ####

def run_simulation(simtime, task, p_rc=0.75, simtime_2=100*10**3 * ms):


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

    Syn_EE = Synapses(G_E, G_E, on_pre='g_E_post += g_bar', delay=gamma)
    Syn_EE.connect(p=0.02)
    Syn_IE = Synapses(G_E, G_I, on_pre='g_E_post += g_bar', delay=gamma)
    Syn_IE.connect(p=0.02)

    Syn_EI = Synapses(G_I, G_E, 'w : 1', on_pre=pre_logic, on_post=post_logic)
    Syn_EI.connect(p=0.02)

    Syn_II = Synapses(G_I, G_I, on_pre='''g_I += 10*g_bar''', delay=gamma)
    Syn_II.connect(p=0.02)

    all_synapses = [Syn_EE, Syn_IE, Syn_EI, Syn_II]

    Poisson_E = PoissonInput(G_E, 'g_E', N=100, rate=5*Hz, weight=g_bar)
    Poisson_I = PoissonInput(G_I, 'g_E', N=100, rate=5*Hz, weight=g_bar)

    #Monitors
    SpikeMonE = SpikeMonitor(G_E[:800], record=True)
    SpikeMonI = SpikeMonitor(G_I[:200], record=True)
    CurrentMon = StateMonitor(G_E, ('I_inh', 'I_exc'), record=range(10))
    StateMonSyn_EI = StateMonitor(Syn_EI, ('w'), record=Syn_EI[:10])
    RateMon = PopulationRateMonitor(G_E)
    monitors = [SpikeMonE, SpikeMonI, StateMonSyn_EI, RateMon, CurrentMon]

    #Run
    net = Network()
    net.add(G_E, G_I, monitors, all_synapses, Poisson_I, Poisson_E)

    G_E.V = 'V_rest + rand() * (theta - V_rest)'
    G_I.V = 'V_rest + rand() * (theta - V_rest)'

    net.run(simtime)

    if task == 'c':
        neuron_idcs = np.random.choice(N_E, 500, replace=False)

        Syn_G_E_sub = Synapses(
            G_E, G_E,
            on_pre='g_E_post += g_bar',
            delay=gamma
        )

        # Create all-to-all connections inside the subset with probability p_rc
        i, j = np.meshgrid(neuron_idcs, neuron_idcs)
        mask = np.random.rand(len(i.flatten())) < p_rc

        Syn_G_E_sub.connect(
            i=i.flatten()[mask],
            j=j.flatten()[mask]
        )

        net.add(Syn_G_E_sub)

        net.run(simtime_2)
    
    
    spike_trains_E = {
        int(i): np.asarray(times / ms)
        for i, times in SpikeMonE.spike_trains().items()
    }

    spike_trains_I = {
        int(i): np.asarray(times / ms)
        for i, times in SpikeMonI.spike_trains().items()
    }


    return {
            'SpikeMonE'     : SpikeMonE,
            'SpikeMonI'     : SpikeMonI,
            'CurrentMon'    : CurrentMon,
            'RateMonE'      : RateMon,
            'SynMon'        : StateMonSyn_EI,
            'spike_trains_E': SpikeMonE.spike_trains(),
            'spike_trains_I': SpikeMonI.spike_trains(),
            }


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
import os
import numpy as np
import xarray as xr
import collections
from sklearn.utils import check_random_state
from hmmlearn import hmm, vhmm
import scipy.signal
from tqdm.autonotebook import tqdm

def lowpass_butterfly(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y

def prep_data(pathname, filename, bandwidth):
    raw_data = xr.load_dataarray(os.path.join(pathname, filename))
    sampling_rate = raw_data.sampling_rate
    if bandwidth > sampling_rate/2:
        print(f"error: required bandwidth {bandwidth} is higher than what's allowed by the original sampling rate {sampling_rate}.")

# Train a suite of models, and keep track of the best model for each
# number of states, and algorithm
def try_fit(data, num_inits=5, num_states=range(1, 10), verbose=False):
    best_scores = collections.defaultdict(dict)
    best_models = collections.defaultdict(dict)
    for n in tqdm(num_states):
        for i in tqdm(range(num_inits)):
            vi = vhmm.VariationalGaussianHMM(n,
                                             n_iter=2000,
                                             covariance_type="full",
                                             implementation="scaling",
                                             tol=1e-6,
                                             random_state=rs,
                                             verbose=verbose)
            vi.fit(sequences, lengths)
            lb = vi.monitor_.history[-1]
            #print(f"Training VI({n}) Variational Lower Bound={lb} "
            #      f"Iterations={len(vi.monitor_.history)} ")
            if best_models["VI"].get(n) is None or best_scores["VI"][n] < lb:
                best_models["VI"][n] = vi
                best_scores["VI"][n] = lb
            em = hmm.GaussianHMM(n,
                                 n_iter=1000,
                                 covariance_type="full",
                                 implementation="scaling",
                                 tol=1e-6,
                                 random_state=rs,
                                 verbose=verbose)
            em.fit(sequences, lengths)
            ll = em.monitor_.history[-1]
            #print(f"Training EM({n}) Final Log Likelihood={ll} "
            #      f"Iterations={len(vi.monitor_.history)} ")
            if best_models["EM"].get(n) is None or best_scores["EM"][n] < ll:
                best_models["EM"][n] = em
                best_scores["EM"][n] = ll
    
    # Display the model likelihood/variational lower bound for each N
    # and show the best learned model
    for algo, scores in best_scores.items():
        best = max(scores.values())
        best_n, best_score = max(scores.items(), key=lambda x: x[1])
        for n, score in scores.items():
            flag = "* <- Best Model" if score == best_score else ""
            print(f"{algo}({n}): {score:.4f}{flag}")
    
        print(f"Best Model {algo}")
        best_model = best_models[algo][best_n]
        print(best_model.transmat_)
        print(best_model.means_)
        print(best_model.covars_)
    return {"EM_models": best_models["EM"], "VI_models":best_models["VI"], "EM_scores": best_scores["EM"], "VI_scores":best_scores["VI"]}



def sort_matrix(original_m, original_t):
    def switch_element(m, t, from_idx, to_idx):
        old_t = t
        old_m = m
        for (i1, i2) in zip(from_idx, to_idx):
            new_t = np.zeros_like(t)
            new_m = np.zeros_like(m)
            for i in range(len(m)):
                new_i = i
                if i == i1:
                    new_m[i] = old_m[i2]
                    new_i = i2
                elif i == i2:
                    new_m[i] = old_m[i1]
                    new_i = i1
                else:
                    new_m[i] = old_m[i]
                for j in range(len(m)):
                    new_j = j
                    if j == i1:
                        new_j = i2
                    elif j == i2:
                        new_j = i1
                    new_t[i][j] = old_t[new_i][new_j]
            old_t = new_t
            old_m = new_m
        return new_m, new_t

    switch_from = []
    switch_to = []
    m=original_m.copy()

    for i in range(len(m)):
        min_idx = i
        for j in range(i+1, len(m)):
            if m[min_idx] > m[j]:
                min_idx = j
        if i != min_idx:
            m[i], m[min_idx] = m[min_idx], m[i]
            switch_from.append(i)
            switch_to.append(min_idx)

    new_m, new_t = switch_element(original_m, original_t, switch_from, switch_to)
    return new_m, new_t


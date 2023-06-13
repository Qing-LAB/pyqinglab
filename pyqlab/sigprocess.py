import numpy as np
from scipy.signal import butter, filtfilt
from sklearn import mixture
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from kneed import KneeLocator
import pandas as pd
import IPython.display as display

def filterdesign_lowpass_butterfly(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_butterfly(data, cutoff, fs, order=5):
    b, a = filterdesign_lowpass_butterfly(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def autocorr(data, nodc = False):
    if nodc:
        data_nodc = data - np.average(data)
        result = np.correlate(data_nodc, data_nodc, mode='full')
    else:
        result = np.correlate(data, data, mode='full')
    return result[result.size//2-1:]

def get_BIC_score(model, X):
    """
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Returns
    -------
    bic : float
    """
    type = model.covariance_type
    score = model.score(X)
    shape = X.shape[0]
    shape_log = np.log(shape)
    (_, features) = model.means_.shape
    components = len(np.unique(model.predict(X)))

    base_score = -2 * score * shape
    type_score = features * (2 * components) - 1 

    if type == 'diag':
        type_score += components * features
    if type == 'full':
        type_score += components * features * ((features + 1) / 2)
    if type == 'spherical':
        type_score += components
    if type == 'tied':
        type_score += features * ((features + 1) / 2)
    return base_score + type_score * shape_log

def get_AIC_score(model, X):
    """
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    Returns
    -------
    aic : float
    """
    type = model.covariance_type
    score = model.score(X)
    shape = X.shape[0]
    (_, features) = model.means_.shape
    components = len(np.unique(model.predict(X)))

    base_score = -2 * score * shape
    type_score = features * (2 * components) - 1 

    if type == 'diag':
        type_score += components * features
    if type == 'full':
        type_score += components * features * ((features + 1) / 2)
    if type == 'spherical':
        type_score += components
    if type == 'tied':
        type_score += features * ((features + 1) / 2)

    return base_score + type_score * 2
    

def fit_GMM(data, startx, length, method = 'gaussian', max_components=4, max_iter=100, init="k-means++", verbose=0, knee_sensitivity=1, model_eval='BIC', tol=1e-3):
    section = data[startx:startx+length].reshape(-1, 1)
    N = np.arange(1, max_components+1)
    models = [None for i in range(len(N))]
    
    if verbose > 1:
        fig=plt.figure()
    
    match method:
        case 'gaussian':
            if model_eval == 'AIC' or model_eval == 'BIC':
                for i in tqdm(range(len(N)), 
                              desc="finding optimal# of components (using GaussianMixture):", 
                              disable=(verbose < 0)):
                    models[i] = mixture.GaussianMixture(
                        n_components=N[i], 
                        covariance_type='full', 
                        max_iter=max_iter, 
                        verbose=0 if verbose<0 else verbose, 
                        tol=tol,
                        init_params=init).fit(section)
                    if verbose > 1:
                        fig.clf()
                        plot_GMMmodel(section, models[i], show_AIC_BIC=False)
                        fig.canvas.draw()
                        display.display(plt.gcf())

                AIC = [m.aic(section) for m in models]
                BIC = [m.bic(section) for m in models]
            elif model_eval == 'FIXED':
                AIC = None
                BIC = None
                M_best = mixture.GaussianMixture(
                        n_components = max_components, 
                        covariance_type = 'full', 
                        max_iter = max_iter, 
                        verbose = 0 if verbose<0 else verbose, 
                        init_params = init).fit(section)
                models = [M_best]
                if verbose > 1:
                    fig.clf()
                    plot_GMMmodel(section, M_best, embed_plot=True, show_AIC_BIC=False)
                    fig.canvas.draw()
                    display.display(plt.gcf())
            
        case 'bayesian':
            if model_eval == 'AIC' or model_eval == 'BIC':
                for i in tqdm(range(len(N)), 
                              desc="finding optimal# of components (using BayesianGaussianMixture):",
                              disable=(verbose < 0)):
                    models[i] = mixture.BayesianGaussianMixture(
                        n_components=N[i], 
                        covariance_type='full', 
                        max_iter=max_iter, 
                        verbose=0 if verbose < 0 else verbose, 
                        tol=tol,
                        init_params = init).fit(section)
                    if verbose > 1:
                        fig.clf()
                        plot_GMMmodel(section, models[i], embed_plot=True, show_AIC_BIC=False)
                        fig.canvas.draw()
                        display.display(plt.gcf())

                AIC = [get_AIC_score(m, section) for m in models]
                BIC = [get_BIC_score(m, section) for m in models]
            elif model_eval == 'FIXED':
                AIC = None
                BIC = None
                M_best = mixture.BayesianGaussianMixture(
                        n_components=max_components, 
                        covariance_type='full', 
                        max_iter=max_iter, 
                        verbose=0 if verbose < 0 else verbose,
                        tol=tol,
                        init_params=init).fit(section)
                models = [M_best]
                if verbose > 1:
                    fig.clf()
                    plot_GMMmodel(section, M_best, embed_plot=True, show_AIC_BIC=False)
                    plt.show()
                    fig.canvas.draw()
                    display.display(plt.gcf())
        case _:
            raise Exception("unknown method set for fitting")
    
    knee_detection = False
    best_idx = 0
    
    match model_eval:
        case 'AIC':
            MDL_EVAL=AIC
            knee_detection = True
        case 'BIC':
            MDL_EVAL=BIC
            knee_detection = True
        case 'FIXED':
            None
        case _:
            raise Exception("unknown method evaluation method")
    
    if knee_detection:
        knee_point = KneeLocator(
            x=N, 
            y=MDL_EVAL, 
            S=knee_sensitivity, 
            curve='convex', 
            direction='decreasing', 
            online=False)

        if knee_point.knee:
            if verbose > 1:
                print(f"model evaluation curve knee point found at: {N[knee_point.knee]}")
            best_idx = knee_point.knee
        else:
            if verbose > 1:
                print(f"model evaluation curve no knee point found, choosing the minimum point {np.argmin(MDL_EVAL)}")
            best_idx = np.argmin(MDL_EVAL)
    
    models[best_idx].AIC = AIC
    models[best_idx].BIC = BIC
    
    return models[best_idx], models, best_idx

def plot_GMMmodel(data, M_best, bins=100, embed_plot=False, show_AIC_BIC=True):
    x = np.linspace(np.min(data), np.max(data), data.size)
    x_axis = data.copy()
    x_axis.sort()

    if not embed_plot:
        fig=plt.figure()
        fig.clf()
    
    if show_AIC_BIC:
        plt.subplot(2, 1, 1)
    plt.text(0, 0, f"n={len(M_best.means_.ravel())}")
    plt.hist(x_axis, 
             bins=bins, 
             histtype='bar', 
             density=True, 
             ec='red', 
             alpha=0.5)
    logprob = M_best.score_samples(x.reshape(-1, 1))
    responsibilities = M_best.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.plot(x, pdf, '-k')
    plt.plot(x, pdf_individual, '--k')
    
    if not embed_plot:
        plt.subplot(2, 1, 2)
    
    if show_AIC_BIC:
        plt.subplot(2, 1, 2)
        if M_best.AIC:
            N = np.arange(1, len(M_best.AIC)+1)
            plt.plot(N, M_best.AIC, '-k', label='AIC')
        if M_best.BIC:
            N = np.arange(1, len(M_best.BIC)+1)
            plt.plot(N, M_best.BIC, '--k', label='BIC')
    
    if not embed_plot:
        plt.show()
        fig.canvas.draw()

def label_data_with_model(data_for_predict, data_for_label, model, plot=True, cmap_name='jet'):
    N=len(model.means_)
    print(f"total # of components: {N}")
    
    cmap=mpl.colormaps[cmap_name]
    ctable = [cmap(i) for i in np.linspace(0, 1, N)]
    
    m = model.means_.ravel().copy() 
    sorted_m_idx = np.argsort(m)
    categorylabels_sorted = np.argsort(sorted_m_idx)
    
    category_unsorted = model.predict(data_for_predict)
    category = np.array([categorylabels_sorted[i] for i in category_unsorted])
    colors = [ctable[int(c)] for c in category]

    category_value = np.array([model.means_[i] for i in category_unsorted])
    
    legend_unsorted = model.predict(m.reshape(-1, 1))
    legend = np.array([categorylabels_sorted[i] for i in legend_unsorted])
    legend_color = [ctable[int(c)] for c in legend]
    
    data = {
        "data_for_predict": data_for_predict.ravel(),
        "data_for_label"  : data_for_label.ravel(),
        "data_label"      : category.ravel(),
        "label_value"     : category_value.ravel()
    }
    
    model_result = {
        "components"      : model.means_.ravel(),
        "weights"         : model.weights_.ravel(),
        "covariances"     : model.covariances_.ravel(),
        "label"           : categorylabels_sorted,
    }
    
    df = pd.DataFrame(data)
    md = pd.DataFrame(model_result)
    
    if plot:
        fig = plt.figure()
        plt.clf()
        w, h = plt.figaspect(np.ones((2, 8)))
        fig.set_figheight(h)
        fig.set_figwidth(w)
        
        dotsize = np.sqrt(model.covariances_.ravel())
        dotsize = dotsize / np.min(dotsize) * 20
        plt.subplot(1, 4, 1)
        plt.scatter(categorylabels_sorted, m, c=legend_color, s=dotsize)

        plt.subplot(1, 4, (2, 4))
        plt.plot(np.arange(0, data_for_predict.size), data_for_predict, '-k', alpha=0.5)
        plt.scatter(np.arange(0, data_for_label.size), data_for_label, c=colors, s=1)
        plt.show()
    
    return (df, md)

def get_event_lifetime(labelled_data: np.array, t: np.array, categories: np.array, index: int) -> np.array:
    category = categories[index]
    mask1 = labelled_data == category
    mask1 = np.insert(mask1, mask1.size, False)
    if mask1[0]:
        mask1 = np.insert(mask1, 0, False)
        mask2 = mask1[:-1] ^ mask1[1:]
    else:
        mask2 = mask1[:-1] ^ mask1[1:]
        mask2 = np.insert(mask2, 0, False)
    t1 = np.insert(t, t.size, t[-1]+t[-1]-t[-2])
    print(labelled_data)
    print(t1)
    print(mask1)
    print(mask2)
    intervals = np.diff(t1[mask2])
    if mask2[0]:
        return intervals[::2]
    else:
        return intervals[1::2]


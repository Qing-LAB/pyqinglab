import numpy as np
from scipy.signal import butter, filtfilt
import sklearn
from sklearn import mixture
import scipy.stats as stats
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from kneed import KneeLocator
import pandas as pd
import IPython.display as display
from typing import Union

GMModel = Union[type(mixture.GaussianMixture), type(mixture.BayesianGaussianMixture)]


def filterdesign_lowpass_butterfly(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_butterfly(data, cutoff, fs, order=5):
    b, a = filterdesign_lowpass_butterfly(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def autocorr(data, nodc=False):
    if nodc:
        data_nodc = data - np.average(data)
        result = np.correlate(data_nodc, data_nodc, mode="full")
    else:
        result = np.correlate(data, data, mode="full")
    return result[result.size // 2 - 1 :]


def get_BIC_score(model: GMModel, X):
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

    if type == "diag":
        type_score += components * features
    if type == "full":
        type_score += components * features * ((features + 1) / 2)
    if type == "spherical":
        type_score += components
    if type == "tied":
        type_score += features * ((features + 1) / 2)
    return base_score + type_score * shape_log


def get_AIC_score(model: GMModel, X):
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

    if type == "diag":
        type_score += components * features
    if type == "full":
        type_score += components * features * ((features + 1) / 2)
    if type == "spherical":
        type_score += components
    if type == "tied":
        type_score += features * ((features + 1) / 2)

    return base_score + type_score * 2


def fit_GMM(
    data: np.array,
    method="gaussian",
    max_components: int = 4,
    max_iter: int = 100,
    init: str = "k-means++",
    verbose: int = 0,
    knee_sensitivity: int = 3,
    model_eval: str = "BIC",
    tol: float = 1e-3,
):
    """
    INPUT:
    data: numpy array of raw sequence, dimension will be ignored by reshape(-1, 1)
    method: can be either 'gaussian' or 'bayesian' to use standard GaussianMixture, or use BayesianGaussianMixture
    max_components: the maximum component that this function will try, depends on the model_eval, 
    this can be the only thing evaluated, or the function will evaluate from N=1 up to N=max_components, and find the best
    one fit using the standard given by model_eval
    init: initial values, default is "k-means++"
    verbose: level of output. set to zero will only show progress bar. Set to negative will give zero output (the GaussianMixture and BayesianGaussianMixture may give some output). 
    Positive number will be used to set the verbose output for the mixture model called. Set verbose >= 10 will plot figures of the histogram and how the models fit
    knee_sensitivity: sensitivity of detecting the knee point of the evaluation method for the models.
    model_eval: method to evaluate which model of different #components fit the data best. Choice can be 'AIC', 'BIC' and 'single'. 
    If it is set to single, the function will only try to fit once with components set to max_components
    tol: threshold for deciding if model fitting has converged

    RETURN:
    a dictionary containing: 'best_model', 'all_evaluated_models', and 'best_model_index'

    """
    sequence = data.reshape(-1, 1)
    N = np.arange(1, max_components + 1)
    models = [None for i in range(len(N))]

    AIC=[]
    BIC=[]
    match method:
        case "gaussian":
            if model_eval == "AIC" or model_eval == "BIC":
                for i in tqdm(
                    range(len(N)),
                    desc="finding optimal# of components (using GaussianMixture):",
                    disable=(verbose < 0),
                ):
                    models[i] = mixture.GaussianMixture(
                        n_components=N[i],
                        covariance_type="full",
                        max_iter=max_iter,
                        verbose=0 if verbose < 0 else verbose,
                        tol=tol,
                        init_params=init,
                    ).fit(sequence)
                    if verbose >= 10:
                        fig = plt.figure()
                        plot_GMMmodel(sequence, models[i], fig, show_AIC_BIC=False)
                        plt.close()
                        display.display(fig)

                AIC = [m.aic(sequence) for m in models]
                BIC = [m.bic(sequence) for m in models]
            else:
                M_best = mixture.GaussianMixture(
                    n_components=max_components,
                    covariance_type="full",
                    max_iter=max_iter,
                    verbose=0 if verbose < 0 else verbose,
                    init_params=init,
                ).fit(sequence)
                models = [M_best]
                if verbose >= 10:
                    fig = plt.figure()
                    plot_GMMmodel(sequence, M_best, fig, show_AIC_BIC=False)
                    plt.close()
                    display.display(fig)

        case "bayesian":
            if model_eval == "AIC" or model_eval == "BIC":
                for i in tqdm(
                    range(len(N)),
                    desc="finding optimal# of components (using BayesianGaussianMixture):",
                    disable=(verbose < 0),
                ):
                    models[i] = mixture.BayesianGaussianMixture(
                        n_components=N[i],
                        covariance_type="full",
                        max_iter=max_iter,
                        verbose=0 if verbose < 0 else verbose,
                        tol=tol,
                        init_params=init,
                    ).fit(sequence)
                    if verbose >= 10:
                        fig = plt.figure()
                        plot_GMMmodel(
                            sequence, models[i], fig, show_AIC_BIC=False
                        )
                        plt.close()
                        display.display(fig)

                AIC = [get_AIC_score(m, sequence) for m in models]
                BIC = [get_BIC_score(m, sequence) for m in models]
            else:
                M_best = mixture.BayesianGaussianMixture(
                    n_components=max_components,
                    covariance_type="full",
                    max_iter=max_iter,
                    verbose=0 if verbose < 0 else verbose,
                    tol=tol,
                    init_params=init,
                ).fit(sequence)
                models = [M_best]
                if verbose >= 10:
                    fig = plt.figure()
                    plot_GMMmodel(sequence, M_best, fig, show_AIC_BIC=False)
                    plt.close()
                    display.display(fig)
        case _:
            raise Exception("Unknown method set for modeling. Please use 'gaussian' or 'bayesian'.")

    knee_detection = False
    best_idx = 0

    match model_eval:
        case "AIC":
            MDL_EVAL = AIC
            knee_detection = True
        case "BIC":
            MDL_EVAL = BIC
            knee_detection = True
        case "single":
            MDL_EVAL = [0]
        case _:
            raise Exception("Unknown method evaluation method, please use 'AIC', 'BIC' or 'single'.")

    min_detection = False
    
    if knee_detection:
        if knee_sensitivity*2 < max_components:
            knee_point = KneeLocator(
                x=N,
                y=MDL_EVAL,
                S=knee_sensitivity,
                curve="convex",
                direction="decreasing",
                online=False,
            )
            if knee_point.knee:
                if verbose > 1:
                    print(
                        f"Model evaluation curve knee point found at: {N[knee_point.knee]}"
                    )
                best_idx = knee_point.knee
            else:
                min_detection = True
        else:
            min_detection = True

       
        if min_detection:
            if verbose > 1:
                print(
                    f"Using the minimum point {np.argmin(MDL_EVAL)} as best model"
                )
            best_idx = np.argmin(MDL_EVAL)

    models[best_idx].AIC = AIC
    models[best_idx].BIC = BIC

    return {
        "best_model": models[best_idx],
        "all_evaluated_models": models,
        "best_model_index": best_idx,
    }


def plot_GMMmodel(
    data: np.array, model: GMModel, fig: Figure=None, bins=100, show_AIC_BIC=True
):
    """
    plot the model together with the histogram of the data
    if show_AIC_BIC is set, will plot also the AIC and BIC values of all models evaluated in previous call to fit_GMM
    """
    x = np.linspace(np.min(data), np.max(data), data.size)
    x_axis = data.copy()
    x_axis.sort()
    active_show = False

    if fig is None:
        fig = plt.figure()
        active_show = True

    if show_AIC_BIC:
        plt.subplot(2, 1, 1)
    plt.text(0, 0, f"n={len(model.means_.ravel())}")
    plt.hist(x_axis, bins=bins, histtype="bar", density=True, ec="red", alpha=0.5)
    logprob = model.score_samples(x.reshape(-1, 1))
    responsibilities = model.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    plt.plot(x, pdf, "-k")
    plt.plot(x, pdf_individual, "--k")

    if show_AIC_BIC and hasattr(model, "AIC") and hasattr(model, "BIC"):
        plt.subplot(2, 1, 2)
        if model.AIC:
            N = np.arange(1, len(model.AIC) + 1)
            plt.plot(N, model.AIC, "-k", label="AIC")
        if model.BIC:
            N = np.arange(1, len(model.BIC) + 1)
            plt.plot(N, model.BIC, "--k", label="BIC")

    if active_show:
        plt.close()
        display.display(fig)


def label_data_with_model(
    data_for_predict: np.array,
    data_for_label: np.array,
    model: GMModel,
    plot: bool = False,
    cmap_name: str = "jet",
):
    N = len(model.means_)
    print(f"total # of components: {N}")

    cmap = mpl.colormaps[cmap_name]
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
        "data_for_label": data_for_label.ravel(),
        "data_label": category.ravel(),
        "label_value": category_value.ravel(),
    }

    model_result = {
        "components": model.means_.ravel(),
        "weights": model.weights_.ravel(),
        "covariances": model.covariances_.ravel(),
        "label": categorylabels_sorted,
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
        plt.plot(np.arange(0, data_for_predict.size), data_for_predict, "-k", alpha=0.5)
        plt.scatter(np.arange(0, data_for_label.size), data_for_label, c=colors, s=1)
        plt.close()
        display.display(fig)

    return (df, md)


def get_event_timing(labelled_events: np.array, label: int) -> np.array:
    """
    INPUTS:
    labelled_events: labelled event series, will be converted to dtype int before detection
    label: event label, int
    RETURN:
    a dictionary containing the index of the switching on/off of the target event (into/or out of),
    the interval between these switchings, and the on-time and off-time.
    Note that the beginning (index 0) and the end (index labelled_events.size) will always be in the switching
    on/off index list so that the "on" events at the beginning and end will be correctly calculated.

    Example:

    """
    mask1 = labelled_events.astype(int) == label
    mask1 = np.insert(mask1, 0, False)
    mask1 = np.insert(mask1, -1, False)
    mask2 = mask1[:-1] ^ mask1[1:]

    t = np.arange(0, mask2.size)

    switch_event = t[mask2]
    switch_interval = np.diff(switch_event)
    on_time = switch_interval[::2]
    off_time = switch_interval[1::2]

    return {
        "switch_event": switch_event,
        "switch_interval": switch_interval,
        "on_time": on_time,
        "off_time": off_time,
    }


def fit_GMM_bysection(data: np.array, window: int, step: int, max_components: int, model_ref: GMModel, tol:float = 1e-3, method: str='gaussian', model_eval: str='BIC', init: str='k-means++', max_iter:int = 500):
    d = data.reshape(-1, 1)
    steprange = np.arange(0, d.size//step)
    startpts = steprange*window
    endpts = startpts + window
    
    for b, e in tdqm(zip(startpts, endpts)):
        if e > d.size:
            e = d.size
        section = d[b:e]
        mdl = fit_GMM(
                data: np.array,
                method=method,
                max_components=max_components,
                max_iter=max_iter,
                init=init,
                verbose=-1,
                model_eval=model_eval,
                tol=tol)
        
    

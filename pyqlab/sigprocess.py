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
    knee_sensitivity: int = 2,
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

    AIC = []
    BIC = []
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
                        plot_GMModel(sequence, models[i], fig, show_AIC_BIC=False)
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
                    plot_GMModel(sequence, M_best, fig, show_AIC_BIC=False)
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
                        plot_GMModel(sequence, models[i], fig, show_AIC_BIC=False)
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
                    plot_GMModel(sequence, M_best, fig, show_AIC_BIC=False)
                    plt.close()
                    display.display(fig)
        case _:
            raise Exception(
                "Unknown method set for modeling. Please use 'gaussian' or 'bayesian'."
            )

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
            raise Exception(
                "Unknown method evaluation method, please use 'AIC', 'BIC' or 'single'."
            )

    min_detection = False

    if knee_detection:
        if knee_sensitivity * 2 < max_components:
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
                print(f"Using the minimum point {np.argmin(MDL_EVAL)} as best model")
            best_idx = np.argmin(MDL_EVAL)

    models[best_idx].AIC = AIC
    models[best_idx].BIC = BIC

    return {
        "best_model": models[best_idx],
        "all_evaluated_models": models,
        "best_model_index": best_idx,
    }


def plot_GMModel(
    data: np.array, model: GMModel, fig: Figure = None, bins=100, show_AIC_BIC=True
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
    marker_base_size: int = 20,
):
    N = len(model.means_)
    print(f"total # of components: {N}")

    cmap = mpl.colormaps[cmap_name]
    ctable = [cmap(i) for i in np.linspace(0, 1, N)]

    m = model.means_.ravel().copy()
    label_reordered_by_m_value = np.argsort(np.argsort(m))
    #print(f"mean values: {m}")
    #print(f"labels ordered by the m value: {label_reordered_by_m_value}")

    data_labelled_by_m = model.predict(data_for_predict)
    data_labelled = np.array([label_reordered_by_m_value[unsorted_label] for unsorted_label in data_labelled_by_m])
    colors_for_labelled_data = [ctable[int(c)] for c in data_labelled]

    data_labelled_value = np.array([model.means_[i] for i in data_labelled_by_m])
    
    legend_unsorted = np.arange(0, m.size)
    legend = np.array([label_reordered_by_m_value[unsorted_label] for unsorted_label in legend_unsorted])
    legend_color = [ctable[int(c)] for c in legend]

    data = {
        "data_for_predict": data_for_predict.ravel(),
        "data_for_label": data_for_label.ravel(),
        "data_label": data_labelled.ravel(),
        "label_value": data_labelled_value.ravel(),
    }

    model_result = {
        "components": model.means_.ravel(),
        "weights": model.weights_.ravel(),
        "covariances": model.covariances_.ravel(),
        "label": label_reordered_by_m_value,
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
        dotsize = dotsize / np.min(dotsize) * marker_base_size
        plt.subplot(1, 4, 1)
        plt.scatter(label_reordered_by_m_value, m, c=legend_color, s=dotsize)

        plt.subplot(1, 4, (2, 4))
        plt.plot(np.arange(0, data_for_predict.size), data_for_predict, "-k", alpha=0.5)
        plt.scatter(
            np.arange(0, data_for_label.size),
            data_for_label,
            c=colors_for_labelled_data,
            s=marker_base_size,
        )
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


def sectionfit_GMM(
    data: np.array,
    window: int,
    step: int,
    max_components: int = 3,
    tol: float = 1e-3,
    method: str = "gaussian",
    model_eval: str = "AIC",
    init: str = "k-means++",
    max_iter: int = 500,
    knee_sensitivity=2,
):
    d = data.reshape(-1, 1)

    steprange = np.arange(0, d.size // step)
    means_buf = np.zeros(steprange.size * max_components, dtype=float)
    sigma_buf = means_buf.copy()
    buf_count = 0
    startpts = (steprange * window).astype(int)
    idx = startpts < d.size
    startpts = startpts[idx]
    endpts = (startpts + window).astype(int)
    mdls = [None for i in startpts]
    fig = None
    pbar = tqdm(np.arange(0, startpts.size), desc="Section fit with GaussianMixture:")
    for i in pbar:
        b = startpts[i]
        e = endpts[i]
        if e > d.size:
            e = d.size
        section = d[b:e]
        mdls[i] = fit_GMM(
            section,
            method=method,
            max_components=max_components,
            max_iter=max_iter,
            init=init,
            verbose=-1,
            model_eval=model_eval,
            knee_sensitivity=knee_sensitivity,
            tol=tol,
        )
        means = mdls[i]["best_model"].means_.ravel()
        sigma = np.sqrt(mdls[i]["best_model"].covariances_.ravel())

        means_buf[buf_count : buf_count + means.size] = means[:]
        sigma_buf[buf_count : buf_count + means.size] = sigma[:]
        buf_count += means.size
        steprange[i] = means.size
        pbar.set_description(
            f"Section fit with GaussianMixture: last found [{means.size}] components:"
        )

    return {
        "means": means_buf[:buf_count],
        "sigma": sigma_buf[:buf_count],
        "number_of_components": steprange[:i],
        "models": mdls,
        "window_size": window,
        "step_size": step,
        "start_points": startpts,
        "end_points": endpts,
    }


def label_sectional_means(
    section_fit_output, max_components, max_iter, tol, model_eval
):
    print("Fitting all means by a Gaussian model")
    all_means = section_fit_output["means"]
    means_model = fit_GMM(
        all_means,
        max_components=max_components,
        max_iter=max_iter,
        tol=tol,
        verbose=-1,
        model_eval=model_eval,
    )

    return means_model


def section_label_data_with_global_model(data, output, global_model):
    pbar = tqdm(
        np.arange(0, output["start_points"].size),
        desc="Going through data again to label with the overall distribution of means",
    )
    d = data.reshape(-1, 1)
    labelled_data = np.zeros(d.size, dtype=int)
    labelled_data_value = np.zeros(d.size, dtype=float)

    for i in pbar:
        b = output["start_points"][i]
        e = output["end_points"][i]
        if e > d.size:
            e = d.size
        section = d[b:e]
        section_model = output["models"][i]["best_model"]
        means = section_model.means_.ravel()
        means_label = global_model.predict(means.reshape(-1, 1))
        data_label_orig = section_model.predict(section.reshape(-1, 1))
        data_label = [means_label[i] for i in data_label_orig]
        data_label_value = [global_model.means_.ravel()[i] for i in data_label]
        labelled_data[b:e] = data_label[:]
        labelled_data_value[b:e] = data_label_value[:]

    return {"labelled_data": labelled_data, "labelled_data_value": labelled_data_value}

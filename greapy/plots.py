import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from greapy import GREA
from greapy.common import get_Mb_from_H0

Mb_SHOES = -19.253  # Eq.10 in https://arxiv.org/pdf/2112.04510
Mb_Planck = get_Mb_from_H0(67.36)


def fig1(
    kappa=np.linspace(3.2, 3.8, 6),
    omega_cdm=0.12,
    h=0.678,
    omega_b=0.0223,
    lw=1.5,
    cmap=cm.RdBu,
    ratio=False,
):
    """
    Generate a figure comparing the effective equation of state and Hubble parameter
    """
    from astropy.cosmology import FlatLambdaCDM

    Omega_m = (omega_cdm + omega_b) / h / h

    LCDM = FlatLambdaCDM(100 * h, Omega_m, Tcmb0=2.7255)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3), sharex=True)
    norm = mcolors.Normalize(vmin=min(kappa), vmax=max(kappa))

    labels = (
        [r"$w$", r"$H/H^{\Lambda\rm CDM}$"]
        if ratio
        else [r"$w(z)$", r"$a\,H(a)\,\rm [km/s/Mpc]$"]
    )

    for p in kappa:
        m = GREA(h, omega_cdm=omega_cdm, omega_b=omega_b, kappa=p)
        zp1 = 1 / m.a
        color = cmap(norm(p))
        scale = 1 / LCDM.H(zp1 - 1) if ratio else m.a

        axs[0].plot(
            zp1, m.w(m.a), c=color, label=r"$\alpha={:.2f}$".format(m.alpha), lw=lw
        )
        axs[1].plot(zp1, scale * m.Hubble(m.a), c=color, lw=lw)

    H_lims = [0.9, 1.18] if ratio else [50, 120]
    for ax, lbl, ylim in zip(
        axs.flatten(),
        labels,
        [[-1.2, -0.4], H_lims],
    ):
        ax.set_xlabel(r"$1+z$", fontsize="x-large")
        ax.set_ylabel(lbl, fontsize="x-large")
        ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.set_xlim([0.5, 1e1 + 1])
        ax.axvline(1, c="lightgray", ls="--")

    axs[0].plot(zp1, -1 * np.ones_like(m.a), c="k", ls="--", lw=1)
    axs[1].plot(zp1, scale * LCDM.H(zp1 - 1).value, c="k", ls="--", lw=1)
    axs[0].legend(ncols=2)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Adjust the position of the colorbar
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", pad=0.01, aspect=40)
    cbar.ax.set_position(
        [
            axs[1].get_position().x1 + 0.01,
            axs[1].get_position().y0,
            0.02,
            axs[1].get_position().height,
        ]
    )
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.set_xlabel(r"$\sqrt{-k}\eta_0$", fontsize="medium", labelpad=10)

    return fig


def fig2(
    chains,
    params_X=["H0", "Omega_m", "omega_cdm", "omega_b", "H0rd"],
    limits={},
    params_Y=["rd", "alpha"],
    colors=["gray", "C0", "C1", "C2", "dodgerblue", "olive"],  #'darkgreen'
    filled=[False, True, True, True, True],
    **kwargs,
):
    from getdist import plots

    g = plots.get_subplot_plotter(width_inch=8, subplot_size_ratio=1)
    g.settings.linewidth = 1.7
    g.rectangle_plot(
        params_X,
        params_Y,
        roots=list(chains.values()),
        filled=filled,
        colors=colors,
        legend_labels=list(chains.keys()),
        param_limits=limits,
        contour_lws=1.2,
        legend_ncol=3,
        **kwargs,
    )
    g.add_x_bands(73.05, 1.0, ax=g.subplots[0, 0])
    g.add_x_bands(73.05, 1.0, ax=g.subplots[1, 0])

    return g


def fig3(
    chains,
    params=["H0", "Omega_m", "omega_b", "omega_cdm", "alpha", "kappa"],
    colors=["gray", "C0", "C1", "C2", "dodgerblue", "olive"],
    limits={},
    filled=True,
):
    from getdist import plots

    g = plots.get_subplot_plotter(width_inch=6, subplot_size_ratio=1)
    g.triangle_plot(
        list(chains.values()),
        params,
        filled=filled,
        contour_colors=colors,
        legend_labels=list(chains.keys()),
        param_limits=limits,
    )
    return g


def fig4(
    chains,
    params=["w0", "wa"],
    colors=["gray", "C0", "C1", "C2", "dodgerblue", "olive"],
    filled=True,
    limits={},
    lw=1.5,
):
    from getdist import plots

    legend_kwargs = dict(
        bbox_to_anchor=[0.03, 0.98],
        legend_loc="upper left",
        fontsize=6,
        ncols=2,
        frameon=False,
    )

    g = plots.get_subplot_plotter()
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3))
    g.add_legend(legend_labels=list(chains.keys()), **legend_kwargs)
    g.plot_2d(
        list(chains.values()),
        params,
        ax=axs[1],
        filled=filled,
        colors=colors,
        legend_labels=list(chains.keys()),
        param_limits=limits,
        add_legend_proxy=True,
        alphas=[0.9] * 6,
        lws=[lw] * 6,
    )

    axs[1].set_xlim(limits["w0"])
    axs[1].set_ylim(limits["wa"])

    ax = axs[0]
    g.plot_2d(
        list(chains.values()),
        params,
        ax=ax,
        filled=filled,
        colors=colors,
        legend_labels=list(chains.keys()),
        param_limits=limits,
        add_legend_proxy=True,
        alphas=[0.9] * 6,
        lws=[lw] * 6,
    )
    # ax.set_xlim(limits["alpha"])
    # ax.set_ylabel(r"$P/P_\mathrm{max}$", fontsize=16)
    # ax.set_xlabel(r"$\alpha$", fontsize=16)
    # ax.yaxis.set_tick_params(reset=True)
    # ax.xaxis.set_tick_params(reset=True)
    for ax in axs:
        ax.axvline(-1, c="k", ls="--", lw=0.5)
        ax.axhline(0, c="k", ls="--", lw=0.5)

    fig.tight_layout()

    return fig, g


def fig4bis(
    chains,
    params=["w0", "wa"],
    colors=["gray", "C0", "C1", "C2", "dodgerblue", "olive"],
    filled=True,
    limits={},
    lw=1.5,
):
    from getdist import plots

    legend_kwargs = dict(
        bbox_to_anchor=[0.03, 0.98],
        legend_loc="upper left",
        fontsize=6,
        ncols=2,
        frameon=False,
    )

    g = plots.get_subplot_plotter()
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3))
    g.add_legend(legend_labels=list(chains.keys()), **legend_kwargs)
    g.plot_2d(
        list(chains.values()),
        params,
        ax=axs[0],
        filled=filled,
        colors=colors,
        legend_labels=list(chains.keys()),
        param_limits=limits,
        add_legend_proxy=True,
        alphas=[0.9] * 6,
        lws=[lw] * 6,
    )

    axs[0].axvline(-1, c="k", ls="--", lw=0.5)
    axs[0].set_xlim(limits["w0"])
    axs[0].set_ylim(limits["wa"])

    ax = axs[1]
    g.plot_1d(
        list(chains.values()),
        "alpha",
        colors=colors,
        normalized=False,
        lws=[lw] * 6,
        ax=ax,
    )
    ax.set_xlim(limits["alpha"])
    ax.set_ylabel(r"$P/P_\mathrm{max}$", fontsize=16)
    ax.set_xlabel(r"$\alpha$", fontsize=16)
    ax.yaxis.set_tick_params(reset=True)
    ax.xaxis.set_tick_params(reset=True)
    fig.tight_layout()

    return fig, g


def fig5(z, samples_w, samples_fde, lw=1.5):
    # labels = [r"$w$", r"$\rho^{\rm eff}_{\rm DE}/\rho^{\rm eff}_{\rm DE,0}$"]
    labels = [r"$w$", r"$f_{\rm DE}$"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharex=True)
    ax = axs[0]
    fill_between(z, samples_w, ax=ax, lw=lw, color="C0", alpha=0.3, label="GREA")
    ax.plot(z, -np.ones_like(z), lw=lw, ls="--", c="lightgray")
    ax.legend()
    ax.set_ylim(-1.2, -0.8)

    ax = axs[1]
    fill_between(z, samples_fde, ax=ax, lw=lw, color="C0", alpha=0.3, label="GREA")
    ax.plot(z, np.ones_like(z), lw=lw, ls="--", c="lightgray")
    ax.set_ylim(0.6, 1.2)

    for ax, lbl in zip(axs, labels):
        ax.set_xlim(0, 3)
        ax.set_ylabel(lbl, fontsize="xx-large")
        ax.set_xlabel(r"$z$", fontsize="xx-large")

    return fig


def fig6():
    return


def fig7(data_chi2, data_lnZ, models=None, vmax_chi2=10, vmax_lnZ=10, fname=None):
    output = fname if fname else "heatmap"

    # Plot chi2 heatmap
    fig1 = plot_heatmap(
        data_chi2,
        models,
        r"$\Delta\chi^2$",
        cmap="RdBu_r",
        heatmap_kwargs={"vmin": -vmax_chi2, "vmax": vmax_chi2},
        fname=fname if fname is None else output + "_chi2",
    )

    # Plot lnZ heatmap
    fig2 = plot_heatmap(
        data_lnZ,
        models,
        r" $\Delta\ln{\mathcal{Z}}$",
        heatmap_kwargs={"vmin": -vmax_lnZ, "vmax": vmax_lnZ},
        fname=fname if fname is None else output + "_lnZ",
    )

    return fig1, fig2


def fig8(
    kappa=np.linspace(3.2, 3.8, 6),
    ax=None,
    omega_cdm=0.12,
    h=0.678,
    omega_b=0.0223,
    sigma8=0.81,
    cmap=cm.RdBu,
    add_data=True,
):
    from greapy.grea import solve_growth, analytical_fsigma8
    from scipy.interpolate import UnivariateSpline

    """
    Generate a figure comparing the growth rate of structure for different values of kappa.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.figure

    norm = mcolors.Normalize(vmin=min(kappa), vmax=max(kappa))

    for p in kappa:
        m = GREA(h, omega_cdm=omega_cdm, omega_b=omega_b, kappa=p)
        color = cmap(norm(p))
        a = np.linspace(0.01, 1, 100)
        z = 1 / a - 1
        Hs = UnivariateSpline(m.a, m.Hubble(m.a) / m.Hubble(1), s=0)
        Hsp = Hs.derivative()
        delta, f, fsigma8 = solve_growth(a, m.Omega_m, sigma8, Hs, Hsp)
        ax.plot(z, fsigma8, c=color)  # , label=rf"$\alpha={m.alpha:.3f}$")

    ax.plot(
        z,
        analytical_fsigma8(a, m.Omega_bc, sigma8),
        c="k",
        ls="--",
        lw=1.0,
    )
    ax.set_xlim(1e-1, 15)
    ax.set_ylabel(r"$f\sigma_8$", fontsize="x-large")
    ax.set_xlabel(r"$z$", fontsize="x-large")
    ax.semilogx()

    if add_data:
        add_fs8_data(ax)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Adjust the position of the colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.01, aspect=40)
    cbar.ax.set_position(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.set_xlabel(r"$\sqrt{-k}\eta_0$", fontsize="medium", labelpad=10)

    return fig


def fig8_bis(
    kappa=np.linspace(3.2, 3.8, 6),
    omega_cdm=0.12,
    h=0.678,
    omega_b=0.0223,
    sigma8=0.81,
    ylims=[[0.2, 0.6], [0.54, 0.56]],
    lw=1.5,
    cmap=cm.RdBu,
    add_data=False,
):
    """
    Generate a figure comparing the effective equation of state and Hubble parameter
    """
    from scipy.interpolate import UnivariateSpline
    from greapy.growth import solve_growth, gamma, analytical_fsigma8, analytical_gamma

    fig, axs = plt.subplots(1, 2, figsize=(11, 3), sharex=True)
    norm = mcolors.Normalize(vmin=min(kappa), vmax=max(kappa))

    for p in kappa:
        m = GREA(h, omega_cdm=omega_cdm, omega_b=omega_b, kappa=p)
        color = cmap(norm(p))
        # a = np.linspace(0.06, 1, 1500)
        a = np.logspace(np.log10(0.06), 0, 1500)
        z = 1 / a - 1
        Hs = UnivariateSpline(m.a, m.Hubble(m.a) / m.Hubble(1), s=0)
        Hsp = Hs.derivative()
        delta, f, fsigma8 = solve_growth(a, m.Omega_m, sigma8, Hs, Hsp)

        # Plot f*sigma8
        ax = axs[0]
        ax.plot(z, fsigma8, c=color, lw=lw)  # , label=rf"$\alpha={m.alpha:.3f}$")

        # Plot gamma
        ax = axs[1]
        ax.plot(
            z, gamma(a, f, m), c=color, lw=lw, label=r"$\alpha={:.2f}$".format(m.alpha)
        )

    labels = [r"$f\sigma_8$", r"$\gamma$"]
    # labels = [r"$\gamma$", r"$f\sigma_8$"]

    for ax, lbl, ylim in zip(
        axs.flatten(),
        labels,
        ylims,
    ):
        ax.set_xlabel(r"$z$", fontsize="x-large")
        ax.set_ylabel(lbl, fontsize="x-large")
        # ax.set_xscale("log")
        ax.set_ylim(ylim)
        ax.set_xlim(1e-1, 3)

    if add_data:
        add_fs8_data(axs[0])

    axs[0].plot(
        z,
        analytical_fsigma8(a, m.Omega_bc, sigma8),
        c="k",
        ls="--",
        lw=1.0,
    )
    axs[1].plot(z, analytical_gamma(a, m.Omega_bc), c="k", ls="--", lw=1.0)
    axs[1].legend(ncols=2)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Adjust the position of the colorbar
    cbar = fig.colorbar(sm, ax=axs, orientation="vertical", pad=0.01, aspect=40)
    cbar.ax.set_position(
        [
            axs[1].get_position().x1 + 0.01,
            axs[1].get_position().y0,
            0.02,
            axs[1].get_position().height,
        ]
    )
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.set_xlabel(r"$\sqrt{-k}\eta_0$", fontsize="medium", labelpad=10)

    return fig


def addCPL(z, chain, legend=False, axs=None):
    r"""
    Add the 1 and 2 sigma bounds on w(z) and f_de(z) for a given chain of CPL parameters, \[w_0\] and \[w_a\].
    """
    a = 1.0 / (1.0 + z)

    def w(a, w0, wa):
        return w0 + wa * (1 - a)

    def fde(a, w0, wa):
        return np.exp(-3 * (1 - a) * wa) * a ** (-3 * (1 + w0 + wa))

    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 3), sharex=True)
    else:
        fig = axs[0].figure

    samples_w = w(a.reshape(-1, 1), chain["w0_fld"], chain["wa_fld"])
    samples_fde = fde(a.reshape(-1, 1), chain["w0_fld"], chain["wa_fld"])

    fill_between(
        1 / a - 1,
        samples_w.T,
        ax=axs[0],
        lw=1.5,
        color="lightgray",
        label=r"$w_0w_a\rm CDM$",
    )
    fill_between(1 / a - 1, samples_fde.T, ax=axs[1], lw=1.5, color="lightgray")
    if legend:
        axs[0].legend()
    return fig


def plot_distances(z, model):
    from astropy.cosmology import FlatLambdaCDM

    lcdm = FlatLambdaCDM(
        H0=100 * model.h,
        Om0=model.Omega_bc,
        Ob0=model.omega_b / model.h**2,
        Tcmb0=2.7259,
        Neff=3.044,
    )

    fig, (ax_H, ax_obs, ax_res) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [1, 3, 1], "hspace": 0.1},
    )
    ax_obs.plot([], [], label=r"$\Lambda$CDM", color="k", linestyle="--", lw=1)

    for m, kwargs in zip(
        [model, lcdm], [{"lw": 2}, {"linestyle": "--", "c": "k", "lw": 1.0}]
    ):
        ax_H.plot(z, m.H(z) / lcdm.H(z).value, **kwargs)
        ax_obs.plot(
            z, m.comoving_distance(z), label=r"Comoving Distance $D_M(z)$", **kwargs
        )
        ax_obs.plot(
            z, m.luminosity_distance(z), label=r"Luminosity Distance $D_L(z)$", **kwargs
        )
        ax_obs.plot(
            z,
            m.angular_diameter_distance(z),
            label=r"Angular Diameter Distance $D_A(z)$",
            **kwargs,
        )
        if m is model:
            ax_obs.legend()
    ax_res.plot(z, model.comoving_distance(z) / lcdm.comoving_distance(z).value, lw=2.0)
    ax_res.axhline(1, color="k", lw=1, ls="--")
    ax_obs.loglog()
    ax_H.set_ylabel(r"$H/H^{\Lambda\rm CDM}$", fontsize="x-large")
    ax_obs.set_ylabel("Distances [Mpc]", fontsize="x-large")
    ax_res.set_ylabel(r"$D/D^{\Lambda\rm CDM}$", fontsize="x-large")
    ax_res.set_xlabel(r"Redshift $z$", fontsize="x-large")
    return plt.gca()


def plot_heatmap(
    data,
    models,
    cbar_label,
    cmap="RdBu",
    heatmap_kwargs={},
    fname=None,
    extension=".png",
):
    import pandas as pd
    import seaborn as sns

    # Create a DataFrame
    df = pd.DataFrame(data, index=models)

    # Plotting the heatmap
    fig = plt.figure(figsize=(8, 7))
    ax = sns.heatmap(
        df, annot=True, cmap=cmap, cbar=False, annot_kws={"size": 16}, **heatmap_kwargs
    )

    # Create a colorbar
    norm = plt.Normalize(
        vmin=heatmap_kwargs.get("vmin"), vmax=heatmap_kwargs.get("vmax")
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_xlabel(cbar_label, fontsize=20)
    cbar.ax.xaxis.set_label_position("bottom")
    cbar.ax.xaxis.set_ticks_position("bottom")

    # Set x-labels horizontal and wrap text
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=0, ha="center", wrap=True, fontsize=14
    )

    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
    if fname:
        fig.savefig(fname + extension, bbox_inches="tight", dpi=300)
        print(f"Saved figure to {fname + extension}")
    return fig


def plot_tension_comparison(
    datasets,
    param="H0",
    mu_ref=73.04,
    sigma_ref=1.04,
    ref_label=r"SH$_0$ES",
    colors=None,
    alpha=1.0,
):
    """
    Grouped horizontal bar chart comparing Gaussian Tension across calibrations.

    Parameters
    ----------
    datasets : dict
        {calibration_label: {model_name: getdist_chain}}
    param : str
        Chain parameter name.
    mu_ref, sigma_ref : float
        Reference mean and sigma.
    ref_label : str
        Label for the reference in the title.

    Returns
    -------
    fig, ax
    """
    from .stats import gaussian_tension

    model_names = list(next(iter(datasets.values())).keys())
    n_models = len(model_names)
    n_ds = len(datasets)

    # Compute tensions for every (calibration, model) pair
    tensions_all = {}
    for label, chains_dict in datasets.items():
        tensions_all[label] = {
            name: gaussian_tension(
                chain.mean([param])[0], chain.std([param])[0], mu_ref, sigma_ref
            )
            for name, chain in chains_dict.items()
        }

    t_max = max(ts[m].tension for ts in tensions_all.values() for m in model_names)
    x_max = t_max + 1.2

    fig, ax = plt.subplots(figsize=(7, max(3, n_models * 0.9)))

    # Shaded tension regions — two overlapping spans create a darker >3σ zone
    ax.axvspan(1, x_max, color="lightgray", alpha=0.1, zorder=0)
    ax.axvspan(2, x_max, color="lightgray", alpha=0.3, zorder=0)
    ax.axvspan(3, x_max, color="lightgray", alpha=0.3, zorder=0)
    ax.axvspan(4, x_max, color="lightgray", alpha=0.3, zorder=0)
    ax.axvspan(5, x_max, color="lightgray", alpha=0.3, zorder=0)

    bar_h = 0.35
    y = np.arange(n_models)
    palette = colors if colors else [f"C{i}" for i in range(n_ds)]
    offsets = [-(n_ds - 1) * bar_h / 2 + i * bar_h for i in range(n_ds)]

    for (label, tensions), color, offset in zip(tensions_all.items(), palette, offsets):
        t_vals = [tensions[m].tension for m in model_names]
        ax.barh(y + offset, t_vals, height=bar_h, label=label, color=color, alpha=alpha)
        for yi, t in zip(y + offset, t_vals):
            ax.text(t + 0.06, yi, f"{t:.2f}$\\sigma$", va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(model_names)
    ax.set_xlabel(r"Gaussian Tension ($\sigma$)", fontsize=11)
    ax.set_title(
        rf"$H_0$ tension vs {ref_label} "
        rf"($H_0={mu_ref}\pm {sigma_ref}$ km/s/Mpc)",
        fontsize=10,
    )
    ax.set_xlim(0, x_max)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    return fig, ax


def add_fs8_data(ax):
    ax.errorbar(
        z_fs8,
        fs8_meas,
        yerr=fs8_err,
        fmt=".",
        c="k",
        alpha=1,
        markerfacecolor="white",
        label="DESI DR1 (ShapeFit+BAO)",
    )
    return


def fill_between(
    x: np.ndarray,
    samples_y: np.ndarray,
    label: str = None,
    ax=None,
    color: str = "gray",
    lw: float = 1.0,
    alpha: float = 0.5,
    quantiles: list = [2.3, 16, 50, 84, 97.7],
):
    """
    Plot median and quantiles for a given (flatten) array of samples

    Args:
        x (np.ndarray): an array with x-values as a numpy array.
        samples_y (np.ndarray): a numpy array with samples for the quantity f=y(x).
        label (str, optional): labels to use in the legend. Defaults to None.
        ax (_type_, optional): a matplotlib axes instance. If None, will create a single plot with default settings. Defaults to None.
        color (str, optional): color for the contours. Defaults to 'gray'.
        lw (float, optional): linewidth for the lines. Defaults to 2.
        alpha (float, optional): transparency of the contour colors. Defaults to 0.5.
        quantiles (list, optional): quantiles of the distribution to plot. Defaults to [2.3, 16, 50, 84, 97.7].
    """
    if ax is None:
        fig, ax = plt.subplots()

    qs = np.percentile(samples_y, q=quantiles, axis=0)
    idx = len(qs) // 2
    median = qs[idx]
    for i in range(1, idx + 1):
        ax.fill_between(
            x.flatten(),
            qs[idx - i].flatten(),
            qs[idx + i].flatten(),
            color=color,
            lw=lw,
            alpha=alpha / i,
        )
    ax.plot(x, median, label=label, c=color, ls="-", lw=lw)

    if ax is None:
        return ax


def build(func):
    def wrapper(fig, chains, *args, output_dir=None, fig_fmt="png", **kwargs):
        labels = list(chains.keys())
        result = None
        for i in range(1, len(labels) + 1):
            i_chains = {lbl: chains[lbl] for lbl in labels[:i]}
            result = func(fig, i_chains, *args, **kwargs)
            if output_dir is not None:
                result.export(output_dir + f"/build{i}.{fig_fmt}", dpi=300)
            else:
                print(f"No output specified, not exporting figure {i}")
        return result

    return wrapper


@build
def build_figure(fig, chains, *args, output_dir=None, fig_fmt="png", **kwargs):
    return fig(chains, *args, **kwargs)


def get_mb(cosmo, z=None, Mb=Mb_SHOES):
    from greapy import GREA

    def dL(z):
        if isinstance(cosmo, GREA):
            return (1 + z) * cosmo.comoving_distance(z)
        else:
            return (1 + z) * cosmo.comoving_distance(z).value

    mb = 5 * np.log10(dL(z)) + 25 + Mb
    return mb


def plot_SN_data(
    Mb=Mb_SHOES,
    ax=None,
    SN=None,
    quiet=True,
    data_kwargs=dict(fmt=".", c="k", alpha=1, markerfacecolor="white"),
):
    from astropy.cosmology import Flatw0waCDM

    # Fiducial Planck 18 LCDM cosmology
    lcdm = Flatw0waCDM(67.36, 0.315, w0=-1, wa=0, Tcmb0=2.7255)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))

    if SN:
        ax.errorbar(
            SN.zcmb, SN.mag - get_mb(lcdm, Mb=Mb), fmt=".", c="lightgray", alpha=0.1
        )

    ax.errorbar(
        binned_PP["z_bin"],
        binned_PP["mean_mu"] - get_mb(lcdm, z=binned_PP["z_bin"], Mb=Mb - Mb_SHOES),
        binned_PP["std_mu"],
        **data_kwargs,
    )

    if not quiet:
        return fig, ax


def plot_SN_residuals(
    z,
    cosmo,
    Mb=Mb_SHOES,
    ax=None,
    plot_data=False,
    data_kwargs=dict(fmt=".", c="k", alpha=1, markerfacecolor="white"),
    **plt_kwargs,
):
    from astropy.cosmology import Flatw0waCDM

    # Fiducial Planck 18 LCDM cosmology
    lcdm = Flatw0waCDM(67.36, 0.315, w0=-1, wa=0, Tcmb0=2.7255)

    if plot_data:
        plot_SN_data(Mb, quiet=True, ax=ax, data_kwargs=data_kwargs)

    ax.plot(z, get_mb(cosmo, z=z) - get_mb(lcdm, z=z), **plt_kwargs)
    ax.set_ylabel(r"$\Delta\mu$", fontsize="x-large")
    ax.set_xlabel(r"$z$", fontsize="x-large")


def plot_BAO_data(axs=None, return_factor=True, quiet=False):
    from greapy.common import get_dV_rs, get_F_AP
    from astropy.cosmology import Flatw0waCDM

    # Fiducial Planck 18 LCDM cosmology
    lcdm = Flatw0waCDM(67.36, 0.315, w0=-1, wa=0, Tcmb0=2.7255)

    fid_factor = (1 / get_dV_rs(zeff, lcdm), 1 / get_F_AP(zeff[1:], lcdm))

    observables = (
        [dV_rd * fid_factor[0], dV_rd_uncertainties * fid_factor[0]],
        [DM_DH * fid_factor[1], DM_DH_uncertainties * fid_factor[1]],
    )

    ylabels = [r"$(D_V/r_d)/(D_V/r_d)^{\rm fid}$", r"$(D_M/D_H)/(D_M/D_H)^{\rm fid}$"]

    ylims = ([0.95, 1.05], [0.93, 1.07])

    if axs is None:
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(12, 3),
            sharex=False,
            sharey=False,
            gridspec_kw={"hspace": 0.2, "wspace": 0.2},
        )

    for ax, obs, ylbl, ylim in zip(axs, observables, ylabels, ylims):
        ax.errorbar(
            zeff if ax == axs[0] else zeff[1:],
            *obs,
            fmt=".",
            c="k",
            markerfacecolor="white",
            label="DESI DR2" if ax == axs[0] else None,
        )
        ax.axhline(1, alpha=0.5, c="lightgray")
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylbl, fontsize="large")

    axs[0].legend(fontsize="large")

    if return_factor:
        return fig, axs, fid_factor

    if not quiet:
        return fig, axs


# Values extracted from the DESI DR2 table
zeff = np.array([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
dV_rd = np.array([7.942, 12.720, 16.050, 19.721, 24.252, 26.055, 31.267])
dV_rd_uncertainties = np.array([0.075, 0.099, 0.110, 0.091, 0.174, 0.398, 0.256])

DM_DH = np.array([0.622, 0.892, 1.223, 1.948, 2.386, 4.518])
DM_DH_uncertainties = np.array([0.017, 0.021, 0.019, 0.045, 0.136, 0.097])

# Redshifts and fiducial fσ₈(z) values (from Table 11)
z_fs8 = [0.295, 0.510, 0.706, 0.919, 1.317, 1.491]
fs8_fid = [0.4723, 0.4733, 0.4608, 0.4398, 0.3944, 0.3750]

# Extracted values of (fs8 / fs8_fid) ± errors (from new table, ShapeFit+BAO row)
fs8_ratio = [0.80, 1.09, 1.05, 0.96, 0.95, 1.16]
fs8_ratio_err = [0.20, 0.13, 0.12, 0.11, 0.08, 0.12]

# Compute absolute fs8 measurements and uncertainties
fs8_meas = np.array(fs8_ratio) * np.array(fs8_fid)
fs8_err = np.array(fs8_ratio_err) * np.array(fs8_fid)


binned_PP = {}
binned_PP["z_bin"] = np.array(
    [
        0.01116293,
        0.0133668,
        0.01600578,
        0.01916576,
        0.02294961,
        0.0274805,
        0.03290591,
        0.03940244,
        0.04718157,
        0.05649652,
        0.06765049,
        0.08100656,
        0.09699949,
        0.11614987,
        0.13908106,
        0.1665395,
        0.19941899,
        0.2387898,
        0.2859335,
        0.34238466,
        0.40998084,
        0.49092238,
        0.58784401,
        0.70390064,
        0.84287005,
        1.00927586,
        1.20853477,
        1.44713289,
        1.73283686,
        2.07494668,
    ]
)

binned_PP["mean_mu"] = np.array(
    [
        33.36205954,
        33.7540725,
        34.11551388,
        34.56042339,
        34.88940671,
        35.26718532,
        35.70517341,
        36.09887092,
        36.49200132,
        36.92427335,
        37.32869082,
        37.70352022,
        38.23478287,
        38.58560125,
        39.00214468,
        39.4227382,
        39.82683467,
        40.26978664,
        40.73504561,
        41.15270732,
        41.61949385,
        42.05098942,
        42.5221873,
        43.03344659,
        43.27504196,
        44.05055077,
        44.71548046,
        44.90434096,
        45.33583338,
        45.4233,
    ]
)

binned_PP["std_mu"] = np.array(
    [
        0.05397207,
        0.0388405,
        0.02752044,
        0.0275267,
        0.01874069,
        0.01918459,
        0.01736414,
        0.02089841,
        0.02134775,
        0.02566793,
        0.03032436,
        0.0265103,
        0.0412344,
        0.02488712,
        0.01892019,
        0.01707523,
        0.01692032,
        0.01472418,
        0.01451447,
        0.01303987,
        0.01794737,
        0.01943375,
        0.02331822,
        0.02881102,
        0.05043822,
        0.08790702,
        0.09244049,
        0.0888809,
        0.14261261,
        0.25966946,
    ]
)

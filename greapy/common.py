"""The common module contains common functions and classes used by the other modules."""

import os
import numpy as np

# Physical constants
C_KMS: float = 299792.458  # Speed of light in km/s


def get_Rm1(samples: dict):
    return [
        print(f"The R-1 for {lbl} is {chain.getGelmanRubin():.3f}")
        for lbl, chain in samples.items()
    ]


def extract_chi2(dataset, path):
    file_path = os.path.join(path, f"{dataset}.minimum.txt")
    chi2_values = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        header = lines[0].strip().split()
        chi2_index = header.index("chi2") - 1

        for line in lines[1:]:
            values = line.strip().split()
            chi2_values.append(float(values[chi2_index]))

    return chi2_values[0]


def extract_lnZ(file, path):
    """
    Extract the main logZ value from a .logZ file.

    Args:
        filepath (str): Path to the .logZ file

    Returns:
        float: The logZ value from the second line

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the logZ value cannot be parsed
    """
    filepath = os.path.join(path, file + ".logZ")
    try:
        with open(filepath, "r") as file:
            lines = file.readlines()

        # Look for the line that starts with "logZ:" (should be line 2, index 2)
        for line in lines:
            line = line.strip()
            if line.startswith("logZ:") and not line.startswith("logZstd:"):
                # Extract the value after "logZ:"
                logz_str = line.split("logZ:")[1].strip()
                return float(logz_str)

        raise ValueError("Could not find logZ value in the file")

    except FileNotFoundError:
        # raise FileNotFoundError(f"File not found: {filepath}")
        print(
            f"File not found: {filepath}, returning 0 instead",
        )
        return 0
    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not parse logZ value from file: {e}")


def get_samples_w_fde(z, chain, samples_fn, param_names, Nsamples=500):
    """
    Load previously computed samples of w(z) and fde(z) or compute them from the chains.
    """
    from tqdm import tqdm
    from greapy import GREA

    def get_w_fde(theta):
        m = GREA(*theta)
        w = m.w(1 / (1 + z))
        fde = m.fde(1 / (1 + z))
        return w, fde

    if os.path.isfile(samples_fn):
        # Load the samples of w(z)
        samples = np.load(samples_fn)
        print(samples["w"].shape, samples["fde"].shape)
        print(
            f" N={len(samples['w'])} samples of w(z), fde(z),etc loaded successfully!"
        )
    else:
        print(
            "\nPreviously computed samples of w(z) not found! Continuing with calculations..."
        )

        ## Retrieve MCMC samples and compute w(z) for each of them
        ind = np.random.randint(len(chain.samples), size=Nsamples)
        weights = chain.weights[ind]
        thetas = np.array([chain[p] for p in param_names]).T[ind]
        tmp = np.array([get_w_fde(theta) for theta in tqdm(thetas)])
        samples = {lbl: tmp[:, i, :] for i, lbl in zip([0, 1], ["w", "fde"])}
        samples["weights"] = weights
        samples["idxs"] = ind
        np.savez_compressed(
            samples_fn, w=samples["w"], fde=samples["fde"], weights=weights, idxs=ind
        )

    return samples


def get_dV_rs(z, cosmo):
    from greapy import GREA

    rd = cosmo.rdrag if isinstance(cosmo, GREA) else 147.09
    H = cosmo.H if isinstance(cosmo, GREA) else lambda z: cosmo.H(z).value
    dM = (
        cosmo.comoving_distance
        if isinstance(cosmo, GREA)
        else lambda z: cosmo.comoving_distance(z).value
    )
    dH = C_KMS / H(z)
    dV = (z * dH * dM(z) ** 2) ** (1 / 3)
    return dV / rd


def get_F_AP(z, cosmo):
    from greapy import GREA

    H = cosmo.H if isinstance(cosmo, GREA) else lambda z: cosmo.H(z).value
    dM = (
        cosmo.comoving_distance
        if isinstance(cosmo, GREA)
        else lambda z: cosmo.comoving_distance(z).value
    )
    return dM(z) * H(z) / C_KMS


def get_Mb_from_H0(H0, Mb_fid=-19.25, H0_fid=73.05):
    return Mb_fid + 5 * np.log10(H0 / H0_fid)


# def get_bestfit(file):
#     column_names = pl.read_csv(file, has_header=True).columns[0].split()[1:]
#     point = np.loadtxt(file)
#     return pl.DataFrame({col: val for col, val in zip(column_names, point)})

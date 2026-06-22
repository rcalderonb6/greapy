"""Common functions and utilities shared across greapy modules."""

import os
import numpy as np
from greapy.stats import gaussian_tension, gaussian_tension_from_chain, get_Rm1  # noqa: F401
from greapy.utils import is_monotonic_increasing  # noqa: F401

# Physical constants
C_KMS: float = 299792.458  # Speed of light in km/s


def extract_chi2(dataset, path):
    """Extract the best-fit $\chi^2$ value from a Cobaya `.minimum.txt` file.

    Args:
        dataset: Dataset name used to construct the filename
            (`<dataset>.minimum.txt`).
        path: Directory containing the `.minimum.txt` file.

    Returns:
        Best-fit $\chi^2$ value from the first data row.
    """
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


def get_bestfit(dataset, path, parameters=None):
    """Extract best-fit parameter values from a Cobaya `.minimum.txt` file.

    Args:
        dataset: Dataset name used to construct the filename
            (`<dataset>.minimum.txt`).
        path: Directory containing the `.minimum.txt` file.
        parameters: Optional list of parameter names to extract. If `None`,
            all parameters are returned.

    Returns:
        Dictionary mapping parameter names to their best-fit float values.

    Raises:
        FileNotFoundError: If the `.minimum.txt` file does not exist.
        ValueError: If the file format is invalid or a value cannot be parsed.
    """
    file_path = os.path.join(path, f"{dataset}.minimum.txt")

    try:
        with open(file_path, "r") as file:
            lines = file.readlines()

        if len(lines) < 2:
            raise ValueError("File must contain at least a header and one data row")

        # Parse header (parameter names) - handle potential spacing issues
        header_line = lines[0].strip()
        if not header_line.startswith("#"):
            raise ValueError("First line should be a header starting with '#'")

        # Remove the '#' and split by whitespace, filtering out empty strings
        header_parts = header_line[1:].split()
        param_names = [part for part in header_parts if part.strip()]

        # Parse the data row (best-fit values)
        data_line = lines[1].strip()
        data_parts = data_line.split()
        data_values = [part for part in data_parts if part.strip()]

        if len(data_values) != len(param_names):
            raise ValueError(
                f"Number of values ({len(data_values)}) doesn't match number of parameters ({len(param_names)})"
            )

        # Convert values to floats and create dictionary
        bestfit_dict = {}
        for param, value_str in zip(param_names, data_values):
            try:
                bestfit_dict[param] = float(value_str)
            except ValueError:
                raise ValueError(
                    f"Could not convert '{value_str}' to float for parameter '{param}'"
                )
        if parameters:
            return {param: bestfit_dict[param] for param in parameters}
        return bestfit_dict

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error parsing file {file_path}: {e}")


def extract_lnZ(file, path):
    """Extract the log-evidence $\ln Z$ from a PolyChord `.logZ` file.

    Args:
        file: Base filename (without extension) used to construct
            `<file>.logZ`.
        path: Directory containing the `.logZ` file.

    Returns:
        The $\ln Z$ value parsed from the file.

    Raises:
        FileNotFoundError: If the `.logZ` file does not exist.
        ValueError: If the `logZ` value cannot be found or parsed.
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
        raise FileNotFoundError(f"File not found: {filepath}")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Could not parse logZ value from file: {e}")


def get_samples_w_fde(z, chain, samples_fn, param_names, Nsamples=500):
    """Load or compute posterior samples of $w(z)$ and $f_\mathrm{de}(z)$.

    If a previously computed `.npz` file exists at `samples_fn`, it is loaded
    directly. Otherwise, samples are drawn from the MCMC chain and computed
    on-the-fly, then saved for future use.

    Args:
        z: Redshift array at which to evaluate $w(z)$ and $f_\mathrm{de}(z)$.
        chain: GetDist `MCSamples` object containing the posterior chain.
        samples_fn: Path to the `.npz` cache file (read if it exists, written
            if it does not).
        param_names: List of parameter names to extract from the chain, in the
            order expected by the `GREA` constructor.
        Nsamples: Number of posterior samples to draw when computing from
            scratch. Default is 500.

    Returns:
        Dictionary with keys `"w"`, `"fde"` (arrays of shape
        `(Nsamples, len(z))`), `"weights"`, and `"idxs"`.
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


def get_dV_rs(z, cosmo, rd=147.09):
    """Compute the BAO volume-averaged distance ratio $D_V(z)/r_s$.

    Args:
        z: Redshift at which to evaluate the distance.
        cosmo: Cosmology object — either a `GREA` instance or an Astropy
            cosmology with `.H(z)` and `.comoving_distance(z)` methods.
        rd: Sound horizon scale in Mpc. Default is 147.09 Mpc.

    Returns:
        Dimensionless ratio $D_V(z) / r_d$, where
        $D_V = (z \, D_H \, D_M^2)^{1/3}$ and $D_H = c/H(z)$.
    """
    from greapy import GREA

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
    """Compute the Alcock-Paczynski parameter $F_\mathrm{AP}(z)$.

    Args:
        z: Redshift at which to evaluate the AP parameter.
        cosmo: Cosmology object — either a `GREA` instance or an Astropy
            cosmology with `.H(z)` and `.comoving_distance(z)` methods.

    Returns:
        Dimensionless AP parameter $F_\mathrm{AP} = D_M(z) H(z) / c$.
    """
    from greapy import GREA

    H = cosmo.H if isinstance(cosmo, GREA) else lambda z: cosmo.H(z).value
    dM = (
        cosmo.comoving_distance
        if isinstance(cosmo, GREA)
        else lambda z: cosmo.comoving_distance(z).value
    )
    return dM(z) * H(z) / C_KMS


def get_Mb_from_H0(H0, Mb_fid=-19.253, H0_fid=73.04):
    """Convert a Hubble constant value to a Type Ia SN absolute magnitude $M_b$.

    Uses the standard distance-ladder relation between $H_0$ and $M_b$:

    $$M_b = M_b^\mathrm{fid} + 5 \log_{10}(H_0 / H_0^\mathrm{fid})$$

    Args:
        H0: Hubble constant in km/s/Mpc.
        Mb_fid: Fiducial absolute magnitude. Default is -19.253 (Riess et al. 2022).
        H0_fid: Fiducial Hubble constant in km/s/Mpc. Default is 73.04.

    Returns:
        Absolute magnitude $M_b$ corresponding to `H0`.
    """
    return Mb_fid + 5 * np.log10(H0 / H0_fid)

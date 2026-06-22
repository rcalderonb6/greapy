#!/usr/bin/env python3
"""
Generate a JCAP-style LaTeX table comparing parameter constraints from multiple MCMC chains.
"""

import argparse
from getdist import loadMCSamples


def make_table(chains, skip=0.3, params=None, ci="sigma", transpose=False, output=None):
    """Generate a JCAP-style LaTeX table of parameter constraints from MCMC chains.

    Args:
        chains: List of chain file prefixes (datasets) to load via GetDist.
        skip: Fraction of each chain to discard as burn-in. Default is 0.3.
        params: List of parameter names to include as table rows (or columns
            if transposed).
        ci: Confidence interval format. Options:

            - `"sigma"` — display mean +/- sigma.
            - `"68"` — display 68% credible interval.
            - `"95"` — display 95% credible interval.
        transpose: If `True`, datasets are rows and parameters are columns.
            Default is `False`.
        output: Path to write the `.tex` file. If `None`, the table is
            printed to stdout.

    Returns:
        LaTeX source string for the complete table environment.
    """
    # Load each chain into a GetDist MCSamples object
    samples = []
    for prefix in chains:
        try:
            samp = loadMCSamples(prefix, settings={"ignore_rows": skip})
            samples.append(samp)
        except Exception as e:
            print(f"Could not load chains for prefix '{prefix}': {e}")

    # Determine which confidence limits to use
    if ci == "sigma":
        use_sigma = True
    else:
        use_sigma = False
        # map '68' -> 0, '95' -> 1 as index for par.limits
        ci_index = 0 if ci == "68" else 1

    # Validate parameters exist in all chains
    for pname in params:
        for samp in samples:
            param_list = samp.getParamNames().list()  # all parameter names in chain
            if pname not in param_list:
                print(f"Parameter '{pname}' not found in chain {samp.name_tag}.")

    # Build table header
    if transpose:
        header_labels = ["Dataset"] + params
    else:
        header_labels = ["Parameter"] + chains

    # Begin LaTeX table code
    tab = []
    tab.append("\\begin{table}[tbp]")
    tab.append("\\centering")
    # Column alignment: one left (l) and rest center (c)
    ncols = len(header_labels)
    align = "l" + " c" * (ncols - 1)
    tab.append(f"\\begin{{tabular}}{{{align}}}")
    tab.append("\\hline")
    # Header row
    header_row = " & ".join(header_labels) + " \\\\"
    tab.append(header_row)
    tab.append("\\hline")
    # Fill rows
    if transpose:
        # Each dataset is a row
        for i, samp in enumerate(samples):
            row = [chains[i]]
            stats = samp.getMargeStats()
            for pname in params:
                par = stats.parWithName(pname)
                if use_sigma:
                    # mean ± sigma
                    val = f"{par.mean:.3g}"
                    err = f"{par.err:.2g}"
                    entry = f"${val}\\pm {err}$"
                else:
                    # credible interval
                    lim = par.limits[ci_index]
                    if lim.lower is None or lim.upper is None:
                        entry = "N/A"
                    else:
                        mean = par.mean
                        low = mean - lim.lower
                        high = lim.upper - mean
                        entry = f"$%.3f^{{+%.3f}}_{{-{low:.3f}}}$" % (mean, high)
                row.append(entry)
            tab.append(" & ".join(row) + " \\\\")
    else:
        # Each parameter is a row
        for pname in params:
            row = [f"${pname}$"]  # parameter names in math mode
            for samp in samples:
                stats = samp.getMargeStats()
                par = stats.parWithName(pname)
                if use_sigma:
                    val = f"{par.mean:.3g}"
                    err = f"{par.err:.2g}"
                    entry = f"${val}\\pm {err}$"
                else:
                    lim = par.limits[ci_index]
                    if lim.lower is None or lim.upper is None:
                        entry = "N/A"
                    else:
                        mean = par.mean
                        low = mean - lim.lower
                        high = lim.upper - mean
                        entry = f"$%.3f^{{+%.3f}}_{{-{low:.3f}}}$" % (mean, high)
                row.append(entry)
            tab.append(" & ".join(row) + " \\\\")
    tab.append("\\hline")
    tab.append("\\end{tabular}")
    # Caption and label (example caption – user should customize as needed)
    tab.append(
        "\\caption{\\label{tab:mytable} Comparison of parameter constraints (CID).}"
    )
    tab.append("\\end{table}")

    latex_code = "\n".join(tab)
    # Output to file or stdout
    if output:
        with open(output, "w") as f:
            f.write(latex_code + "\n")
    else:
        print(latex_code)
    print("Table generated successfully!")
    return latex_code


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Create a JCAP-style LaTeX table from GetDist chains"
    )
    parser.add_argument(
        "-c",
        "--chains",
        nargs="+",
        required=True,
        help="List of chain prefixes (datasets). E.g. chain1 chain2 etc.",
    )
    parser.add_argument(
        "-p",
        "--params",
        nargs="+",
        required=True,
        help="List of parameter names to include in the table.",
    )
    parser.add_argument(
        "--ci",
        choices=["sigma", "68", "95"],
        default="sigma",
        help="Confidence output: 'sigma' for mean +/- sigma, '68' or '95' for two-sided CL.",
    )
    parser.add_argument(
        "-t",
        "--transpose",
        action="store_true",
        help="Transpose table (datasets as rows, parameters as columns).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output .tex filename (default: print to stdout).",
    )

    parser.add_argument(
        "-s",
        "--skip",
        type=float,
        default=0.3,
        help="Fraction of the chain to ignore as burn in.",
    )

    args = parser.parse_args()

    make_table(
        chains=args.chains,
        skip=args.skip,
        params=args.params,
        ci=args.ci,
        transpose=args.transpose,
        output=args.output,
    )


if __name__ == "__main__":
    main()

"""Console script for greapy."""

import typer

app = typer.Typer()


@app.command()
def sample(
    likelihoods: str = typer.Argument(..., help="Likelihoods to use for sampling"),
    output: str = typer.Option(
        "test", "--output", "-o", help="Output file for results"
    ),
    priors: str = typer.Option(
        "baseline", "--priors", "-p", help="Priors to use for sampling"
    ),
    method: str = typer.Option(
        "MCMC",
        "--method",
        "-m",
        help="Sampling method to use (e.g., MCMC, Nested Sampling)",
    ),
    resume: bool = typer.Option(
        False, "--resume", "-r", help="Resume from previous run"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
    test: bool = typer.Option(False, "--test", "-t", help="Run in test mode"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Clean up and start new run"
    ),
    Rminus1: float = typer.Option(
        0.1, "--Rminus1", "-r", help="R-1 parameter for sampling"
    ),
    minimize: bool = typer.Option(
        False,
        "--minimize",
        "-min",
        help="Compute the MAP (Maximum A Posteriori) estimate from the chains",
    ),
):
    """Run the sampling process with the specified parameters."""
    from greapy.cobaya import run_mcmc
    from cobaya.yaml import yaml_load_file

    priors = yaml_load_file(priors)

    results = run_mcmc(
        likelihoods,
        priors,
        method,
        output,
        debug=debug,
        test=test,
        resume=resume,
        Rminus1=Rminus1,
        force=force,
    )
    if minimize:
        print("Computing the MAP (Maximum A Posteriori) estimate from the chains...")
        # results = compute_map(results, output)
    print(f"Results saved to {output}")
    return results


@app.command()
def plot():
    """Placeholder for plotting functionality."""
    from greapy.plots import fig1
    import numpy as np

    fig = fig1(np.linspace(3.2, 4.2, 10))
    fig.savefig("fig1.png")
    pass


@app.command()
def table(
    chains: list[str] = typer.Option(
        ...,
        "--chains",
        "-c",
        help="Space-separated list of chain prefixes (datasets). E.g. 'chain1 chain2'",
    ),
    params: list[str] = typer.Option(
        ...,
        "--params",
        "-p",
        help="Space-separated list of parameter names. E.g. 'omega_b omega_cdm h'",
    ),
    output: str = typer.Option(
        None, "--output", "-o", help="Output .tex filename (default: print to stdout)"
    ),
    ci: str = typer.Option(
        "sigma",
        "--ci",
        help="Confidence output: 'sigma' for mean±sigma, '68' or '95' for two-sided CL",
    ),
    transpose: bool = typer.Option(
        False,
        "--transpose",
        "-t",
        help="Transpose table (datasets as rows, parameters as columns)",
    ),
    skip: float = typer.Option(
        0.3, "--skip", "-s", help="Fraction of the chain to ignore as burnin"
    ),
):
    """Generate a JCAP-style LaTeX table comparing parameter constraints from multiple MCMC chains."""
    from greapy.textable import make_table

    # Validate ci parameter
    if ci not in ["sigma", "68", "95"]:
        typer.echo(
            f"Error: ci must be one of 'sigma', '68', '95', got '{ci}'", err=True
        )
        raise typer.Exit(1)

    make_table(
        chains=chains,
        skip=skip,
        params=params,
        ci=ci,
        transpose=transpose,
        output=output,
    )


def main():
    return app()


if __name__ == "__main__":
    main()

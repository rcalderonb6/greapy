import marimo

__generated_with = "0.14.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    slider=mo.ui.slider(0,10,1)
    return (slider,)


@app.cell
def _(slider):
    slider
    return


@app.cell
def _(slider):
    5*slider.value
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

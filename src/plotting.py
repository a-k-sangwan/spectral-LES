from matplotlib.pyplot import subplots, rcParams

def gen_plot(xlabel="", ylabel="", fSize=(13, 10)):
    fig, ax = subplots(figsize=fSize)
    rcParams.update(
    {"font.size": 25, "font.family": "times new roman", "mathtext.fontset": "stix", "lines.linewidth": 3, "savefig.dpi": 300}
    )
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25, rotation=90, labelpad=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    return fig, ax
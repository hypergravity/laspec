from matplotlib import rc, rcParams


def set_xminor(b=True):
    rcParams["xtick.minor.bottom"] = b  ## draw x axis bottom minor ticks
    rcParams["xtick.minor.top"] = b
    rcParams["xtick.minor.visible"] = b
    return


def set_yminor(b=True):
    rcParams["ytick.minor.left"] = b  ## draw y axis left minor ticks
    rcParams["ytick.minor.right"] = b  ## draw y axis right minor ticks
    rcParams["ytick.minor.visible"] = b  ## visibility of minor ticks on y-axis
    return


def set_cham(fontsize=15, xminor=True, yminor=True, latex=True,
             xtick_top=True, xtick_bottom=True, xtick_major_size=4, xtick_minor_size=2,
             xtick_major_width=1.2, xtick_direction="in",
             ytick_left=True, ytick_right=True, ytick_major_size=4, ytick_minor_size=2,
             ytick_major_width=1.2, ytick_direction="in",
             ):

    if latex:
        rc('text', usetex=True)
        rcParams["mathtext.fontset"] = "cm"
        rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    else:
        rc('text', usetex=False)
        rcParams["mathtext.fontset"] = "dejavusans"
        rcParams['text.latex.preamble'] = ""

    # font size
    rcParams["font.size"] = fontsize

    # xticks
    rcParams["xtick.top"] = xtick_top  # draw ticks on the top side
    rcParams["xtick.bottom"] = xtick_bottom  # draw ticks on the bottom side
    rcParams["xtick.major.size"] = xtick_major_size  # major tick size in points
    rcParams["xtick.minor.size"] = xtick_minor_size  # minor tick size in points
    rcParams["xtick.major.width"] = xtick_major_width  # major tick width in points
    rcParams["xtick.direction"] = xtick_direction  # direction: {in, out, inout}

    # yticks
    rcParams["ytick.left"] = ytick_left  # draw ticks on the left side
    rcParams["ytick.right"] = ytick_right  # draw ticks on the right side
    rcParams["ytick.major.size"] = ytick_major_size  # major tick size in points
    rcParams["ytick.minor.size"] = ytick_minor_size  # minor tick size in points
    rcParams["ytick.major.width"] = ytick_major_width  # major tick width in points
    rcParams["ytick.direction"] = ytick_direction  # direction: {in, out, inout}

    set_xminor(xminor)
    set_yminor(yminor)
    return

# capture cursor position ===============

# ref:
# https://matplotlib.org/stable/users/event_handling.html

# import matplotlib.pylab as plt
# import numpy as np
#
# f,a = plt.subplots()
# x = np.linspace(1,10,100)
# y = np.sin(x)
# a.plot(x,y)
# pos = []
# def onclick(event):
#     pos.append([event.xdata,event.ydata])
# f.canvas.mpl_connect('button_press_event', onclick)
# f.show()


def black_labels(labels, ind_black):
    # labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if i in ind_black:
            label.set_color("white")
            label.set_weight(600)
            label.set_bbox(
                dict(
                    boxstyle="round,pad=.25",
                    linewidth=0.5,
                    facecolor="black",
                    edgecolor="black",
                )
            )
    return


if __name__ == "__main__":
    set_cham()
    set_xminor()
    set_yminor()

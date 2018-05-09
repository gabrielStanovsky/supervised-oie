""" Usage:
   pr_plot --in=DIR_NAME [--out=OUTPUT_DIR] [--outputtype=FILETYPE]

Output PR curve and corresponding AUC curves to file.
Filename will be pr.filetype, auc.filetype under the given folder.

Options:
  --in=DIR_NAME            Folder in which to search for *.dat files, all of which should be in a P/R column format (outputs from benchmark.py)
  --out=OUTPUT_FILENAME    Output directory
  --filetype               Will determine the format. Possible formats: pdf, pgf, png
"""

import os
import ntpath
import numpy as np
from glob import glob
from docopt import docopt
import matplotlib.pyplot as plt
import logging
from operator import itemgetter
from scipy.integrate import simps
import pdb
logging.basicConfig(level = logging.INFO)

def trend_name(path):
    ''' return a system trend name from dat file path '''
    head, tail = ntpath.split(path)
    ret = tail or ntpath.basename(head)
    return ret.split('.')[0]

def get_pr(path):
    ''' get PR curve from file '''
    with open(path) as fin:
        # remove header line
        fin.readline()
        [p, r] = zip(*[map(lambda x: float(x), line.strip().split('\t')) for line in fin])
        return p, r

NICE_COLORS = [u'#ff7f0e', u'#1f77b4', u'#2ca02c', u'#d62728']
NICE_MARKERS = ['o', '^', 's', "v"]
NICE_LINE_STYLES = [':', '-.', '--', '-']

def plot_pr_curve(pr_ls, filename):
    """
    Plot PR curves to file.
    pr_ls - List of curve names and list of AUC values and corresponding colors
    filename - In which to save the figure.
    """
    fig = plt.figure(frameon=False)
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    colors = []

    for line_index, (name, (p, r)) in enumerate(pr_ls):
        logging.debug(name)
        lines = ax.plot(r,
                        p,
                        label = name,
                        color = NICE_COLORS[line_index],
                        marker= NICE_MARKERS[line_index],
                        markerfacecolor = 'none',
                        markevery = len(p) - 1,
                        markersize = 6,
                        linewidth = 1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        colors.append(lines[0].get_color())

    # Set figure properties and save
    logging.info("Plotting P/R graph to {}".format(filename))
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    leg = ax.legend(loc="lower right",
                    borderaxespad=0.2)

    leg.get_frame().set_linewidth(0.0)

    plt.savefig(filename,
                bbox_inches='tight')
    return colors


def plot_auc(auc_ls, filename, output_folder):
    """
    Plot AUC bars to file.
    pr_dic - List of curve names and AUC values
    filename - in which to save the figure.
    """
    ind = np.arange(len(auc_ls))
    auc_ls = sorted(auc_ls,
                    key = lambda (name, val, color): val)
    with open(os.path.join(output_folder, "auc.dat"), 'w') as fout:
        fout.write('\t'.join(["system", "auc"])+"\n")
        for auc_ind, (name, val, color) in enumerate(auc_ls):
            fout.write("{}\t{}\n".format(name, val))

    names, vals, colors = zip(*auc_ls)
    logging.info("Plotting AUC chart to {}".format(filename))
    fig, ax = plt.subplots()
    width = 0.35
    rects1 = ax.bar(ind, vals, width, color = colors)
    for i, v in enumerate(vals):
            ax.text(i + .1, v + .01 , "{:.2f}".format(v))
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(names)
    plt.savefig(filename)


if __name__ == '__main__':
    args = docopt(__doc__)
    input_folder = args['--in']
    output_folder = args['--out'] \
                    if args['--out'] \
                       else input_folder
    filetype = args['--outputtype'] \
               if args['--outputtype'] \
                  else "png"

    # plot graphs for all *.dat files in input path
    fig = plt.figure()
    aucs = []
    pr_ls = []

    files = [os.path.join(input_folder,
                          fn.strip())
             for fn
             in open(os.path.join(input_folder, 'systems.txt'))]

    for _file in files:
        p, r = get_pr(_file)
        name = trend_name(_file)
        pr_ls.append((name, (p,r)))
        aucs.append((name, np.trapz(p, x = r))) # Record AUC

    colors = plot_pr_curve(pr_ls,
                           os.path.join(output_folder, "pr.{}".format(filetype)))

    plot_auc([(name, auc, color)
              for ((name, auc), color)
              in zip(aucs, colors)],
             os.path.join(output_folder, "auc.{}".format(filetype)),
             output_folder
    )

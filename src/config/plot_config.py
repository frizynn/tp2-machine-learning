import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


plt.style.use('seaborn-v0_8-whitegrid')


mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12

mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.linewidth'] = 0.5



mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['lines.markeredgewidth'] = 1.5

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.right'] = True

mpl.rcParams['legend.frameon'] = True
mpl.rcParams['legend.framealpha'] = 0.8
mpl.rcParams['legend.edgecolor'] = 'black'
mpl.rcParams['legend.fancybox'] = True

mpl.rcParams['axes.prop_cycle'] = cycler(color=[
    '#1f77b4',  
    '#ff7f0e',  
    '#2ca02c',  
    '#d62728',  
    '#9467bd',  
    '#8c564b',  
    '#e377c2',  
    '#7f7f7f',  
    '#bcbd22',  
    '#17becf'  
])

mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['figure.facecolor'] = '#f0f0f0'  
mpl.rcParams['figure.edgecolor'] = 'white'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.1

mpl.rcParams['figure.subplot.left'] = 0.1
mpl.rcParams['figure.subplot.right'] = 0.9
mpl.rcParams['figure.subplot.bottom'] = 0.1
mpl.rcParams['figure.subplot.top'] = 0.9
mpl.rcParams['figure.subplot.wspace'] = 0.2
mpl.rcParams['figure.subplot.hspace'] = 0.2

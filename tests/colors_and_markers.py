
import matplotlib as mpl
from collections import OrderedDict
import matplotlib.pyplot as plt

def make_like_colab(fig, ax):
    # use a gray background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#E6E6E6')

    # draw solid white grid lines
    ax.grid(color='white', linestyle='solid')

    # hide axis spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # hide top and right ticks
    # ax.xaxis.tick_bottom()
    # ax.yaxis.tick_left()

    # lighten ticks and labels
    # ax.tick_params(colors='gray', direction='out')
    # for tick in ax.get_xticklabels():
    #     tick.set_color('gray')
    # for tick in ax.get_yticklabels():
    #     tick.set_color('gray')

    # remove the notch on the ticks (but show the label)
    ax.tick_params(axis=u'both', which=u'both', length=0)

def export_legend(ax, filename, ncol=6):
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(*ax.get_legend_handles_labels(), frameon=False, loc='lower center', ncol=ncol)
    fig3  = legend.figure
    fig3.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    make_like_colab(fig3,ax2)
    fig3.savefig(filename, dpi="figure", bbox_inches=bbox)

names_dict = OrderedDict({
'superansac': 'SupeRANSAC',
'superansac (GaU)': 'SupeRANSAC',
'superansac (MAGSAC)': 'SupeRANSAC',
'superansac (MSAC)': 'SupeRANSAC',

'RANSAC OpenCV': 'RANSAC [OpenCV]',
'skimage': 'RANSAC [skimage]',
'RHO OpenCV': 'RHO [OpenCV]',


'kornia github': 'LO-RANSAC [kornia]',
'kornia': 'LO-RANSAC [kornia-GPU]',
'pydegensac github': 'LO$^{+}$-RANSAC \\ [pydeg]',
'pydegensac-lafcheck github': 'LO$^{+}$-RANSAC-LAF \\ [pydeg]',

'GC-RANSAC github': 'GC-RANSAC [Author]',
'GCRANSAC OpenCV': 'GC-RANSAC [OpenCV]',
'GC-RANSAC-PROSAC github': 'GC-RANSAC-PROSAC [Author]',

'MAGSAC github': 'MAGSAC [Author]',
'MAGSAC OpenCV': 'MAGSAC++ [OpenCV]',
'MAGSAC++ github': 'MAGSAC++ [Author]',

'poselib': 'LO-RANSAC [PoseLib]',


'vsac': 'VSAC [Author]',
'VSAC-PROSAC github': 'VSAC-PROSAC [Author]',


'LSQ OpenCV': 'LSQ [OpenCV]',
'LMEDS OpenCV': 'LMEDS [OpenCV]',
'pycolmap': 'LO-RANSAC [PyCOLMAP]',
    
'LSQ-on-GT-inliers': 'LSQ on GT inliers'
})




markers_dict = { 
    'superansac': 'o',
    'superansac (GaU)': 'o',
    'superansac (MAGSAC)': 'o',
    'superansac (MSAC)': 'o',
    
    'RANSAC OpenCV': '<',
    'skimage': r'$\circledcirc$',
    'RHO OpenCV': r'$\odot$',


    'kornia github': 'o',
    'kornia': 'o',
    'pydegensac github': 's',
    'pydegensac-lafcheck github': 'o',

    'GC-RANSAC github': "^",
    'GCRANSAC OpenCV': "<",
    'GC-RANSAC-PROSAC github': "^",
    
    'poselib': 'h',
    'ac-ransac github SIFT': '<',
    'ac-ransac github ORB': 'v',

    'MAGSAC github':'>',
    'MAGSAC OpenCV': 'o',
    'MAGSAC++ github': 'v',

    'vsac': 'P',
    'VSAC-PROSAC github': 'P',

    'LSQ OpenCV':'o',
    'LMEDS OpenCV': 'X',
    'pycolmap': 'D',
    
    'LSQ-on-GT-inliers': 'o',
}

colormap_dict = {
    "RANSAC OpenCV": mpl.cm.get_cmap('tab10')(0),
    "skimage": mpl.cm.get_cmap('tab10')(0),
    "RHO OpenCV": mpl.cm.get_cmap('tab10')(0),
    
    "kornia github":mpl.cm.get_cmap('tab10')(2),
    "kornia":mpl.cm.get_cmap('tab10')(2),
    
    "pydegensac github": mpl.cm.get_cmap('tab10')(2),
    "pydegensac-lafcheck github":  mpl.cm.get_cmap('tab10')(2),
    
    "GC-RANSAC github":  mpl.cm.get_cmap('tab10')(1),
    "GCRANSAC OpenCV":  mpl.cm.get_cmap('tab10')(1),
    "GC-RANSAC-PROSAC github":  mpl.cm.get_cmap('tab10')(1),
  
    'poselib': mpl.cm.get_cmap('tab10')(2),
    'ac-ransac github SIFT': mpl.cm.get_cmap('tab10')(5),
    'ac-ransac github ORB': mpl.cm.get_cmap('tab10')(5),
    
    "MAGSAC github":  mpl.cm.get_cmap('tab10')(6),
    "MAGSAC OpenCV":  mpl.cm.get_cmap('tab10')(7),
    "MAGSAC++ github":  mpl.cm.get_cmap('tab10')(4),
    
    "vsac":  mpl.cm.get_cmap('tab10')(9) ,
    "VSAC-PROSAC github": mpl.cm.get_cmap('tab10')(1),
    
    "LSQ OpenCV":  mpl.cm.get_cmap('tab10')(7),
    "LMEDS OpenCV":  mpl.cm.get_cmap('tab10')(8),
    "pycolmap": mpl.cm.get_cmap('tab10')(5),
    
    "LSQ-on-GT-inliers": mpl.cm.get_cmap('tab10')(7),
    
    'superansac': mpl.cm.get_cmap('tab10')(3),
    'superansac (MAGSAC)': mpl.cm.get_cmap('tab10')(3),
    'superansac (MSAC)': mpl.cm.get_cmap('tab10')(3)
}

colormap_dict_sampler = {
    "Uniform": mpl.cm.get_cmap('tab10')(0),
    "PROSAC": mpl.cm.get_cmap('tab10')(1),
    "PNAPSAC": mpl.cm.get_cmap('tab10')(2),
    "Importance": mpl.cm.get_cmap('tab10')(3),
    "ARSampler": mpl.cm.get_cmap('tab10')(4)
}

markers_dict_scoring = {
    'RANSAC': "^",
    'MSAC': "*",
    'MAGSAC': "D",
    'ACRANSAC': 'o',
}

colormap_dict_scoring = {
    "RANSAC": mpl.cm.get_cmap('tab10')(0),
    "MSAC": mpl.cm.get_cmap('tab10')(1),
    "MAGSAC": mpl.cm.get_cmap('tab10')(2),
    "ACRANSAC": mpl.cm.get_cmap('tab10')(3),
    "GaU": mpl.cm.get_cmap('tab10')(4),
    "ML": mpl.cm.get_cmap('tab10')(5)
}
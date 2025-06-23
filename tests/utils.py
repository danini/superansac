import numpy as np
import h5py

from PIL import Image
import torch
import time
from functions import point_matching, normalize_keypoints
from lightglue import LightGlue, SuperPoint
from romatch import roma_outdoor

def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    try:
        #with self.lock:
        with h5py.File(filename, 'r') as f:
            keys = [key for key in f.keys()]
            for key in keys:
                dict_to_load[key] = f[key][()]
    except:
        print('Cannot find file {}'.format(filename))
    return dict_to_load

def append_h5(dict_to_save, filename, replace=False):
    '''Saves dictionary to HDF5 file'''

    with h5py.File(filename, 'a') as f:
        #with self.lock:
        for key in dict_to_save:
            if replace and key in f.keys():
                del f[key]
            f.create_dataset(key, data=dict_to_save[key])

def read_h5(key, filename):
    '''Saves dictionary to HDF5 file'''

    with h5py.File(filename, 'a') as f:
        #with self.lock:
        if key in f.keys():
            return np.array(f.get(key))
        else:
            return None
        

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from multiprocessing import Pool as ThreadPool 

def tocuda(data):
    # convert tensor data in dictionary to cuda when it is a tensor
    for key in data.keys():
        if type(data[key]) == torch.Tensor:
            data[key] = data[key].cuda()
    return data

def get_pool_result(num_processor, fun, args):
    pool = ThreadPool(num_processor)
    pool_res = pool.map(fun, args)
    pool.close()
    pool.join()
    return pool_res

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M
    
def torch_skew_symmetric(v):

    zero = torch.zeros_like(v[:, 0])

    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)

    return M

def get_sequences(is_test):
    # ordered list of sequences
    if is_test:
        return {
            'british_museum': 'BM',
            'florence_cathedral_side': 'FCS',
            'lincoln_memorial_statue': 'LMS',
            'london_bridge': 'LB',
            'milan_cathedral': 'MC',
            'mount_rushmore': 'MR',
            'piazza_san_marco': 'PSM',
            'reichstag': 'RS',
            'sagrada_familia': 'SF',
            'st_pauls_cathedral': 'SPC',
            'united_states_capitol': 'USC',
        }
    else:
        return {
            'sacre_coeur': 'SC',
            'st_peters_square': 'SPS',
        }


def convert_bagsize_key(bagsize):
    if bagsize == 'bag3':
        return 'subset: 3 images'
    elif bagsize == 'bag5':
        return 'subset: 5 images'
    elif bagsize == 'bag10':
        return 'subset: 10 images'
    elif bagsize == 'bag25':
        return 'subset: 25 images'
    elif bagsize == 'bags_avg':
        return 'averaged over subsets'
    else:
        raise ValueError('Unknown bag size')


def parse_json(filename, verbose=False):
    with open(filename, 'r') as f:
        if verbose:
            print('Parsing "{}"'.format(filename))
        return json.load(f)


def get_plot_defaults():
    return {
        'font_size_title': 32,
        'font_size_axes': 25,
        'font_size_ticks': 22,
        'font_size_legend': 10,
        'line_width': 3,
        'marker_size': 10,
        'dpi': 600,
    }

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

def color_picker_features(label):
    cmap = matplotlib.cm.get_cmap('tab20')
#     cmap = matplotlib.cm.get_cmap('jet')

    # list position determines color: move them around if there are conflicts between similar colors
    # the legend will be sorted alphabetically
    known = [
        'CV-SIFT',
        'CV-RootSIFT',
        'CV-AKAZE',
        'CV-ORB',
        'CV-SURF',
        'CV-FREAK',
#         'VL-DoG-SIFT',
#         'VL-DoGAff-SIFT',
#         'VL-Hess-SIFT',
#         'VL-HessAffNet-SIFT',
        'ContextDesc',
        'D2-Net (SS)',
        'D2-Net (MS)',
        'SuperPoint',
        'LF-Net',
        'LogPolarDesc',
        'DoG-HardNet',
        'DoG-SOSNet',
        'GeoDesc',
        'Key.Net-HardNet',
        'Key.Net-SOSNet',
        'L2-Net',
#         'DSP-SIFT',
#         'Colmap-SIFT',
        'R2D2 (waf-n16)',
        'R2D2 (wasf-n16)',
        'R2D2 (wasf-n8-big)',
        'R2D2 (best model)',
    ]

#     if label == 'Key.Net/HardNet':
#         label = 'KeyNet/HardNet'
    for i, s in enumerate(known):
        if s.lower() in label.lower():
            if i >=23:
                return (.25, .25, .25, 1)
            else:
                 return cmap(i)
#             return cmap(np.linspace(0, 1.0, len(known)))[i]

    raise ValueError('Could not find color for method "{}"- > please see utils.py'.format(label))

#     if len(known


# run sanity checks on a (loaded) results file
def result_is_valid(res, num_features, task, matcher, use_flann, use_cne=None):
    errors = []
    if task not in ['stereo', 'multiview', 'both']:
         errors += ['Incorrect value for task ("{}")'.format(task)]

    # 1. no USC or RS
    if 'united_states_capitol' in res['phototourism']['results']:
        errors += ['Mismatch: Found USC!']
    if 'reichstag' in res['phototourism']['results']:
        errors += ['Mismatch: Found RS!']

    # 2. no 3bag
    if task in ['multiview', 'both']:
        for scene in res['phototourism']['results']:
            if '3bag' in res['phototourism']['results'][scene]['multiview']['run_avg']:
                errors += ['Mismatch: Found 3bag (scene: "{}"")!'.format(scene)]

    # 3. both/either
    if task in ['stereo', 'both']:
        if res['config']['config_phototourism_stereo']['matcher']['symmetric']['reduce'] != matcher:
            errors += ['Mismatch: matching strategy']
    elif task in ['multiview', 'both']:
        if res['config']['config_phototourism_multiview']['matcher']['symmetric']['reduce'] != matcher:
            errors += ['Mismatch: FLANN']

    # 4. flann
    if task in ['stereo', 'both']:
        if res['config']['config_phototourism_stereo']['matcher']['flann'] != use_flann:
            errors += ['Mismatch: FLANN']
    elif task in ['multiview', 'both']:
        if res['config']['config_phototourism_multiview']['matcher']['flann'] != use_flann:
            errors += ['Mismatch: FLANN']

    # 5. number of features
    if res['config']['config_common']['num_keypoints'] != num_features:
        errors += ['Mismatch: number of features']

    # 6. CNe
    # None to ignore, True/False to check
    if use_cne is not None:
        if task in ['stereo', 'both']:
            is_using_cne = res['config']['config_phototourism_stereo']['outlier_filter']['method'].lower() == 'cne-bp-nd'
            if is_using_cne != use_cne:
                errors += ['Mismatch: CNe (stereo)']
        elif task in ['multiview', 'both']:
            is_using_cne = res['config']['config_phototourism_multiview']['outlier_filter']['method'].lower() == 'cne-bp-nd'
            if is_using_cne != use_cne:
                errors += ['Mismatch: CNe (multiview)']

    if errors:
        return False, '\n'.join(errors)
    else:
        return True, ''

def detect_and_load_data(data, args, detector, matcher):
    # Database labels
    label1 = "-".join(data["id1"].split("/")[-3:])
    label2 = "-".join(data["id2"].split("/")[-3:])

    # Try loading the point matches from the database file
    matches = read_h5(f"{args.features.lower()}-{label1}-{label2}", args.output_db_path)
    scores = read_h5(f"{args.features.lower()}-{label1}-{label2}-scores", args.output_db_path)

    if matches is not None and scores is not None:
        return matches, scores

    if args.features == "splg":
        img1 = data["img1"]
        img2 = data["img2"]

        start_time = time.time()
        # Detect keypoints by SuperPoint + SuperGlue, LoFTR, or GlueStick
        matches, scores = point_matching(img1, img2, args.features, detector, matcher, args.device)
        # Saving to the database
        append_h5({f"{args.features.lower()}-{label1}-{label2}": matches}, args.output_db_path)
        append_h5({f"{args.features.lower()}-{label1}-{label2}-scores": scores}, args.output_db_path)
        end_time = time.time()
        print(f"Point matching took {end_time - start_time:.2f} seconds")

    elif args.features == "RoMA":
        start_time = time.time()
        # Match
        warp, certainty = detector.match(data['path1'], data['path2'], device = args.device)
        # Sample matches for estimation
        matches, certainty = detector.sample(warp, certainty)
        # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
        W_A, H_A = Image.open(data['path1']).size
        W_B, H_B = Image.open(data['path2']).size
        kptsA, kptsB = detector.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

        matches = np.hstack((kptsA.cpu().numpy(), kptsB.cpu().numpy()))
        scores = np.ones(matches.shape[0])

        # Saving to the database
        append_h5({f"{args.features.lower()}-{label1}-{label2}": matches}, args.output_db_path, replace=False)
        append_h5({f"{args.features.lower()}-{label1}-{label2}-scores": scores}, args.output_db_path, replace=False)
        end_time = time.time()
        print(f"Point matching took {end_time - start_time:.2f} seconds")
    else:
        raise NotImplementedError
    
    return matches, scores
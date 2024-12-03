## Installation

Clone the repository and its submodules:
```
git clone git@github.com:danini/superansac.git
```

Make sure that you have the necessary OpenCV libraries installed:
```
sudo apt-get update
sudo apt-get install libopencv-dev libopencv-contrib-dev libarpack++2-dev libarpack2-dev libsuperlu-dev cmake build-essential libboost-all-dev libeigen3-dev
```

Install SupeRANSAC by running 
```
pip install .
```

## Evaluation - Python

The testing scripts for SupeRANSAC and the baselines are located in folder `tests/`.
The results for essential, fundamental, and homography matrix estimation are obtained by running
```
python tester-X-superansac.py
python tester-X-baseline.py
```
where X is E/F/H. Note that this process may take a while. 
However, before running, you need to install the feature detectors and set up the datasets as described below. 
Also, you need to install the baselines you want to test. 

### Install feature detectors

To use SuperPoint + LightGlue features, install LightGlue as follows:
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

To use RoMA features, install it as follows:
```
git clone https://github.com/Parskatt/RoMa && cd RoMa
pip install -e .
```

### Evaluation on the PhotoTourism dataset
Download the data from the CVPR tutorial "RANSAC in 2020":
```
wget http://cmp.felk.cvut.cz/~mishkdmy/CVPR-RANSAC-Tutorial-2020/RANSAC-Tutorial-Data-EF.tar
tar -xf  RANSAC-Tutorial-Data-EF.tar
```

Then run the notebook `examples/relative_pose_evaluation_phototourism.ipynb`.

### Evaluation on the ScanNet dataset
Download the data from the test set for relative pose estimation used in SuperGlue (~250Mb for 1500 image pairs only):
```
wget https://www.polybox.ethz.ch/index.php/s/lAZyxm62WUh27Zl/download
unzip ScanNet_test.zip -d <path to extract the ScanNet test set>
```

Then run the notebook `examples/relative_pose_evaluation_scannet.ipynb`.


### Evaluation on the 7Scenes dataset
Download the [7Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and put it where it suits you. You can also only download one scene and later specify this scene in the dataloader constructor.

Then run the script `runners/run_7scenes.py`.


### Evaluation on the ETH3D dataset
Download the [ETH3D dataset](https://www.eth3d.net/datasets) (training split of the high-res multi-view, undistorted images + GT extrinsics & intrinsics should be enough) and put it where it suits you. The input argument 'downsize_factor' can be used to downscale the images, because they can be quite large otherwise.

Then run the script `runners/run_eth3d.py`.


### Evaluation on the LaMAR dataset
Download the [CAB scene of the LaMAR dataset](https://cvg-data.inf.ethz.ch/lamar/CAB.zip), and unzip it to your favourite location. Note that we only use the images in `CAB/sessions/query_val_hololens`.


### Evaluation on the KITTI dataset
Download the [KITTI odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale images and poses), and unzip them to your favourite location.
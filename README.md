# SupeRANSAC ([Paper](https://arxiv.org/pdf/2407.20219))

## About

SupeRANSAC is a Python lbriary that provides bindings for an advanced RANSAC C++ implementation using pybind11. 
It supports a wide variety of sampling, scoring, local optimization, and inlier selection techniques for robust model estimation tasks.
It provides estimators for homography, essential, fundamental matrix, rigid and absolute pose estimation.

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

# Jupyter Notebook Examples

- **Homography Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_homography_fitting_splg.ipynb)

- **Homography Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_homography_fitting_roma.ipynb)

- **Fundamental Matrix Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_fundamental_matrix_fitting_splg.ipynb)

- **Fundamental Matrix Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_fundamental_matrix_fitting_roma.ipynb)

- **Essential Matrix Fitting with SuperPoint + LightGlue Matches**  
  Example available at: [notebook](examples/example_essential_matrix_fitting_splg.ipynb)

- **Essential Matrix Fitting with RoMA Matches**  
  Example available at: [notebook](examples/example_essential_matrix_fitting_roma.ipynb)


## Usage

The library provides several model estimation functions and settings that allow customization of the RANSAC pipeline. Below are the primary features and their usage.

### Importing the Library

```python
import superansac
from superansac import ScoringType, SamplerType, LocalOptimizationType, InlierSelectorType, NeighborhoodType, CameraType, RANSACSettings
```

### Example: Estimate a Homography Matrix

```python
# Example correspondences and image sizes
correspondences = [
    ([x1, y1], [x2, y2]),
    ([x3, y3], [x4, y4]),
    # Add more correspondences...
]
image_sizes = [(width1, height1), (width2, height2)]

# Configure RANSAC settings
config = RANSACSettings()
config.min_iterations = 100
config.max_iterations = 1000
config.inlier_threshold = 3.0
config.confidence = 0.99
config.scoring = ScoringType.MAGSAC
config.sampler = SamplerType.PROSAC
config.local_optimization = LocalOptimizationType.GCRANSAC
config.final_optimization = LocalOptimizationType.LSQ

# Estimate homography
result = superansac.estimateHomography(correspondences, image_sizes, config=config)
print("Estimated Homography Matrix:", result) 
```

### Supported Estimation Functions

#### 1. Homography Estimation
- **Function**: `superansac.estimateHomography`
- **Description**: Estimates a homography matrix from 2D-2D point correspondences.
- **Parameters**:
  - `correspondences`: A list of paired 2D points.
  - `image_sizes`: A tuple of source and target image sizes.
  - `config`: An instance of `RANSACSettings`.
  - `probabilities` *(optional)*: Correspondence probabilities.

#### 2. Fundamental Matrix Estimation
- **Function**: `superansac.estimateFundamentalMatrix`
- **Description**: Estimates a fundamental matrix from 2D-2D point correspondences.
- **Parameters**:
  - `correspondences`: A list of paired 2D points.
  - `image_sizes`: A tuple of source and target image sizes.
  - `config`: An instance of `RANSACSettings`.
  - `probabilities` *(optional)*: Correspondence probabilities.

#### 3. Essential Matrix Estimation
- **Function**: `superansac.estimateEssentialMatrix`
- **Description**: Estimates an essential matrix using 2D-2D point correspondences and intrinsic matrices.
- **Parameters**:
  - `correspondences`: A list of paired 2D points.
  - `image_sizes`: A tuple of source and target image sizes.
  - `config`: An instance of `RANSACSettings`.
  - `intrinsics_src`: Source camera intrinsic matrix.
  - `intrinsics_dst`: Destination camera intrinsic matrix.
  - `probabilities` *(optional)*: Correspondence probabilities.

#### 4. Rigid Transformation Estimation
- **Function**: `superansac.estimateRigidTransform`
- **Description**: Estimates a 6D rigid transformation matrix from 3D-3D point correspondences.
- **Parameters**:
  - `correspondences`: A list of paired 3D points.
  - `bounding_box_sizes`: Size of the bounding box.
  - `config`: An instance of `RANSACSettings`.
  - `probabilities` *(optional)*: Correspondence probabilities.

#### 5. Absolute Pose Estimation
- **Function**: `superansac.estimateAbsolutePose`
- **Description**: Estimates the absolute pose of a camera using 2D-3D correspondences.
- **Parameters**:
  - `correspondences`: A list of paired 2D-3D points.
  - `camera_type`: Type of the camera (e.g., `CameraType.SimpleRadial`).
  - `camera_params`: Camera parameters.
  - `config`: An instance of `RANSACSettings`.
  - `probabilities` *(optional)*: Correspondence probabilities.


### Advanced Configuration: RANSACSettings

The ```RANSACSettings``` class provides fine-grained control over the RANSAC pipeline. Customize it as needed:

```
config = RANSACSettings()
config.min_iterations = 100
config.max_iterations = 1000
config.inlier_threshold = 3.0
config.confidence = 0.99
config.scoring = ScoringType.MAGSAC
config.sampler = SamplerType.PROSAC
config.local_optimization = LocalOptimizationType.LSQ
config.neighborhood = NeighborhoodType.Grid
```

### Enumerations and Their Values

The library supports several enumeration types to customize sampling, scoring, and other pipeline components:

**Scoring Types**
- ScoringType.RANSAC
- ScoringType.MSAC
- ScoringType.MAGSAC
- ScoringType.ACRANSAC

**Sampler Types**
- SamplerType.Uniform
- SamplerType.PROSAC
- SamplerType.NAPSAC
- SamplerType.ProgressiveNAPSAC
- SamplerType.ImportanceSampler
- SamplerType.ARSampler

**Local Optimization Types**
- LocalOptimizationType.Nothing
- LocalOptimizationType.LSQ
- LocalOptimizationType.IteratedLSQ
- LocalOptimizationType.NestedRANSAC
- LocalOptimizationType.GCRANSAC

**Neighborhood Types**
- NeighborhoodType.Grid
- NeighborhoodType.BruteForce

**Camera Types**
- CameraType.SimpleRadial
- CameraType.SimplePinhole

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
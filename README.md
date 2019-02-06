## Parallax CPP

This library uses a novel approach at estimating the essential matrix. 
The essential matrix is then used to determine which 3D world points are actually moving.
For more information see the wiki.

### Install

This library is build to be installed onto your machine. To install it, clone the
repository, and navigate to the directory.

``` bash
mkdir build
cd build
cmake ..
make
sudo make install


```

If you have multiple versions of OpenCV on your system, you can specify which version to use by changing the `OpenCV_DIR` tag. For example

``` bash
mkdir build
cd build
cmake -DOpenCV_DIR=/usr/local/share/OpenCV ..
make
sudo make install
```

### How to use in your own project

```c++

// Include the parallax detector library
#include "parallax_detector/parallax_detector.h"

// Declare a Parallax Detector Object
gnsac::ParallaxDetector  parallax_detec_;

// Class object that stores the essential matrix, rotation matrix, translation vector and others. See below for more information
common::EHypothesis results_

// Initialize the Parallax Detector Object 
// @param gnsac_solver_filename The absolute file path to the file with all of the gnsac parameters. If 
//                              the file name is empty, default parameters will be used.
parallax_initiated_ = parallax_detec_.Init(gnsac_solver_filename);

// Run the parallax compensation algorithm which computes and returns essential matrix, rotation matrix, and translation vector.
// @param ud_prev_matched_ Calibrated image points in the previous frame
// @param ud_curr_matched_ Corresponding calibrated image points in the current frame
// @param moving_parallax_ Output vector of bools corresponding to each matched points. If moving_parallax_[i] is true, then point
//                         ud_curr_matched_[i] is considered moving perpendicular to the epipolar lines and has velocity in the world frame.
results_ parallax_detec_.ParallaxCompensation(ud_prev_matched_, ud_curr_matched_, moving_parallax_);


```
### common::EHypothesis
```c++
class EHypothesis
{
public:
	EHypothesis();
	EHypothesis(Eigen::Matrix3d& E);
	EHypothesis(Eigen::Matrix3d& E, Eigen::Matrix3d& R, Eigen::Vector3d& t);
	Eigen::Matrix3d E;            // Essential Matrix
	Eigen::Matrix3d R;            // Rotation Matrix
	Eigen::Vector3d t;            // Translation Vector
	double cost;                  // Resulting Error from the optimization algorithm used to compute the essential matrix
	bool has_RT;                  // If true, R and t have been calculated
};


```

### Parameters

See the wiki
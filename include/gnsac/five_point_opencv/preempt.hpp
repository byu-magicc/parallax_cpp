#ifndef __PREEMPT__H__
#define __PREEMPT__H__

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <vector>
#include <time.h>
#include <map>
#include "five_point_opencv/calib3d.hpp"

// For this flexible function, we assume that the points are already in the normalized image plane. No undistorting points necessary!
cv::Mat findEssentialMatPreempt2(std::vector<cv::Point2f> pts1, std::vector<cv::Point2f> pts2, float threshold, int niters, int blocksize, std::string method = "");

#endif

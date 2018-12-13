#ifndef PARALLAX_DETECTOR_H
#define PARALLAX_DETECTOR_H

#include <eigen3/Eigen/Dense>
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/core.hpp"
#include <vector>
#include <string>
#include <experimental/filesystem>

// GNSAC
#include "common/solvers.h" 
#include "common/common.h"

namespace fs = std::experimental::filesystem;

namespace gnsac {


class ParallaxDetector {

public:

  ParallaxDetector()=default;
  ~ParallaxDetector()=default;


  /**
  * \brief  Initializes the solver based on the parameter file.
  */
  bool Init(const std::string solver_param_filename);

  /**
  * \brief Identifies the matched features that have some velocity perpendicular to the epipolar lines.
  * \detail Computes the essential matrix using a five point algorithm. It then creates two velocity fields
  *         for each point: a paralax field and a perpendicual field. Then it measures the perpendicular velocity
  *         of each point. If this velocity is above a certain threshold, then it sets the flag for that point.    
  * @param prev_pts Matched undistorted points from the previous frame
  * @param curr_pts matched undistorted points from the current frame
  * @param moving_parallax sets a flag (true) for every matched point if the point has a velocity component perpendicular to the parallax field. 
  * @return Returns true if there are still more than 10 matched features. 
  */
  void ParallaxCompensation(const std::vector<cv::Point2f>& prev_pts, const  std::vector<cv::Point2f>& curr_pts, std::vector<bool>& moving_parallax);

  void SetParallaxThrehold(double parallax_threshold);

private:

  /**
  * \brief  Gives the parallax "field"
  * \detail Stores the parallax and perpendicular unit vectors at each point in the list of points
  *         for the given essential matrix.
  */
  void GetParallaxField(cv::Mat& E, const cv::Point2f& loc, cv::Point2f& perpendicular, cv::Point2f& parallel);

  /**
  * \brief  Thresholds the given points
  * \detail Sets the moving flag if the perpendicular velocity is greater than the parallax_threshold_.
  */
  void ThresholdVelocities(cv::Mat& E, cv::Mat& R, const std::vector<cv::Point2f>& imagePts1, const std::vector<cv::Point2f>& imagePts2, 
                           std::vector<cv::Point2f>& pointVelocities, std::vector<cv::Point2f>& velRotated, std::vector<bool>& moving);



  double parallax_threshold_;  /**< Minimum Perpendicular (to epipolar line) Velocity */

  std::shared_ptr<common::ESolver> solver_;  /**< GNSAC Solver - optimizes the Essential Matrix on the SO(2) & SO(3) manifold,
                                                    rejects outliers using LMEDS or RANSAC  */

};

}

#endif //PARALLAX_DETECTOR_H
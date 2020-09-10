#include "parallax_detector/parallax_detector.h"


namespace gnsac {

bool ParallaxDetector::Init(const std::string solver_param_filename)
{

  bool initiated = true;

  // if(!fs::exists(solver_param_filename))
  // {
  //   throw common::Exception("File \"" + solver_param_filename + "\" does not exist.");
  //   initiated = false;
  // }
  solver_ = common::ESolver::from_yaml(solver_param_filename);

  return initiated;
}

// ----------------------------------------------------------------------------

common::EHypothesis ParallaxDetector::ParallaxCompensation(const std::vector<cv::Point2f>& prev_pts, const  std::vector<cv::Point2f>& curr_pts,std::vector<bool>& moving_parallax)
{
  // Convert points to eigen
  int n_pts = curr_pts.size();
  common::scan_t pts1_eig = common::scan_t(n_pts);
  common::scan_t pts2_eig = common::scan_t(n_pts);
  for (int i = 0; i < n_pts; i++)
  {
    pts1_eig[i] << prev_pts[i].x, prev_pts[i].y;
    pts2_eig[i] << curr_pts[i].x, curr_pts[i].y;
  }

  // Calculate essential Matrix
  cv::Mat E, R;
  common::EHypothesis result;
  solver_->find_best_hypothesis(pts1_eig, pts2_eig, Eigen::Matrix4d::Identity(), result);
  eigen2cv(result.E, E);
  if(result.has_RT)          // The Five Point OpenCV solver doesn't compute R and T
    eigen2cv(result.R, R);
  else
  {
    // Calculate R and T
    // Not sure why, but the cheirality check only works part of the time, even when we
    // filter essential Matrix outliers. It is much more robust to compare the traces
    // of the Matrices, since the larger the trace, the smaller the rotation angle.
    cv::Mat R1, R2, T;
    cv::decomposeEssentialMat(E, R1, R2, T);
    R = (cv::trace(R1)[0] > cv::trace(R2)[0]) ? R1 : R2;
  }

  // Threshold velocities based on R and E.
  std::vector<cv::Point2f> point_velocities;
  std::vector<cv::Point2f> vel_rotated;
  ThresholdVelocities(E, R, prev_pts, curr_pts, point_velocities, vel_rotated, moving_parallax);

  return result;
}

// ----------------------------------------------------------------------------


void ParallaxDetector::SetParallaxThreshold(double parallax_threshold)
{

  parallax_threshold_ = parallax_threshold;

}

// ----------------------------------------------------------------------------

void ParallaxDetector::SetErrorType(const ErrorType type)
{
  et_ = type;
}

// ----------------------------------------------------------------------------

void ParallaxDetector::GetParallaxField(cv::Mat& E, const cv::Point2f& loc, cv::Point2f& perpendicular, cv::Point2f& parallel)
{
  cv::Mat line = E * (cv::Mat_<double>(3, 1) << loc.x, loc.y, 1);
  perpendicular = cv::Point2f(line.at<double>(0), line.at<double>(1));  // Mark:: This line is parallel to the epipolar line. 
  perpendicular = perpendicular / sqrt(perpendicular.x * perpendicular.x + perpendicular.y * perpendicular.y);
  parallel = cv::Point2f(perpendicular.y, -perpendicular.x);            // Mark:: This line is perpendicular to the epipolar line.

}

// ----------------------------------------------------------------------------

// void ParallaxDetector::ThresholdVelocities(cv::Mat& E, cv::Mat& R, const std::vector<cv::Point2f>& imagePts1, const std::vector<cv::Point2f>& imagePts2, 
//                                          std::vector<cv::Point2f>& pointVelocities, std::vector<cv::Point2f>& velRotated, std::vector<bool>& moving)
// {
//   // Use rotation as euclidian homography Matrix to transform points to 2nd frame
//   // H_e = R + T_x*n
//   cv::Mat H_e = R;
//   std::vector<cv::Point2f> imagePts1_warped;
//   cv::perspectiveTransform(imagePts1, imagePts1_warped, H_e); // Mark:: why arn't we using the essential matrix in this transform to account for 
//                                                               // rotational and translational changes?

//   // cv::perspectiveTransform(imagePts1, imagePts1_warped, E); // Mark:: why arn't we using the essential matrix in this transform to account for 
//   //                                                             // rotational and translational changes?

//   // Calculate the point velocities (velocity = parallax + actual velocity)
//   pointVelocities = std::vector<cv::Point2f>(imagePts1.size());
//   for (int i = 0; i < imagePts1.size(); i++)
//     pointVelocities[i] = imagePts2[i] - imagePts1_warped[i];

//   // Use the essential Matrix to calculate the field normal and parallel vectors
//   std::vector<cv::Point2f> fieldPerpendicular = std::vector<cv::Point2f>(imagePts1.size());
//   std::vector<cv::Point2f> fieldParallel = std::vector<cv::Point2f>(imagePts1.size());
//   for (int i = 0; i < imagePts1.size(); i++)
//     GetParallaxField(E, imagePts1[i], fieldPerpendicular[i], fieldParallel[i]);

//   // Use the dot product to find the directionality of the essential Matrix field
//   // The essential Matrix will often flip back and forth between directions, so
//   // it's important to verify which direction is correct.
//   int numPosDotProduct = 0;
//   int numNegDotProduct = 0;
//   for (int i = 0; i < imagePts1.size(); i++)
//   {
//     if (pointVelocities[i].dot(fieldParallel[i]) > 0)
//       numPosDotProduct++;
//     else
//       numNegDotProduct++;
//   }
//   int multiplier = (numPosDotProduct > numNegDotProduct) ? 1 : -1;

//   // Use the dot product to find parallel and perpendicular components of the velocity vectors
//   velRotated = std::vector<cv::Point2f>(imagePts1.size());
//   for (int i = 0; i < imagePts1.size(); i++)
//   {
//     double velParallel = pointVelocities[i].dot(fieldParallel[i]);
//     double velPerpendicular = pointVelocities[i].dot(fieldPerpendicular[i]);
//     velRotated[i] = cv::Point2f(velPerpendicular, velParallel)*multiplier;
//     // std::cout << velRotated[i] << std::endl;
//   }

//   // Determine which points are moving
//   moving = std::vector<bool>(imagePts1.size());
//   for (int i = 0; i < imagePts1.size() && i < imagePts1.size(); i++)
//     moving[i] = fabs(velRotated[i].x) > parallax_threshold_ || velRotated[i].y < -parallax_threshold_;

// }

void ParallaxDetector::ThresholdVelocities(cv::Mat& E, cv::Mat& R, const std::vector<cv::Point2f>& imagePts1, const std::vector<cv::Point2f>& imagePts2, 
                                         std::vector<cv::Point2f>& pointVelocities, std::vector<cv::Point2f>& velRotated, std::vector<bool>& moving)

{  
  std::vector<bool> moving_temp;
  for (unsigned i = 0; i < imagePts1.size(); i++)
  {

      cv::Mat_<double> x1(3/*rows*/,1 /* cols */); 

      x1(0,0)=imagePts1[i].x; 
      x1(1,0)=imagePts1[i].y; 
      x1(2,0)=1.0; 

      cv::Mat_<double> x2(3,1);
      x2(0,0)=imagePts2[i].x; 
      x2(1,0)=imagePts2[i].y; 
      x2(2,0)=1.0; 

      float error = 10000;

      switch(et_)
      {
        case ALGEBRAIC:
        {
          error = AlgebraicError(x1,x2,E);
          break;
        }
        case SAMPSON:
        {
          error = SampsonError(x1,x2,E);
          break;
        }
        case SINGLE_IMAGE:
        {
          error = SingleImageError(x1,x2,E);
          break;
        }
      }

      bool is_moving = false;
      if(error > parallax_threshold_)
      {
        is_moving = true;
      }
      else
      {
        is_moving = false;
        if(i < 30)
        {
          // std::cout << "error: " << std::endl << x1 << std::endl << x2 << std::endl << error << std::endl;
        }
      }

      moving_temp.push_back(is_moving);

  }
  moving = moving_temp;
}

// ----------------------------------------------------------------------------

float ParallaxDetector::AlgebraicError(const cv::Mat& x1, const cv::Mat& x2, const cv::Mat& E)
{

  cv::Mat temp = x2.t()*E*x1;

  return fabs(temp.at<double>(0,0));
}

// ----------------------------------------------------------------------------

float ParallaxDetector::SingleImageError(const cv::Mat& x1, const cv::Mat& x2, const cv::Mat& E)
{

  cv::Mat ell2 = E*x1;
  float a2 = ell2.at<double>(0,0);
  float b2 = ell2.at<double>(1,0);

  cv::Mat temp = x2.t()*E*x1;

  return fabs(temp.at<double>(0,0)/sqrt(pow(a2,2)+ pow(b2,2)));

}

// ----------------------------------------------------------------------------

float ParallaxDetector::SampsonError(const cv::Mat& x1, const cv::Mat& x2, const cv::Mat& E)
{

  cv::Mat ell1 = (x2.t()*E).t();
  float a1 = ell1.at<double>(0,0);
  float b1 = ell1.at<double>(0,1);

  cv::Mat ell2 = E*x1;
  float a2 = ell2.at<double>(0,0);
  float b2 = ell2.at<double>(1,0);

  cv::Mat temp = x2.t()*E*x1;


  return fabs(temp.at<double>(0,0)/sqrt(pow(a1,2)+ pow(b1,2) + pow(a2,2)+ pow(b2,2)));

}

}
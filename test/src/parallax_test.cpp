// GNSAC
#include "parallax_detector/parallax_detector.h"
#include "common/solvers.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <yaml-cpp/yaml.h>
#include <experimental/filesystem>


namespace fs = std::experimental::filesystem;


const double kPI = 3.14159265;

// ----------------------------------------------------------------------------

// Generate matching points from two different time samples. 
// world_points1 is from the first time sample, and 
// world_points2 is from the second. Since some of the points
// will represent moving features, some of world_points2
// will will be offset from world_points1 points. 
void GenerateWorldPoints(int num_points, int& moving_velocity_final, int& parallax_final, std::vector<cv::Point3f>& world_points1, std::vector<cv::Point3f>& world_points2)
{
	// Create a uniform random distribution generator. This will be 
	// used to create features in the world frame. 
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> feature_dist(-20, 20);
	std::uniform_real_distribution<> velocity_dist(1, 20);
	std::uniform_real_distribution<> height_dist(0, 0.5);
	std::uniform_real_distribution<> parallax_dist(2, 5);

	// Clear history
	world_points1.clear();
	world_points2.clear();
	

	// Generate points in NED coordinates
	moving_velocity_final = num_points*0.1;
	parallax_final = num_points*0.2;
	float Ts = 1.0/15.0;                         // Time sample is 15 frames per second. 
	for(int i = 0; i < num_points; i++)
	{

		cv::Point3f p1(feature_dist(gen),feature_dist(gen),-height_dist(gen));
		cv::Point3f p2;

		// Make 10 percent of the points have moving velocity
		if(i < moving_velocity_final)
		{
			std::cout << "moving here" << std::endl;
			// Create random velocity in the x and y direction
			cv::Point3f v2(velocity_dist(gen)*Ts,velocity_dist(gen)*Ts,0);
			p2 = p1 + v2;

		}
		// Make 10 percent of the points be closer to the camera
		// to create parallax
		else if(i < parallax_final)
		{

			// Create parallax by adding height to the feature
			cv::Point3f par(0,0,-parallax_dist(gen));
			p1 = p1 +par;
			p2 = p1;

		}
		else 
		{
			p2=p1;
		}

		// p2 = p1;

		world_points1.push_back(p1);
		world_points2.push_back(p2);
}
}

// ----------------------------------------------------------------------------
// NED rotation matrix
cv::Mat EulerAnglesToRotationMatrix(double phi, double theta, double psi, bool degrees)
{

	// convert to radians
	if(degrees)
	{
		phi = phi*kPI/180.0;
		theta = theta*kPI/180.0;
		psi = psi*kPI/180.0;
	}

	cv::Mat R_x(3,3,CV_32F);
	R_x.at<float>(0,0) = 1;
	R_x.at<float>(0,1) = 0;
	R_x.at<float>(0,2) = 0;
	R_x.at<float>(1,0) = 0;
	R_x.at<float>(1,1) = cos(phi);
	R_x.at<float>(1,2) = sin(phi);
	R_x.at<float>(2,0) = 0;
	R_x.at<float>(2,1) = -sin(phi);
	R_x.at<float>(2,2) = cos(phi);

	cv::Mat R_y(3,3,CV_32F);
	R_y.at<float>(0,0) = cos(theta);
	R_y.at<float>(0,1) = 0;
	R_y.at<float>(0,2) = -sin(theta);
	R_y.at<float>(1,0) = 0;
	R_y.at<float>(1,1) = 1;
	R_y.at<float>(1,2) = 0;
	R_y.at<float>(2,0) = sin(theta);
	R_y.at<float>(2,1) = 0;
	R_y.at<float>(2,2) = cos(theta);

	cv::Mat R_z(3,3,CV_32F);
	R_z.at<float>(0,0) = cos(psi);
	R_z.at<float>(0,1) = sin(psi);
	R_z.at<float>(0,2) = 0;
	R_z.at<float>(1,0) = -sin(psi);
	R_z.at<float>(1,1) = cos(psi);
	R_z.at<float>(1,2) = 0;
	R_z.at<float>(2,0) = 0;
	R_z.at<float>(2,1) = 0;
	R_z.at<float>(2,2) = 1;

	return R_x*R_y*R_z;

}

// Test the parallax compensation algorithm
TEST(ParallaxTest, ProjectionTest) {
fs::path test_path = fs::path(__FILE__).remove_filename().parent_path();
std::cout << test_path << std::endl;
std::string param_filename = test_path / "param/test_param.yaml";


// Set the camera matrix to identity so we work in the normalized image plane. 
cv::Mat camera_matrix = cv::Mat::eye(3,3,CV_32F);

// Get the world points from two different time frames.
std::vector<cv::Point3f> world_points1, world_points2;
std::vector<cv::Point2f> image_points1, image_points2;
int num_points = 10;
int moving_velocity_final;
int parallax_final;
GenerateWorldPoints(num_points, moving_velocity_final, parallax_final, world_points1, world_points2);

// Allocate memory for the rotation vectors
cv::Mat vR1_wc1(3,1,CV_32F);    // Rotation from world to first camera frame.
cv::Mat vR2_c1c2(3,1,CV_32F);   // Rotation from first camera frame to second camera frame. 

// Generate initial rotation matarix
cv::Mat R1_wc1 = EulerAnglesToRotationMatrix(0,0,0, true);

// Convert the rotation matrix to Rodrigues vector
cv::Rodrigues(R1_wc1, vR1_wc1);

std::cout << "here0" << std::endl;

// create initial translation vector and rotate it to the camera frame. 
cv::Mat T_wc1_c1(3,1,CV_32F);
T_wc1_c1.at<float>(0) = 10;
T_wc1_c1.at<float>(1) = 0;
T_wc1_c1.at<float>(2) = 15;
std::cout << "T_wc1_c1: " << T_wc1_c1 << std::endl;
std::cout << "R1_wc1*T_wc1_c1: " << R1_wc1*T_wc1_c1 << std::endl;

T_wc1_c1 = R1_wc1*T_wc1_c1;

std::cout << "here1" << std::endl;

// Project the world points onto the image plane in the first camera frame
cv::projectPoints(world_points1, vR1_wc1, T_wc1_c1, camera_matrix, cv::Mat(), image_points1);

std::cout << "Printing image points 1" << std::endl;
for(int i = 0; i < image_points1.size(); i++)
{

	std::cout << image_points1[i] << std::endl;

}


// Generate intermediate rotation matrix
cv::Mat R1_c1c2 = EulerAnglesToRotationMatrix(1,1,1, true);

// Convert the rotation matrix to Rodrigues vector
cv::Rodrigues(R1_c1c2*R1_wc1, vR2_c1c2);

// Create translation vector from camera frame 1 to camera frame 2
cv::Mat T_c1c2_c2(3,1,CV_32F);
T_c1c2_c2.at<float>(0) = 0.5;
T_c1c2_c2.at<float>(1) = 0;
T_c1c2_c2.at<float>(2) = 0;

// T_c1c2_c2 = R1_c1c2*R1_wc1*T_c1c2_c2;

// Project the world points onto the image plane in the second frame
cv::projectPoints(world_points2, vR2_c1c2, T_c1c2_c2 + R1_c1c2*T_wc1_c1, camera_matrix, cv::Mat(), image_points2);

std::cout << "Printing image points 2" << std::endl;
for(int i = 0; i < image_points2.size(); i++)
{

	std::cout << image_points2[i] << std::endl;

}



// Create the solver
common::EHypothesis result;
std::vector<bool> is_moving;

gnsac::ParallaxDetector pd;
pd.Init(param_filename);       // Load param file
pd.SetParallaxThreshold(0.001);  // Set the threshold value
result = pd.ParallaxCompensation(image_points1,image_points2,is_moving);

std::cout << std::endl;
std::cout << "Printing values from gnsac." << std::endl;
std::cout << "E: " << std::endl << result.E << std::endl;
std::cout << "R: " << std::endl << result.R << std::endl;
std::cout << "T: " << std::endl << result.t << std::endl;



// Compose the true essential matrix
cv::Mat T_c1c2_c2_normalized(3,1,CV_32F);
cv::Mat temp = cv::Mat::zeros(3,3,CV_32F);
temp.at<float>(0,1) = -T_c1c2_c2.at<float>(2,0);
temp.at<float>(0,2) = T_c1c2_c2.at<float>(1,0);
temp.at<float>(1,0) = T_c1c2_c2.at<float>(2,0);
temp.at<float>(1,2) = -T_c1c2_c2.at<float>(0,0);
temp.at<float>(2,0) = T_c1c2_c2.at<float>(1,0);
temp.at<float>(2,1) = T_c1c2_c2.at<float>(0,0);
cv::normalize(temp,temp);
cv::normalize(T_c1c2_c2, T_c1c2_c2_normalized);


cv::Mat E_cv, R1,R2, t,R_cv;
E_cv = cv::findEssentialMat(image_points1,image_points2, 1,cv::Point2d(0,0), cv::RANSAC, 0.999, 1e-4);
cv::decomposeEssentialMat(E_cv, R1,R2,t);

if(cv::trace(R1).val > cv::trace(R2).val)
{
	R_cv = R1;
} else {
	R_cv = R2;
}

std::cout << std::endl;
std::cout << "Printing opencv values" << std::endl;
std::cout << "E cv: " << std::endl << E_cv << std::endl;
std::cout << "R_cv: " << std::endl << R1 << std::endl;
std::cout << "t_cv: " << std::endl << t << std::endl;


std::cout << std::endl;
std::cout << "Printing true values" << std::endl;
std::cout << "E truth: " << std::endl << temp*R1_c1c2 << std::endl;
std::cout << "R truth: " << std::endl << R1_c1c2 << std::endl;
std::cout << " |T_truth| " << std::endl << T_c1c2_c2_normalized << std::endl;


std::cout << "GNSAC is moving results" << std::endl;
for (int i = 0; i < is_moving.size() ; i++)
{
	std::cout << is_moving[i] << std::endl;
}

std::vector<cv::Point2f> point_velocities;
std::vector<cv::Point2f> vel_rotated;


pd.ThresholdVelocities(E_cv,R_cv,image_points1,image_points2,point_velocities,vel_rotated,is_moving);
std::cout << "OPENCV is moving results" << std::endl;
for (int i = 0; i < is_moving.size() ; i++)
{
	std::cout << is_moving[i] << std::endl;
}


}



int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

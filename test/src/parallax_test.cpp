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


// Keeps track of the number of correct classifications.
struct Score {

	float correct_moving     = 0;
	float incorrect_moving   = 0;
	float correct_parallax   = 0;
	float incorrect_parallax = 0;
	float correct_others     = 0;
	float incorrect_others   = 0;
	float total_correct      = 0;
	float total_incorrect    = 0;
	float total_points       = 0;

	float correct_essential_matrix     = 0;
	float correct_rotation_matrix      = 0;
	float correct_translation_matrix   = 0;
	float incorrect_essential_matrix   = 0;
	float incorrect_rotation_matrix    = 0;
	float incorrect_translation_matrix = 0;

};


// Generate matching points from two different time samples. 
// world_points1 is from the first time sample, and 
// world_points2 is from the second. Since some of the points
// will represent moving features, some of world_points2
// will will be offset from world_points1 points. 
void GenerateWorldPoints(int num_points, float& moving_velocity_final, int& parallax_final, std::vector<cv::Point3f>& world_points1, std::vector<cv::Point3f>& world_points2)
{
	// Create a uniform random distribution generator. This will be 
	// used to create features in the world frame. 
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> feature_dist(-20, 20);
	std::uniform_real_distribution<> velocity_dist(2, 5);
	std::uniform_real_distribution<> height_dist(-1, 1);
	std::uniform_real_distribution<> parallax_dist(3, 5);
	std::uniform_real_distribution<> sgn(-1, 1);

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

			int s = 1;
			if(sgn(gen) < 0)
				s = -1;

			cv::Point3f v2(velocity_dist(gen)*Ts,velocity_dist(gen)*Ts,0);
			

			p2 = p1 + s*v2;

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
			// cv::Point3f par(0,0,-parallax_dist(gen));
			// p1 = p1 +par;
			// p2 = p1;
			p2=p1;
		}

		// p2 = p1;

		world_points1.push_back(p1);
		world_points2.push_back(p2);
}
}

// ----------------------------------------------------------------------------
// Transforms a 3x1 vector to a normalized skew symmetric matrix
cv::Mat Vec2Skew(const cv::Mat& vec)
{

	cv::Mat T_skew = cv::Mat::zeros(3,3,CV_64F);
	T_skew.at<double>(0,1) = -vec.at<double>(2,0);
	T_skew.at<double>(0,2) =  vec.at<double>(1,0);
	T_skew.at<double>(1,0) =  vec.at<double>(2,0);
	T_skew.at<double>(1,2) = -vec.at<double>(0,0);
	T_skew.at<double>(2,0) = -vec.at<double>(1,0);
	T_skew.at<double>(2,1) =  vec.at<double>(0,0);
	cv::normalize(T_skew,T_skew);

	return T_skew;

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

	cv::Mat R_x(3,3,CV_64F);
	R_x.at<double>(0,0) = 1;
	R_x.at<double>(0,1) = 0;
	R_x.at<double>(0,2) = 0;
	R_x.at<double>(1,0) = 0;
	R_x.at<double>(1,1) = cos(phi);
	R_x.at<double>(1,2) = sin(phi);
	R_x.at<double>(2,0) = 0;
	R_x.at<double>(2,1) = -sin(phi);
	R_x.at<double>(2,2) = cos(phi);

	cv::Mat R_y(3,3,CV_64F);
	R_y.at<double>(0,0) = cos(theta);
	R_y.at<double>(0,1) = 0;
	R_y.at<double>(0,2) = -sin(theta);
	R_y.at<double>(1,0) = 0;
	R_y.at<double>(1,1) = 1;
	R_y.at<double>(1,2) = 0;
	R_y.at<double>(2,0) = sin(theta);
	R_y.at<double>(2,1) = 0;
	R_y.at<double>(2,2) = cos(theta);

	cv::Mat R_z(3,3,CV_64F);
	R_z.at<double>(0,0) = cos(psi);
	R_z.at<double>(0,1) = sin(psi);
	R_z.at<double>(0,2) = 0;
	R_z.at<double>(1,0) = -sin(psi);
	R_z.at<double>(1,1) = cos(psi);
	R_z.at<double>(1,2) = 0;
	R_z.at<double>(2,0) = 0;
	R_z.at<double>(2,1) = 0;
	R_z.at<double>(2,2) = 1;

	return R_x*R_y*R_z;

}

//--------------------------------------------------------------------------------------------------------------------------------------------------

void GenerateImagePoints(const std::vector<cv::Point3f>& world_points1, const std::vector<cv::Point3f>& world_points2,
	                  std::vector<cv::Point2f>& image_points1, std::vector<cv::Point2f>& image_points2,
	                  cv::Mat& E_truth, cv::Mat& R_truth, cv::Mat& T_norm_truth)
{



	// Create a uniform random distribution generator. This will be 
	// used to create the transform from camera frame 1 to camera frame 2. 
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> angle_dist(-3, 3);          // degrees
	std::uniform_real_distribution<> translation_dist(-2, 2);

	// Set the camera matrix to identity so we work in the normalized image plane. 
	cv::Mat camera_matrix = cv::Mat::eye(3,3,CV_64F);
	float velocity = 2;  // m/s

	// Allocate memory for the rotation and translation vectors
	cv::Mat vR1_wc1(3,1,CV_64F);    // Rotation from world to camera frame 1
	cv::Mat vR2_c1c2(3,1,CV_64F);   // Rotation from camera frame 1 to camera frame 2. 
	cv::Mat T_wc1_c1(3,1,CV_64F);   // Translation vector from world to camera frame 1
	cv::Mat T_c1c2_c2(3,1,CV_64F);  // Translation vector from camera frame 1 to camera frame 2

		// Generate initial rotation matarix and convert it to rodriguez vector
	cv::Mat R1_wc1 = EulerAnglesToRotationMatrix(45,0,90, true);
	cv::Rodrigues(R1_wc1, vR1_wc1);

	// create initial translation vector and rotate it to the camera frame. 
	T_wc1_c1.at<double>(0) = 10;
	T_wc1_c1.at<double>(1) = 0;
	T_wc1_c1.at<double>(2) = 15;
	T_wc1_c1 = R1_wc1*T_wc1_c1;



	// Project the world points onto the image plane in camera frame 1
	cv::projectPoints(world_points1, vR1_wc1, T_wc1_c1, camera_matrix, cv::Mat(), image_points1);


	// Generate intermediate rotation matrix and convert it to rodriguez vector
	cv::Mat R1_c1c2 = EulerAnglesToRotationMatrix(angle_dist(gen),angle_dist(gen),angle_dist(gen), true);
	cv::Rodrigues(R1_c1c2*R1_wc1, vR2_c1c2);

	// Create translation vector from camera frame 1 to camera frame 2
	T_c1c2_c2.at<double>(0) = translation_dist(gen) + 0.0001;
	T_c1c2_c2.at<double>(1) = translation_dist(gen);
	T_c1c2_c2.at<double>(2) = translation_dist(gen);
	cv::normalize(T_c1c2_c2,T_c1c2_c2);
	T_c1c2_c2 = T_c1c2_c2*velocity;

	// Project the world points onto the image plane in the second frame
	cv::projectPoints(world_points2, vR2_c1c2, T_c1c2_c2 + R1_c1c2*T_wc1_c1, camera_matrix, cv::Mat(), image_points2);

	// Compose the True essential matrix, rotation matrix, and normalized translation vector
		
	cv::Mat T_skew = Vec2Skew(T_c1c2_c2);

	R_truth = R1_c1c2.clone();
	E_truth = T_skew*R1_c1c2;
	cv::normalize(T_c1c2_c2, T_norm_truth);
	
}

//--------------------------------------------------------------------------------------------------------------------------------------------------

void ScoreMethod(const std::vector<bool>& is_moving, const float& moving_velocity_final, const int& parallax_final, Score& score,
								 const cv::Mat& E_truth, const cv::Mat& R_truth, const cv::Mat& T_norm_truth, 
								 const cv::Mat& E_method, const cv::Mat& R_method, const cv::Mat& T_norm_method){

	// Score the methods
	for( unsigned i; i < is_moving.size(); i++)
	{

		if(i < moving_velocity_final)
		{

			if(is_moving[i] == true)
			{
				score.correct_moving++;
				score.total_correct++;
				score.total_points++;

			}
			else
			{
				score.incorrect_moving++;
				score.total_incorrect++;
				score.total_points++;
			}

		}
		else if(i < parallax_final)
		{
			if(is_moving[i] == false)
			{
				score.correct_parallax++;
				score.total_correct++;
				score.total_points++;
			}
			else
			{
				score.incorrect_parallax++;
				score.total_incorrect++;
				score.total_points++;
			}
		}
		else
		{
			if(is_moving[i] == false)
			{
				score.correct_others++;
				score.total_correct++;
				score.total_points++;
			}
			else
			{
				score.incorrect_others++;
				score.total_incorrect++;
				score.total_points++;
			}
		}


	}


	if (cv::norm(E_truth - E_method) < 0.5) // I don't know of an intuitive way to think of this.
	{
		score.correct_essential_matrix++;
	}
	else
	{
		score.incorrect_essential_matrix++;
	}

	if (fabs(cv::trace(R_truth.t()*R_method)[0]- 3) < 0.001) // Less than 1 degree error in the roll pitch and yaw
	{
		score.correct_rotation_matrix++;
	}
	else
	{
		score.incorrect_rotation_matrix++;
	}

	if (cv::norm(T_norm_truth - T_norm_method) < 0.001)
	{
		score.correct_translation_matrix++;
	}
	else
	{
		std::cout << "T_norm error: " << cv::norm(T_norm_truth - T_norm_method) << std::endl;
		score.incorrect_translation_matrix++;
	}




}

//--------------------------------------------------------------------------------------------------------------------------------------------------

float PrintScore(Score score, int method)
{


	if(method == 1)
	{
		std::cout << std::endl << "Scores for GNSAC: " << std::endl;
	}
	else if(method ==2)
	{
		std::cout << std::endl << "Scores for OpenCV: " << std::endl;
	}
	else
	{
		std::cout << std::endl << "Scores for Truth: " << std::endl;
	}

	std::cout << "Pts: " << std::endl;
	std::cout << "Correct Moving: " << score.correct_moving/(score.correct_moving + score.incorrect_moving)*100.0 << "%" << std::endl;
	std::cout << "Correct Parallax: " << score.correct_parallax/(score.correct_parallax + score.incorrect_parallax)*100.0 << "%" << std::endl;
	std::cout << "Correct Others: " << score.correct_others/(score.correct_others + score.incorrect_others)*100.0 << "%" << std::endl;
	std::cout << "Correct Total: " << score.total_correct/(score.total_correct + score.total_incorrect)*100.0 << "%" << std::endl;
	std::cout << "Total Number of Points: " << score.total_points << std::endl << std::endl;

	std::cout << "Transformation: " << std::endl;
	std::cout << "Correct Essential Matrix: " << score.correct_essential_matrix/(score.correct_essential_matrix + score.incorrect_essential_matrix)*100.0 << "%" << std::endl;
	std::cout << "Correct Rotation Matrix: " << score.correct_rotation_matrix/(score.correct_rotation_matrix + score.incorrect_rotation_matrix)*100.0 << "%" << std::endl;
	std::cout << "Correct Translation Matrix: " << score.correct_translation_matrix/(score.correct_translation_matrix + score.incorrect_translation_matrix)*100.0 << "%" << std::endl;
	std::cout << "Total Number of Iterations: " << score.correct_essential_matrix + score.incorrect_essential_matrix << std::endl << std::endl;


	return score.total_correct/score.total_points*100.0;

}

//--------------------------------------------------------------------------------------------------------------------------------------------------


// Test the parallax compensation algorithm
TEST(ParallaxTest, ProjectionTest) {

// Grab the GNSAC parameters
fs::path test_path = fs::path(__FILE__).remove_filename().parent_path();
std::cout << test_path << std::endl;
std::string param_filename = test_path / "param/test_param.yaml";


// Parameters
int num_sims = 20;              // Number of times to run the tests
int num_points = 300;           // Number of points to generate for each test
float moving_velocity_final;    // Points with indexs < moving_velocity_final will be moving in the world plane.
int parallax_final;             // Points with index > moving_velocity_final and < parallax_final will be closer to the camera to create parallax. 

// Declare the score objects. These will keep track of the number of correct
// point classification (moving, parallax, other)
Score score_gnsac;
Score score_cv;
Score score_truth;

// World points and corresponding image points from two different time frames.
std::vector<cv::Point3f> world_points1, world_points2;
std::vector<cv::Point2f> image_points1, image_points2;

// Declare variables
cv::Mat E_truth(3,3,CV_64F);            // True essential matrix
cv::Mat R_truth(3,3,CV_64F);            // True rotation matrix
cv::Mat T_norm_truth(3,1,CV_64F);       // True normalized translation vector
common::EHypothesis gnsac_result;       // Store gnsac results for E,R, and T_norm
cv::Mat E_gnsac, T_norm_gnsac,R_gnsac;  // OpenCV results for E,R, and T_norm
cv::Mat E_cv, R1,R2, T_norm_cv,R_cv;    // OpenCV results for E,R, and T_norm
std::vector<bool> gnsac_is_moving;      // If true, the point is moving perpendicular to the parallax field
std::vector<bool> cv_is_moving;         // If true, the point is moving perpendicular to the parallax field
std::vector<bool> truth_is_moving;      // If true, the point is moving perpendicular to the parallax field

gnsac::ParallaxDetector gnsac;          // Declare the solver
gnsac.Init(param_filename);             // Load param file


for (unsigned i = 0; i < num_sims; i++)
{

	GenerateWorldPoints(num_points, moving_velocity_final, parallax_final, world_points1, world_points2);

	GenerateImagePoints(world_points1, world_points2,image_points1, image_points2,E_truth, R_truth, T_norm_truth);


	//
	// Implement GNSAC
	//

	// gnsac.SetParallaxThreshold(0.0005);  // Set the threshold value
	gnsac.SetParallaxThreshold(0.00005);  // Set the threshold value
	gnsac_result = gnsac.ParallaxCompensation(image_points1,image_points2,gnsac_is_moving);
	eigen2cv(gnsac_result.E, E_gnsac);
	eigen2cv(gnsac_result.R, R_gnsac);
	eigen2cv(gnsac_result.t, T_norm_gnsac);

	//
	// Implement OpenCV
	//


	E_cv = cv::findEssentialMat(image_points1,image_points2, 1,cv::Point2d(0,0), cv::RANSAC, 0.999, 1e-5);
	cv::recoverPose(E_cv, image_points1, image_points2, R_cv, T_norm_cv);

	// Reconstruct the essential matrix using the correct rotation and norm translation
	cv::Mat T_skew = Vec2Skew(T_norm_cv);
	E_cv = T_skew*R_cv;








	std::vector<cv::Point2f> point_velocities; 
	std::vector<cv::Point2f> vel_rotated;
	gnsac.ThresholdVelocities(E_cv,R_cv,image_points1,image_points2,point_velocities,vel_rotated,cv_is_moving);

	// std::cout << "cv: t_norm: " << T_norm_cv << std::endl;
	// std::cout << "t truth: " << T_norm_truth << std::endl; 

	//
	// Test Truth
	//
	gnsac.ThresholdVelocities(E_truth,R_truth,image_points1,image_points2,point_velocities,vel_rotated,truth_is_moving);

	// Score the opencv and gnsac methods
	ScoreMethod(gnsac_is_moving, moving_velocity_final, parallax_final, score_gnsac,
									 E_truth, R_truth, T_norm_truth, 
									 E_gnsac, R_gnsac, T_norm_gnsac);

	ScoreMethod(cv_is_moving, moving_velocity_final, parallax_final, score_cv,
									 E_truth, R_truth, T_norm_truth, 
									 E_cv, R_cv, T_norm_cv);

	ScoreMethod(truth_is_moving, moving_velocity_final, parallax_final, score_truth,
								 E_truth, R_truth, T_norm_truth, 
								 E_truth, R_truth, T_norm_truth);



}




float final_score_gnsac = PrintScore(score_gnsac,1);
float final_score_cv = PrintScore(score_cv,2);
float final_score_truth = PrintScore(score_truth,3);


ASSERT_TRUE(final_score_cv > 90.0 && final_score_gnsac > 90.0);




// std::cout << std::endl;
// std::cout << "Printing values from gnsac." << std::endl;
// std::cout << "E: " << std::endl << E_gnsac << std::endl;
// std::cout << "R: " << std::endl << R_gnsac << std::endl;
// std::cout << "T: " << std::endl << T_norm_gnsac << std::endl;
// std::cout << "Printing values from gnsac." << std::endl;
// std::cout << "E: " << std::endl << gnsac_result.E << std::endl;
// std::cout << "R: " << std::endl << gnsac_result.R << std::endl;
// std::cout << "T: " << std::endl << gnsac_result.t << std::endl;

// std::cout << std::endl;
// std::cout << "Printing opencv values" << std::endl;
// std::cout << "E cv: " << std::endl << E_cv << std::endl;
// std::cout << "R_cv: " << std::endl << R_cv << std::endl;
// std::cout << "t_cv: " << std::endl << T_norm_cv << std::endl;


// std::cout << std::endl;
// std::cout << "Printing true values" << std::endl;
// std::cout << "E truth: " << std::endl << E_truth << std::endl;
// std::cout << "R truth: " << std::endl << R_truth << std::endl;
// std::cout << " |T_truth| " << std::endl << T_norm_truth << std::endl;


// std::cout << "GNSAC is moving results" << std::endl;
// for (int i = 0; i < parallax_final ; i++)
// {
// 	std::cout << gnsac_is_moving[i] << std::endl;
// 	// if(gnsac_is_moving[i] == false && i < moving_velocity_final)
// 	// {

// 	// }
// }




// std::cout << "OPENCV is moving results" << std::endl;
// for (int i = 0; i < parallax_final ; i++)
// {
// 	std::cout << cv_is_moving[i] << std::endl;
// }


}



int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

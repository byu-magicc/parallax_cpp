// GNSAC
#include "parallax_detector/parallax_detector.h"
#include "common/solvers.h"
#include <opencv2/opencv.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <random>


// Generate matching points from two different time samples. 
// world_points1 is from the first time sample, and 
// world_points2 is from the second. Since some of the points
// will represent moving features, some of world_points2
// will will be offset from world_points1 points. 
void generate_world_points(int num_points, int& moving_velocity_final, int& parallax_final, std::vector<cv::Point3f>& world_points1, std::vector<cv::Point3f>& world_points2)
{
	// Create a uniform random distribution generator. This will be 
	// used to create features in the world frame. 
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<> feature_dist(-50, 50);
	std::uniform_int_distribution<> velocity_dist(-1, 1);
	std::uniform_int_distribution<> parallax_dist(1, 5);

	// Clear history
	world_points1.clear();
	world_points2.clear();
	

	// Generate points
	moving_velocity_final = num_points*0.1;
	parallax_final = num_points*0.2;
	for(int i = 0; i < num_points; i++)
	{

		cv::Point3f p1(feature_dist(gen),feature_dist(gen),1);
		cv::Point3f p2;

		// Make 10 percent of the points have moving velocity
		if(i < moving_velocity_final)
		{

			// Create random velocity in the x and y direction
			cv::Point3f v2(velocity_dist(gen),velocity_dist(gen),0);
			p2 = p1 + v2;

		}
		// Make 10 percent of the points be closer to the camera
		// to create parallax
		else if(i < parallax_final)
		{

			// Create parallax by adding height to the feature
			cv::Point3f par(0,0,parallax_dist(gen));
			p1 = p1+par;
			p2 = p1;

		}
		else 
		{
			p2=p1;
		}

		world_points1.push_back(p1);
		world_points2.push_back(p2);
}
}




// Test the parallax compensation algorithm
TEST(ParallaxTest, ProjectionTest) {


// Get the world points from two different time frames.
std::vector<cv::Point3f> world_points1, world_points2;
int num_points = 50;
int moving_velocity_final = 0;
int parallax_final = 0;
generate_world_points(num_points, moving_velocity_final, parallax_final, world_points1, world_points2);


// Generate initial rotation from world to camera

// cv::Mat camera_matrix = cv::Mat::eye(3,3,CV_32F);

// cv::projectPoints(world_points, cv::Vec3f(0,0,0), cv::Vec3f(0,0,0), camera_matrix, cv::Mat(), image_points);



// common::EHypothesis result;
// std::cout <<image_points << std::endl;

// std::cout << "points1 " << world_points1 << std::endl;
// std::cout << "world_points2 " << world_points2 << std::endl;
// std::cout << "moving_velocity_final " << moving_velocity_final << std::endl;
// std::cout << "parallax_final " << parallax_final << std::endl;


}



int main(int argc, char **argv) {

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

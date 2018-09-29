#include <eigen3/Eigen/Dense>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/eigen.hpp"
#include "ptr/GN_step.h"
#include "common.h"
#include <vector>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace cv;

void timing(string yaml_filename)
{

}

void accuracy(string yaml_filename)
{
	// Point data
	VideoPointData video_data = VideoPointData(yaml_filename);
	if (video_data.RT.size() < video_data.pts1.size())
	{
		printf("Error: Missing truth data (%d < %d)\n", (int)video_data.RT.size(), (int)video_data.pts1.size());
		assert(0);
	}
	else
		printf("Warning: Truth data size does not match point data size (%d != %d)\n",
			(int)video_data.RT.size(), (int)video_data.pts1.size());

	// Convert Eigen to OpenCV matrices.
	cv::Mat camera_matrix;
	cv::Mat dist_coeffs;
	eigen2cv(video_data.camera_matrix, camera_matrix);
	eigen2cv(video_data.dist_coeffs, dist_coeffs);

	// Loop for all points
	cv::RNG rng(cv::getCPUTickCount());
	std::ofstream log_file;
	log_file.open("../logs/log_test.bin");
	for (int frame = 0; frame < video_data.pts1.size(); frame++)
	{
		// Undistort points
		vector<Point2f> pts1_f, pts2_f;
		undistortPoints(video_data.pts1[frame], pts1_f, camera_matrix, dist_coeffs);
		undistortPoints(video_data.pts2[frame], pts2_f, camera_matrix, dist_coeffs);

		// Convert to double
		vector<Point2d> pts1, pts2;
		for (int i = 0; i < pts1_f.size(); i++)
		{
			pts1.push_back(pts1_f[i]);
			pts2.push_back(pts2_f[i]);
		}		

		// Calculate essential matrix
		Mat R0 = Mat::eye(3, 3, CV_64F);
		Mat t0 = (Mat_<double>(8, 8) << rng.gaussian(1), rng.gaussian(1), rng.gaussian(1));
		Mat R2, t2;
		vector<Mat> all_hypotheses;
		Mat E = findEssentialMatGN(pts1, pts2, R0, t0, R2, t2, all_hypotheses, 100, 10, true, false, false);

		// Calculate error to truth essential matrix
		Matrix3d R2_eig;
		Vector3d t2_eig;
		cv2eigen(R2, R2_eig);
		cv2eigen(t2, t2_eig);
		Vector3d err = err_truth(R2_eig, t2_eig, video_data.RT[frame]);
		log_file.write((char*)&err, sizeof(double) * 3);
		if(frame < 5)
			cout << err << endl;
	}
	log_file.close();	
}

int main(int argc, char *argv[])
{
	// Get rid of first arg (executable name)
	argc--; argv++;

	// Make sure there are sufficient arguments
	if(argc < 2)
	{
		cout << "Usage: ./cli pts_in [timing, accuracy, or sweep]" << endl;
		return 0;
	}

	string yaml_filename = string(argv[0]);
	string s = string(argv[1]);
	if(s == "timing")
		timing(yaml_filename);
	else if(s == "accuracy")
		accuracy(yaml_filename);
	// else if(s == "sweep")
	// 	sweep_sensor_noise(argc - 1, argv + 1);
	else
		cout << "Usage: cli [timing, accuracy, or sweep]" << endl;
}
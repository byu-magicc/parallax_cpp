#include <eigen3/Eigen/Dense>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ptr/GN_step.h"
#include "common.h"
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace cv;

void timing(string yaml_filename)
{
	// Point data
	VideoPointData video_data = VideoPointData(yaml_filename);

	// Undistort points
	int frame = 0;
	vector<Point2f> pts1;
	vector<Point2f> pts2;
	cv::Mat camera_matrix;
	cv::Mat dist_coeffs;
	eigen2cv(video_data.camera_matrix, camera_matrix);
	eigen2cv(video_data.dist_coeffs, dist_coeffs);
	undistortPoints(video_data.data2[frame], pts2, camera_matrix, dist_coeffs);
	undistortPoints(video_data.data1[frame], pts1, camera_matrix, dist_coeffs);
	cout << pts1.size() << " points" << endl;

	// Write random data to plot
	std::ofstream log_file;
	log_file.open("../logs/log_test.bin");
	for (int i = 0; i < video_data.data1.size(); i++)
	{
		double n_pts = video_data.data1[i].size();
		log_file.write((char*)&n_pts, sizeof(double));
	}
	log_file.close();
}

void accuracy(int argc, char *argv[])
{
	
}

int main(int argc, char *argv[])
{
	// Get rid of first arg (executable name)
	argc--; argv++;

	// Make sure there are sufficient arguments
	if(argc < 2)
	{
		cout << "Usage: ./cli pts_in [timing, comp_mpda, or sweep]" << endl;
		return 0;
	}

	string yaml_filename = string(argv[0]);
	string s = string(argv[1]);
	if(s == "timing")
		timing(yaml_filename);
	// else if(s == "comp_mpda")
	// 	compare_mpda(argc - 1, argv + 1);
	// else if(s == "sweep")
	// 	sweep_sensor_noise(argc - 1, argv + 1);
	else
		cout << "Usage: cli [timing, comp_mpda, or sweep]" << endl;
}
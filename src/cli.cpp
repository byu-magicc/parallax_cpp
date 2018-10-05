#include <eigen3/Eigen/Dense>
#include <iostream>
#include "gnsac_ptr_eig.h"
#include "gnsac_ptr_ocv.h"
#include "common.h"
#include <vector>
#include <fstream>
#include <random>
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace common;

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

	// Random number generation
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(seed);
	std::normal_distribution<double> dist(0.0, 1.0);

	// Loop for all points
	std::ofstream accuracy_log_file;
	std::ofstream timing_log_file;
	accuracy_log_file.open("../logs/gnsac_ptr_opencv/accuracy.bin");
	timing_log_file.open("../logs/gnsac_ptr_opencv/timing.bin");
	int frames = video_data.pts1.size();
	common::progress(0, frames);
	for (int frame = 0; frame < frames; frame++)
	{
		// Undistort points
		scan_t pts1, pts2;
		undistort_points(video_data.pts1[frame], pts1, video_data.camera_matrix);
		undistort_points(video_data.pts2[frame], pts2, video_data.camera_matrix);

		// Calculate essential matrix
		Matrix3d R0 = Matrix3d::Identity();
		Vector3d t0;
		t0 << dist(rng), dist(rng), dist(rng);
		Matrix3d R2;
		Vector3d t2;
		tic();
		cat_timer_reset();
		Matrix3d E = gnsac_ptr_opencv::findEssentialMatGN(pts1, pts2, R0, t0, R2, t2, 100, 10, true, false);
		timeMeasurement time_E = toc("FindE", 1, 2, false);
		timing_log_file.write((char*)get_cat_times(), sizeof(double) * TIME_CATS_COUNT);
		timing_log_file.write((char*)&time_E.actualTime, sizeof(double));		

		// Calculate error to truth essential matrix
		Vector2d err = common::err_truth(R2, t2, video_data.RT[frame]);
		accuracy_log_file.write((char*)&err, sizeof(double) * 2);
		common::progress(frame + 1, frames);
	}
	accuracy_log_file.close();
	timing_log_file.close();
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
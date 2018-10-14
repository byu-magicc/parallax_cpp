#include <eigen3/Eigen/Dense>
#include <iostream>
#include "gnsac_ptr_eig.h"
#include "gnsac_ptr_ocv.h"
#include "common.h"
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <memory>
#include <experimental/filesystem>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/eigen.hpp"

using namespace std;
using namespace Eigen;
using namespace common;
namespace fs = std::experimental::filesystem;

void run_test(string video_str, string solver_str)
{
	// Create folder for results
	string result_directory = fs::path("../logs") / video_str / solver_str;
	cout << result_directory << endl;
	if(!fs::exists(result_directory))
		fs::create_directory(result_directory);

	// Load point data
	string video_yaml = fs::path("../params/videos") / (video_str + ".yaml");
	cout << video_yaml << endl;
	VideoPointData video_data = VideoPointData(video_yaml);
	if (video_data.RT.size() < video_data.pts1.size())
	{
		printf("Error: Missing truth data (%d < %d)\n", (int)video_data.RT.size(), (int)video_data.pts1.size());
		assert(0);
	}
	else
		printf("Warning: Truth data size does not match point data size (%d != %d)\n",
			(int)video_data.RT.size(), (int)video_data.pts1.size());

	// Load solver, copy yaml file into folder for reference
	string solver_yaml = fs::path("../params/solvers") / (solver_str + ".yaml");
	cout << solver_yaml << endl;
	shared_ptr<ESolver> solver = ESolver::from_yaml(solver_yaml, result_directory);
	string solver_copy_filename = fs::path(result_directory) / "solver.yaml";
	if(fs::exists(solver_copy_filename))
		fs::remove(solver_copy_filename);
	fs::copy(solver_yaml, solver_copy_filename);

	// Random number generation
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(seed);
	std::normal_distribution<double> dist(0.0, 1.0);

	// Open log files
	std::ofstream accuracy_log_file;
	std::ofstream timing_log_file;
	accuracy_log_file.open(fs::path(result_directory) / "accuracy.bin");
	timing_log_file.open(fs::path(result_directory) / "timing.bin");

	// Loop for all points
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
		tic();
		cat_timer_reset();
		EHypothesis result;
		solver->find_best_hypothesis(pts1, pts2, video_data.RT[frame], result);
		timeMeasurement time_E = toc("FindE", 1, 2, false);
		timing_log_file.write((char*)get_cat_times(), sizeof(double) * TIME_CATS_COUNT);
		timing_log_file.write((char*)&time_E.actualTime, sizeof(double));

		// Calculate error to truth essential matrix.
		Vector2d err = common::err_truth(result.R, result.t, video_data.RT[frame]);
		accuracy_log_file.write((char*)err.data(), sizeof(double) * 2);
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
	string usage_str = "Usage: ./cli video solver";
	if(argc < 2)
	{
		cout << usage_str << endl;
		return 0;
	}
	string video = string(argv[0]);
	string solver = string(argv[1]);
	run_test(video, solver);
}
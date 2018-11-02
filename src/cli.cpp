#include <eigen3/Eigen/Dense>
#include <iostream>
#include "gnsac_ptr_eig.h"
#include "gnsac_ptr_ocv.h"
#include "gnsac_eigen.h"
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

void concat_files(string name1, string name2, string name_out)
{
	std::ifstream file1(name1);
	std::ifstream file2(name2);
	if(!file1.is_open())
	{
		cout << "Error opening file " << name1 << endl;
		throw common::Exception("Error opening file");
	}
		
	if(!file2.is_open())
	{
		cout << "Error opening file " << name2 << endl;
		throw common::Exception("Error opening file");
	}
	std::ofstream file_out(name_out);
	file_out << file1.rdbuf() << endl << file2.rdbuf();
	file1.close();
	file2.close();
	file_out.close();
}

void run_test(string video_str, string solver_str, string test_str, int frames)
{
	// Create folder for results
	string dir_part1 = fs::path("../logs");
	string dir_part2 = fs::path("../logs") / video_str;
	string dir_part3 = fs::path("../logs") / video_str / solver_str;
	string result_directory = fs::path("../logs") / video_str / solver_str / test_str;
	cout << result_directory << endl;
	if(!fs::exists(dir_part1))
		fs::create_directory(dir_part1);
	if(!fs::exists(dir_part2))
		fs::create_directory(dir_part2);
	if(!fs::exists(dir_part3))
		fs::create_directory(dir_part3);
	if(!fs::exists(result_directory))
		fs::create_directory(result_directory);

	// Load point data
	string video_yaml = fs::path("../param/videos") / (video_str + ".yaml");
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

	// Concatenate solver and test to create a run yaml
	string solver_yaml = fs::path("../param/solvers") / (solver_str + ".yaml");
	string test_yaml = fs::path("../param/tests") / (test_str + ".yaml");
	string run_yaml = fs::path(result_directory) / "solver.yaml";
	cout << "Solver: " << solver_yaml << endl;
	cout << "Tests: " << test_yaml << endl;
	concat_files(test_yaml, solver_yaml, run_yaml);
	shared_ptr<ESolver> solver = ESolver::from_yaml(run_yaml);

	// Random number generation
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(seed);
	std::normal_distribution<double> dist(0.0, 1.0);

	// Open log files
	common::init_logs(run_yaml, result_directory);

	// Loop for all points
	if (frames == -1)
		frames = video_data.pts1.size();
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
		common::write_log(common::log_timing, (char*)get_cat_times(), sizeof(double) * TIME_CATS_COUNT);
		common::write_log(common::log_timing, (char*)&time_E.actualTime, sizeof(double));
		common::write_log(common::log_timing_verbose, (char*)get_cat_times_verbose(), sizeof(double) * TIME_CATS_VERBOSE_COUNT);
		common::write_log(common::log_timing_verbose, (char*)&time_E.actualTime, sizeof(double));

		// Calculate error to truth essential matrix.
		Vector4d err;
		if(result.has_RT)
			err = common::err_truth(result.R, result.t, video_data.RT[frame]);
		else
			err = common::err_truth(result.E, video_data.RT[frame]);
		common::write_log(common::log_accuracy, (char*)err.data(), sizeof(double) * 4);
		common::progress(frame + 1, frames);
	}
	common::close_logs();
}

int main(int argc, char *argv[])
{
	//gnsac_eigen::run_tests();

	// Get rid of first arg (executable name)
	argc--; argv++;

	// Make sure there are sufficient arguments
	string usage_str = "Usage: ./cli video solver test [frames]";
	if(argc < 3)
	{
		cout << usage_str << endl;
		return 0;
	}
	string video = string(argv[0]);
	string solver = string(argv[1]);
	string test = string(argv[2]);
	int frames = (argc >= 4) ? atoi(argv[3]) : -1;
	run_test(video, solver, test, frames);
}
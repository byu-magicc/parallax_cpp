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

void run_test(string video_yaml, string solver_yaml, string five_point_directory, string result_directory)
{
	// Create folder for results
	if(!fs::exists(result_directory))
		fs::create_directory(result_directory);

	// Load point data
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
	//shared_ptr<ESolver> solver = ESolver::from_yaml(solver_yaml);
	YAML::Node node = YAML::LoadFile(solver_yaml);
	shared_ptr<gnsac_ptr_eigen::GNSAC_Solver> solver = make_shared<gnsac_ptr_eigen::GNSAC_Solver>(solver_yaml, node);
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
	solver->init_optimizer_log(result_directory, false);
	solver->init_comparison_log(result_directory, five_point_directory);

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

void compare_hypotheses(string video_yaml, string solver_yaml, string five_point_directory, string result_directory)
{
	// Create folder for results
	if(!fs::exists(result_directory))
		fs::create_directory(result_directory);

	// Load point data
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
	//shared_ptr<ESolver> solver_ = ESolver::from_yaml(solver_yaml);
	YAML::Node node = YAML::LoadFile(solver_yaml);
	shared_ptr<gnsac_ptr_eigen::GNSAC_Solver> solver = make_shared<gnsac_ptr_eigen::GNSAC_Solver>(solver_yaml, node);
	string solver_copy_filename = fs::path(result_directory) / "solver.yaml";
	if(fs::exists(solver_copy_filename))
		fs::remove(solver_copy_filename);
	fs::copy(solver_yaml, solver_copy_filename);

	// Random number generation
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(seed);
	std::normal_distribution<double> dist_normal(0.0, 1.0);

	// Open log files
	solver->init_optimizer_log(result_directory, true);
	solver->init_comparison_log(result_directory, five_point_directory);

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
		std::uniform_int_distribution<> dist(0, pts1.size() - 1);
		scan_t subset1, subset2;
		EHypothesis result;
		gnsac_ptr_eigen::getSubset(pts1, pts2, subset1, subset2, 5, dist, rng);
		solver->find_best_hypothesis(pts1, pts2, video_data.RT[frame], result);

		// Update progress bar
		common::progress(frame + 1, frames);
	}
}

int main(int argc, char *argv[])
{
	// Get rid of first arg (executable name)
	argc--; argv++;

	// Make sure there are sufficient arguments
	string usage_str = "Usage: ./cli [full, hypo] video_yaml solver_yaml five_point_directory results_folder";
	if(argc < 5)
	{
		cout << usage_str << endl;
		return 0;
	}
	string run_type = string(argv[0]);
	string video_yaml = string(argv[1]);
	string solver_yaml = string(argv[2]);
	string five_point_directory = string(argv[3]);
	string results_folder = string(argv[4]);
	if (run_type == "full")
		run_test(video_yaml, solver_yaml, five_point_directory, results_folder);
	else if (run_type == "hypo")
		compare_hypotheses(video_yaml, solver_yaml, five_point_directory, results_folder);
	else
		cout << usage_str << endl;
}
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

void run_test(string video_yaml, string solver_yaml, string result_directory)
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
	shared_ptr<ESolver> solver = ESolver::from_yaml(solver_yaml);
	string solver_copy_filename = fs::path(result_directory) / "solver.yaml";
	if(fs::exists(solver_copy_filename))
		fs::remove(solver_copy_filename);
	fs::copy(solver_yaml, solver_copy_filename);

	// Random number generation
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(seed);
	std::normal_distribution<double> dist(0.0, 1.0);

	// Loop for all points
	std::ofstream accuracy_log_file;
	std::ofstream timing_log_file;
	accuracy_log_file.open(fs::path(result_directory) / "accuracy.bin");
	timing_log_file.open(fs::path(result_directory) / "timing.bin");
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
		EHypothesis initial_guess = EHypothesis(Matrix3d::Zero(), R0, t0);
		EHypothesis result;
		solver->find_best_hypothesis(pts1, pts2, initial_guess, result);
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

void compare_hypotheses(string video_yaml, string solver_yaml, string result_directory)
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
	solver->max_iterations = 1000;
	solver->exit_tolerance = 0;
	if(fs::exists(solver_copy_filename))
		fs::remove(solver_copy_filename);
	fs::copy(solver_yaml, solver_copy_filename);

	// Random number generation
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(seed);
	std::normal_distribution<double> dist_normal(0.0, 1.0);

	// Open log files
	std::ofstream accuracy_log_file, timing_log_file, comparison_tr_log_file, comparison_gn_log_file;
	accuracy_log_file.open(fs::path(result_directory) / "5-point_accuracy.bin");
	timing_log_file.open(fs::path(result_directory) / "5-point_timing.bin");
	comparison_tr_log_file.open(fs::path(result_directory) / "5-point_comparison_tr.bin");
	comparison_gn_log_file.open(fs::path(result_directory) / "5-point_comparison_gn.bin");
	solver->init_optimizer_log(fs::path(result_directory) / "optimizer.bin", true);

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
		gnsac_ptr_eigen::getSubset(pts1, pts2, subset1, subset2, 5, dist, rng);

		// Convert to Point2f
		int n_pts = subset1.size();
		vector<cv::Point2d> subset1_cv = vector<cv::Point2d>(n_pts);
		vector<cv::Point2d> subset2_cv = vector<cv::Point2d>(n_pts);
		for (int i = 0; i < n_pts; i++)
		{
			subset1_cv[i].x = subset1[i](0);
			subset1_cv[i].y = subset1[i](1);
			subset2_cv[i].x = subset2[i](0);
			subset2_cv[i].y = subset2[i](1);
		}

		// Calc (multiple) hypotheses using 5-point algorithm
		tic();
		cat_timer_reset();
		cv::Mat E_cv = findEssentialMat(subset1_cv, subset2_cv, cv::Mat::eye(3, 3, CV_64F));
		timeMeasurement time_E = toc("FindE", 1, 2, false);
		timing_log_file.write((char*)get_cat_times(), sizeof(double) * TIME_CATS_COUNT);
		timing_log_file.write((char*)&time_E.actualTime, sizeof(double));
		if(E_cv.rows % 3 != 0 || (E_cv.rows > 0 && E_cv.cols != 3))
		{
			printf("Invalid essential matrix size: [%d x %d]\n", E_cv.rows, E_cv.cols);
			exit(EXIT_FAILURE);
		}
		int n_hypotheses_5P = E_cv.rows / 3;
		vector<Matrix3d> hypotheses_5P = vector<Matrix3d>(n_hypotheses_5P);
		for(int i = 0; i < n_hypotheses_5P; i++)
		{
			Map<Matrix<double, 3, 3, RowMajor>> E_i = Map<Matrix<double, 3, 3, RowMajor>>(&E_cv.at<double>(i * 3, 0));
			hypotheses_5P[i] = E_i;
		}

		// Calc (single) hypothesis using GN algorithm
		Matrix3d R0 = Matrix3d::Identity();
		Vector3d t0;
		t0 << dist(rng), dist(rng), dist(rng);
		tic();
		cat_timer_reset();
		EHypothesis initial_guess = EHypothesis(Matrix3d::Zero(), R0, t0);
		vector<common::EHypothesis> hypotheses_GN;
		solver->generate_hypotheses(subset1, subset2, initial_guess, hypotheses_GN);
		time_E = toc("FindE", 1, 2, false);
		timing_log_file.write((char*)get_cat_times(), sizeof(double) * TIME_CATS_COUNT);
		timing_log_file.write((char*)&time_E.actualTime, sizeof(double));
		
		// Score each hypothesis (up to 10, 11th is for GN)
		// (error should be very small since they are minimum subsets)
		vector<double> mean_err = vector<double>(11, -1);
		for(int i = 0; i < n_hypotheses_5P; i++)
			mean_err[i] = sampson_err(hypotheses_5P[i], subset1, subset2)[1];
		mean_err[10] = sampson_err(hypotheses_GN[0].E, subset1, subset2)[1];
		accuracy_log_file.write((char*)&mean_err[0], sizeof(double) * 11);

		// Find out which 5-point E is closest to the truth and which is closest to GN.
		// TODO: Should R errors be penalized more than t?
		Vector2d vec_none;
		vec_none << -1, -1;
		scan_t dist_truth = scan_t(11, vec_none);
		scan_t dist_GN = scan_t(10, vec_none);
		for(int i = 0; i < n_hypotheses_5P; i++)
		{
			dist_truth[i] = err_truth(hypotheses_5P[i], video_data.RT[frame]);
			dist_GN[i] = dist_E(hypotheses_5P[i], hypotheses_GN[0].E);
		}
		dist_truth[10] = err_truth(hypotheses_GN[0].E, video_data.RT[frame]);
		comparison_tr_log_file.write((char*)&dist_truth[0], sizeof(double) * 11 * 2);
		comparison_gn_log_file.write((char*)&dist_GN[0], sizeof(double) * 10 * 2);

		// Update progress bar
		common::progress(frame + 1, frames);
	}
	accuracy_log_file.close();
	comparison_tr_log_file.close();
	comparison_gn_log_file.close();
	timing_log_file.close();
}

int main(int argc, char *argv[])
{
	// Get rid of first arg (executable name)
	argc--; argv++;

	// Make sure there are sufficient arguments
	string usage_str = "Usage: ./cli [full, hypo] video_yaml solver_yaml results_folder";
	if(argc < 3)
	{
		cout << usage_str << endl;
		return 0;
	}
	string run_type = string(argv[0]);
	string video_yaml = string(argv[1]);
	string solver_yaml = string(argv[2]);
	string results_folder = string(argv[3]);
	if (run_type == "full")
		run_test(video_yaml, solver_yaml, results_folder);
	else if (run_type == "hypo")
		compare_hypotheses(video_yaml, solver_yaml, results_folder);
	else
		cout << usage_str << endl;
}
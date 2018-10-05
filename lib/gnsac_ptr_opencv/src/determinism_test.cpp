#include "opencv2/core/core.hpp"
#include "common.h"
#include <eigen3/Eigen/Dense>
#include "gnsac_ptr_ocv.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace gnsac_ptr_opencv;

#define write_hypothesis(checker, model) \
{ \
	write_check(checker, (char*)model.E_.ptr<double>(), sizeof(double)*3*3); \
	write_check(checker, (char*)model.R_.ptr<double>(), sizeof(double)*3*3); \
	write_check(checker, (char*)model.TR_.ptr<double>(), sizeof(double)*3*3); \
	write_check(checker, (char*)model.t_.ptr<double>(), sizeof(double)*3*1); \
	write_check(checker, (char*)model.E, sizeof(double)*3*3); \
	write_check(checker, (char*)model.R, sizeof(double)*3*3); \
	write_check(checker, (char*)model.TR, sizeof(double)*3*3); \
	write_check(checker, (char*)model.t, sizeof(double)*3*1); \
	write_check(checker, (char*)&model.cost, sizeof(double)); \
}

cv::Mat rnd_cv_matrix(int rows, int cols, cv::RNG& rng)
{
	cv::Mat A = cv::Mat::zeros(rows, cols, CV_64F);
	for(int r = 0; r < A.rows; r++)
		for(int c = 0; c < A.cols; c++)
			A.at<double>(r, c) = rng.gaussian(1);
	return A;
}

void determinism_test(int trial)
{
	// Open logfile
	common::DeterminismChecker checker = common::DeterminismChecker("determinism", trial);

	// Video data
	string yaml_filename = "../params/holodeck.yaml";
	common::VideoPointData video_data = common::VideoPointData(yaml_filename);

	// Init random number generator
	unsigned seed = 0;
	cv::RNG rng(0);
	for(int i = 0; i < 100; i++)
	{
		// First make sure random number generator is deterministic!
		write_check_val(checker, rng.gaussian(1));

		// Load real data, undistort points
		common::scan_t pts1_eig, pts2_eig;
		write_check_val(checker, video_data.pts1[i].size());
		write_check_val(checker, video_data.pts2[i].size());
		write_check(checker, (char*)video_data.pts1[i].data(), sizeof(Vector2d) * video_data.pts1[i].size());
		write_check(checker, (char*)video_data.pts2[i].data(), sizeof(Vector2d) * video_data.pts2[i].size());
		common::undistort_points(video_data.pts1[i], pts1_eig, video_data.camera_matrix);
		common::undistort_points(video_data.pts2[i], pts2_eig, video_data.camera_matrix);		
		write_check_val(checker, pts1_eig.size());
		write_check_val(checker, pts2_eig.size());
		write_check(checker, (char*)pts1_eig.data(), sizeof(Vector2d) * pts1_eig.size());
		write_check(checker, (char*)pts2_eig.data(), sizeof(Vector2d) * pts2_eig.size());

		// Convert to Point2d
		int n_pts = pts1_eig.size();
		vector<cv::Point2d> pts1 = vector<cv::Point2d>(n_pts);
		vector<cv::Point2d> pts2 = vector<cv::Point2d>(n_pts);
		for (int i = 0; i < n_pts; i++)
		{
			pts1[i].x = pts1_eig[i](0);
			pts1[i].y = pts1_eig[i](1);
			pts2[i].x = pts2_eig[i](0);
			pts2[i].y = pts2_eig[i](1);
		}		
		write_check(checker, (char*)&pts1[0], sizeof(double) * 2 * pts1.size());
		write_check(checker, (char*)&pts2[0], sizeof(double) * 2 * pts2.size());

		// Sinc 
		write_check_val(checker, sinc(rng.gaussian(1)));

		// GN Hypothesis
		cv::Mat R0_m = rnd_cv_matrix(3, 3, rng);
		cv::Mat t0_m = rnd_cv_matrix(3, 1, rng);
		GNHypothesis bestModel = GNHypothesis(R0_m, t0_m);
		write_hypothesis(checker, bestModel);

		// Default init
		GNHypothesis model;
		write_hypothesis(checker, model);

		// Full score
		write_check_val(checker, score_LMEDS(pts1, pts2, bestModel.E, 1e10));
		write_check_val(checker, score_LMEDS2(pts1, pts2, bestModel.E, 1e10));
		write_check_val(checker, score_LMEDS(pts1, pts2, model.E, 1e10));
		write_check_val(checker, score_LMEDS2(pts1, pts2, model.E, 1e10));
		write_check_val(checker, score_LMEDS(pts1, pts2, model.E, 1e10));
		write_check_val(checker, score_LMEDS2(pts1, pts2, model.E, 1e10));

		// Get subset
		vector<cv::Point2d> subset1;
		vector<cv::Point2d> subset2;
		getSubset(pts1, pts2, subset1, subset2, 5, rng);
		write_check(checker, (char*)&subset1[0], sizeof(double) * 2 * subset1.size());
		write_check(checker, (char*)&subset2[0], sizeof(double) * 2 * subset2.size());

		// Copy model
		bestModel.cost = rng.gaussian(1);
		copyHypothesis(bestModel, model);
		write_hypothesis(checker, bestModel);
		write_hypothesis(checker, model);

		for(int j = 0; j < 10; j++)
		{
			GN_step(subset1, subset2, model.R, model.TR, model.E, model.R, model.TR, model.t, 1, false);
			write_hypothesis(checker, model);
		}
		for(int j = 0; j < 10; j++)
		{
			GN_step(subset1, subset2, model.R, model.TR, model.E, model.R, model.TR, model.t, 1, true);
			write_hypothesis(checker, model);
		}
		
		// Calculate essential matrix
		Matrix3d R0 = Matrix3d::Identity();
		Vector3d t0;
		t0 << rng.gaussian(1), rng.gaussian(1), rng.gaussian(1);
		Matrix3d R2;
		Vector3d t2;
		Matrix3d E = findEssentialMatGN(pts1_eig, pts2_eig, R0, t0, R2, t2, 100, 10, true, false);
		write_check(checker, (char*)E.data(), sizeof(double)*3*3);
		write_check(checker, (char*)R2.data(), sizeof(double)*3*3);
		write_check(checker, (char*)t2.data(), sizeof(double)*3*1);

		// Error
		write_check(checker, (char*)video_data.RT[i].data(), sizeof(double)*4*4);
		Vector2d err = common::err_truth(R2, t2, video_data.RT[i]);
		write_check(checker, (char*)err.data(), sizeof(double)*2);
	}	
}

int main(int argc, char *argv[])
{
	// Get rid of first arg (executable name)
	argc--; argv++;

	// Make sure there are sufficient arguments
	if(argc < 1)
	{
		cout << "Usage: ./det_check trial_number" << endl;
		return 0;
	}
	determinism_test(atoi(argv[0]));
}
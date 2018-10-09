#include "common.h"
#include "solvers.h"
#include <eigen3/Eigen/Dense>
#include "gnsac_ptr_eig.h"
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <iostream>

using namespace std;
using namespace Eigen;
using namespace gnsac_ptr_eigen;

void determinism_test(int trial)
{
	// Open logfile
	common::DeterminismChecker checker = common::DeterminismChecker("determinism", trial);

	// Video data
	string yaml_filename = "../params/videos/holodeck.yaml";
	common::VideoPointData video_data = common::VideoPointData(yaml_filename);

	// Solver
	string solver_yaml = "../params/solvers/gnsac_ptr_eigen.yaml";
	shared_ptr<common::ESolver> solver = common::ESolver::from_yaml(solver_yaml);

	// Init random number generator
	std::default_random_engine rng(0);
	std::normal_distribution<double> dist(0.0, 1.0);
	for(int i = 0; i < 100; i++)
	{
		// First make sure random number generator is deterministic!
		write_check_val(checker, dist(rng));

		// Load real data, undistort points
		common::scan_t pts1, pts2;
		write_check_val(checker, video_data.pts1[i].size());
		write_check_val(checker, video_data.pts2[i].size());
		write_check(checker, (char*)video_data.pts1[i].data(), sizeof(Vector2d) * video_data.pts1[i].size());
		write_check(checker, (char*)video_data.pts2[i].data(), sizeof(Vector2d) * video_data.pts2[i].size());
		common::undistort_points(video_data.pts1[i], pts1, video_data.camera_matrix);
		common::undistort_points(video_data.pts2[i], pts2, video_data.camera_matrix);		
		write_check_val(checker, pts1.size());
		write_check_val(checker, pts2.size());
		write_check(checker, (char*)pts1.data(), sizeof(Vector2d) * pts1.size());
		write_check(checker, (char*)pts2.data(), sizeof(Vector2d) * pts2.size());

		// Calculate essential matrix
		Matrix3d R0 = Matrix3d::Identity();
		Vector3d t0;
		t0 << dist(rng), dist(rng), dist(rng);
		common::EHypothesis initial_guess = common::EHypothesis(Matrix3d::Zero(), R0, t0);
		common::EHypothesis result;
		solver->find_best_hypothesis(pts1, pts2, initial_guess, result);
		write_check(checker, (char*)result.E.data(), sizeof(double)*3*3);
		write_check(checker, (char*)result.R.data(), sizeof(double)*3*3);
		write_check(checker, (char*)result.t.data(), sizeof(double)*3*1);

		// Error
		write_check(checker, (char*)video_data.RT[i].data(), sizeof(double)*4*4);
		Vector2d err = common::err_truth(result.R, result.t, video_data.RT[i]);
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
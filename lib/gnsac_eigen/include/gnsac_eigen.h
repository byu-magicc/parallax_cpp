#ifndef GNSAC_EIGEN_H
#define GNSAC_EIGEN_H

#include "comm_loaders.h"
#include <yaml-cpp/yaml.h>
#include "common.h"
#include "solvers.h"
#include <eigen3/Eigen/Eigen>
#include <fstream>

namespace gnsac_eigen
{

class SO3
{
public:
	SO3();
	SO3(const Eigen::Matrix3d& R);
	Eigen::Matrix3d R;
	void boxplus(const Eigen::Vector3d& delta, SO3& result);
};

class SO2
{
public:
	SO2();
	SO2(const Eigen::Matrix3d& R);
	SO2(const Eigen::Vector3d& v);
	Eigen::Matrix3d R;

	// To get the vector, we use v = R'*[0; 0; 1]. This is the 3rd row of the matrix,
	// Hence to initialize the map we need to use v(R.data() + 2).
	// Changing it directly isn't allowed, because it would alter R.
	const Eigen::Map<Eigen::Matrix<double, 3, 1>, Eigen::Unaligned, Eigen::Stride<1, 3> > v;
	void boxplus(const Eigen::Vector2d& delta, SO2& result);
};

class EManifold
{
public:
	EManifold();
	EManifold(const Eigen::Matrix3d& R, const Eigen::Matrix3d& TR);
	EManifold(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	Eigen::Matrix3d E;
	const Eigen::Map<Eigen::Matrix3d> R;
	const Eigen::Map<Eigen::Matrix3d> TR;
	const Eigen::Map<Eigen::Matrix<double, 3, 1>, Eigen::Unaligned, Eigen::Stride<1, 3> > t;
	void boxplus(const Eigen::Matrix<double, 5, 1>& delta, EManifold& result);
private:
	SO3 rot;
	SO2 vec;
	void updateE();
};

enum_str(optimizer_t, optimizer_t_str, optimizer_GN, optimizer_LM)

enum_str(cost_function_t, cost_function_t_str, cost_algebraic, cost_single, cost_sampson)

enum_str(implementation_t, implementation_t_str, impl_eig, impl_ptr)

enum_str(consensus_t, consensus_t_str, consensus_RANSAC, consensus_LMEDS)

enum_str(initial_guess_t, initial_guess_t_str, init_random, init_previous, init_truth)

enum_str(pose_disambig_t, pose_disambig_t_str, disambig_none, disambig_chierality, disambig_trace)

class GNSAC_Solver
{
public:
	GNSAC_Solver(std::string yaml_filename, YAML::Node node, std::string result_directory);

	void generate_hypotheses(const common::scan_t& subset1, const common::scan_t& subset2, const common::EHypothesis& initial_guess, std::vector<common::EHypothesis>& hypotheses);

	void refine_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& best_hypothesis, common::EHypothesis& result);

	void find_best_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const Eigen::Matrix4d& RT_truth, common::EHypothesis& result);

	double score_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& hypothesis);

private:
	void init_optimizer_log(std::string result_directory);

	void init_comparison_log(std::string result_directory);

public:
	optimizer_t optimizer;
	cost_function_t optimizer_cost;
	cost_function_t scoring_cost;
	implementation_t scoring_impl;
	consensus_t consensus_alg;
	initial_guess_t initial_guess_method;
	pose_disambig_t pose_disambig_method;
	int n_subsets;
	int max_iterations;
	double exit_tolerance;
	double RANSAC_threshold;
	double LM_lambda;
	//GNHypothesis previous_result;

private:
	bool log_optimizer;
	bool log_optimizer_verbose;
	bool log_comparison;
	std::ofstream optimizer_log_file;
	std::ofstream accuracy_log_file;
	std::ofstream comparison_tr_log_file;
	std::ofstream comparison_gn_log_file;
	Eigen::Matrix4d RT_truth;
};

}

#endif //GNSAC_EIGEN_H
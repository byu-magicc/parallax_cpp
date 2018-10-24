
#include <eigen3/Eigen/Dense>
#include <vector>
#include "gnsac_eigen.h"
#include <iostream>
#include <random>
#include <chrono>
#include <experimental/filesystem>

using namespace std;
using namespace Eigen;
namespace fs = std::experimental::filesystem;

namespace gnsac_eigen
{

double sinc(double x)
{
    // Taylor series expansion of sin is:           x - x^3/3! + x^5/5! - ...
    // Thus the Taylor series expansion of sinc is: 1 - x^2/3! + x^4/5! - ...
    // Double precision is approximately 16 digits. Since the largest term is x^2,
    // we will experience numerical underflow if |x| < 1e-8.
    // Of course, floating point arithmetic can handle much smaller values than this (as small as 1e-308).
    // I haven't seen any problems with small numbers so far, so we could just check for division by zero,
    // but this solution is guarenteed not to have problems.
    if (fabs(x) < 1e-8)
        return 1;
    else
        return sin(x) / x;
}

void skew(Vector3d& v, Matrix3d& Tx)
{
	Tx << 0,    -v(2),   v(1),
		  v(2),  0,     -v(0),
		 -v(1),  v(0),   0;
}

void axisAngleGetR(Vector3d& w, Matrix3d& dR)
{
	double theta = w.norm();
	Matrix3d wx;
	skew(w, wx);

	// R = eye(3) + sinc(theta)*wx + 0.5*sinc(theta/2)^2*wx^2;	
	double sinc2 = sinc(theta / 2);
	dR = Matrix3d::Identity() + sinc(theta)*wx + 0.5 * sinc2 * sinc2 * wx * wx;
}

SO3::SO3() : R(Matrix3d::Identity())
{

}

SO3::SO3(Matrix3d& _R) : R(_R)
{

}

void SO3::boxplus(const Eigen::Vector3d& v, SO3& R2)
{
	Matrix3d dR;
	axisAngleGetR(v, dR);
	R2 = dR * R;
}




GNSAC_Solver::GNSAC_Solver(std::string yaml_filename, YAML::Node node, std::string result_directory)
{
	string optimizer_str, optimizer_cost_str, scoring_cost_str, scoring_impl_str, 
		consensus_alg_str, initial_guess_method_str, pose_disambig_str;
	common::get_yaml_node("optimizer", yaml_filename, node, optimizer_str);
	common::get_yaml_node("optimizer_cost", yaml_filename, node, optimizer_cost_str);
	common::get_yaml_node("scoring_cost", yaml_filename, node, scoring_cost_str);
	common::get_yaml_node("scoring_impl", yaml_filename, node, scoring_impl_str);
	common::get_yaml_node("consensus_alg", yaml_filename, node, consensus_alg_str);
	common::get_yaml_node("initial_guess", yaml_filename, node, initial_guess_method_str);
	common::get_yaml_node("n_subsets", yaml_filename, node, n_subsets);
	common::get_yaml_node("max_iterations", yaml_filename, node, max_iterations);
	common::get_yaml_node("exit_tolerance", yaml_filename, node, exit_tolerance);
	common::get_yaml_node("log_optimizer", yaml_filename, node, log_optimizer);
	common::get_yaml_node("log_comparison", yaml_filename, node, log_comparison);
	common::get_yaml_node("pose_disambig", yaml_filename, node, pose_disambig_str);
	cout << "log_optimizer " << log_optimizer << endl;
	cout << "log_comparison " << log_comparison << endl;

	optimizer = (optimizer_t)common::get_enum_from_string(optimizer_t_str, optimizer_str);
	optimizer_cost = (cost_function_t)common::get_enum_from_string(cost_function_t_str, optimizer_cost_str);
	scoring_cost = (cost_function_t)common::get_enum_from_string(cost_function_t_str, scoring_cost_str);
	scoring_impl = (implementation_t)common::get_enum_from_string(implementation_t_str, scoring_impl_str);
	consensus_alg = (consensus_t)common::get_enum_from_string(consensus_t_str, consensus_alg_str);
	initial_guess_method = (initial_guess_t)common::get_enum_from_string(initial_guess_t_str, initial_guess_method_str);
	pose_disambig_method = (pose_disambig_t)common::get_enum_from_string(pose_disambig_t_str, pose_disambig_str);

	if(consensus_alg == consensus_RANSAC)
		common::get_yaml_node("RANSAC_threshold", yaml_filename, node, RANSAC_threshold);
	if(optimizer == optimizer_LM)
		common::get_yaml_node("LM_lambda", yaml_filename, node, LM_lambda);
	if(log_optimizer)
	{
		common::get_yaml_node("log_optimizer_verbose", yaml_filename, node, log_optimizer_verbose);
		cout << "log_optimizer_verbose " << log_optimizer_verbose << endl;
		init_optimizer_log(result_directory);
	}
	if(log_comparison)
		init_comparison_log(result_directory);
}

void GNSAC_Solver::init_optimizer_log(string result_directory)
{
	exit_tolerance = 0; // Data is all garbled if each run has a different number of iterations...
	optimizer_log_file.open(fs::path(result_directory) / "optimizer.bin");
}

void GNSAC_Solver::init_comparison_log(string result_directory)
{
	exit_tolerance = 0;
	accuracy_log_file.open(fs::path(result_directory) / "5-point_accuracy.bin");
	comparison_tr_log_file.open(fs::path(result_directory) / "5-point_comparison_tr.bin");
	comparison_gn_log_file.open(fs::path(result_directory) / "5-point_comparison_gn.bin");
}

void GNSAC_Solver::generate_hypotheses(const common::scan_t& subset1, const common::scan_t& subset2, const common::EHypothesis& initial_guess, std::vector<common::EHypothesis>& hypotheses)
{

}

void GNSAC_Solver::refine_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& best_hypothesis, common::EHypothesis& result)
{

}

void GNSAC_Solver::find_best_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const Eigen::Matrix4d& RT_truth, common::EHypothesis& result)
{

}

double GNSAC_Solver::score_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& hypothesis)
{

}

void run_tests()
{

}

}
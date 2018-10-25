
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

void skew(const Vector3d& v, Matrix3d& Tx)
{
	Tx << 0,    -v(2),   v(1),
		  v(2),  0,     -v(0),
		 -v(1),  v(0),   0;
}

void axisAngleGetR(const Vector3d& w, Matrix3d& dR)
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

SO3::SO3(const Matrix3d& _R) : R(_R)
{

}

void SO3::boxplus(const Eigen::Vector3d& delta, SO3& result)
{
	Matrix3d dR;
	axisAngleGetR(delta, dR);
	result.R = dR * R;
}

SO2::SO2() : R(Matrix3d::Identity()), v(R.data() + 2)
{
	
}

SO2::SO2(const Matrix3d& _R) : R(_R), v(R.data() + 2)
{

}

SO2::SO2(const Vector3d& _v) : v(R.data() + 2)
{
	Vector3d t_unit = unit(_v);
	double theta = acos(t_unit(2));
	double snc = sinc(theta);
	Vector3d w;
	w << t_unit(1)/snc, -t_unit(0)/snc, 0;
	axisAngleGetR(w, R);
}

void SO2::boxplus(const Eigen::Vector2d& delta_, SO2& result)
{
	Matrix3d dR;
	Vector3d delta;
	delta << delta_(0), delta_(1), 0;
	axisAngleGetR(delta, dR);
	result.R = dR * R;
}

EManifold::EManifold() : rot(), vec(), R(rot.R.data()), TR(vec.R.data()), t(vec.v)
{
	updateE();
}

EManifold::EManifold(const Eigen::Matrix3d& _R, const Eigen::Matrix3d& _TR) : rot(_R), vec(_TR), R(rot.R.data()), TR(vec.R.data()), t(vec.v)
{
	updateE();
}

EManifold::EManifold(const Eigen::Matrix3d& _R, const Eigen::Vector3d& _t) : rot(_R), vec(_t), R(rot.R.data()), TR(vec.R.data()), t(vec.v)
{
	updateE();
}

void EManifold::boxplus(const Matrix<double, 5, 1>& delta, EManifold& result)
{
	rot.boxplus(delta.head(3), result.rot);
	vec.boxplus(delta.tail(2), result.vec);
	result.updateE();
}

void EManifold::updateE()
{
	//E = Tx*R;
	Matrix3d Tx;
	skew(t, Tx);
	E = Tx*R;
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
	return 0;

}

void run_tests()
{
	std::default_random_engine rng(0);
	std::normal_distribution<double> dist(0.0, 1.0);
	//std::uniform_int_distribution<> dist(0, pts1.size() - 1);

	for(int i = 0; i < 100; i++)
	{
		// Test axisAngleGetR
		double theta = dist(rng);
		Matrix3d R;
		R << cos(theta), -sin(theta), 0,
		     sin(theta), cos(theta),  0,
		     0,          0,           1;
		Vector3d w;
		w << 0, 0, theta;
		Matrix3d R_;
		axisAngleGetR(w, R_);
		common::checkMatrices("R_cos&sin", "R_axisAngle", R, R_);

		// Boxplus
		// Negation
		common::fill_rnd(w, dist, rng);
		axisAngleGetR(w, R);
		SO3 rot(R), rot2;
		rot.boxplus(-w, rot2);
		common::checkMatrices("R.boxplus(-w)", "Identity", rot2.R, Matrix3d::Identity());

		// Boxplus order
		Vector3d w1, w2;
		common::fill_rnd(w1, dist, rng);
		common::fill_rnd(w2, dist, rng);
		Matrix3d R1, R2;
		axisAngleGetR(w1, R1);
		axisAngleGetR(w2, R2);
		rot = SO3(R1);
		rot.boxplus(w2, rot);
		Matrix3d R12 = R2*R1;
		common::checkMatrices("R1.boxplus(w2)", "R2*R1", rot.R, R12);

		// Unit vector
		Vector3d v;
		common::fill_rnd(v, dist, rng);
		v = unit(v);
		SO2 vec(v);
		common::checkMatrices("SO2(v).v", "v", vec.v, v);

		// Ensure no rotation for unit vector in z direction
		rot = SO3(vec.R);
		w << 0, 0, 1;
		rot.boxplus(w, rot);
		SO2 vec2(rot.R);
		common::checkMatrices("v", "v.boxplus([0; 0; 1])", v, vec2.v);
	}
}

}
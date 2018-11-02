#ifndef GNSAC_PTR_EIG_H
#define GNSAC_PTR_EIG_H

#include "common.h"
#include "solvers.h"
#include <eigen3/Eigen/Dense>
#include <fstream>

namespace gnsac_ptr_eigen
{

double sinc(double x);

void skew(const double* w, double* T);

void mult3pt(const double* A, const Eigen::Vector2d pt, double* y);

void mult31(const double* A, const double* x, double* y);

double mult3vtv(const double* x, const double* y);

double multptv(const Eigen::Vector2d pt, const double* v);

void mult33(const double* A, const double* B, double* C);

double normsqr31(const double* v);

double norm31(const double* v);

void unit31(const double* v, double* v2);

void tran33(const double* A, double* B);

void tran_nxm(const double* A, int n, int m, double* B);

void unitvector_getV(const double* R, double* v);

void t2TR(const double* t, double* TR);

void elementaryVector3(int idx, double* v);

void printArray(const char* s, const double* p, int n, int m, int sfAfter = 4, int sfBefore = 5);

void printPoints(const char* s, common::scan_t pts);

void zero5(double* r);

void eye3(double* I);

void mult3s(const double* A, double s, double* B);

void mult31s(const double* A, double s, double* B);

void add33(const double* A, const double* B, double* C);

void copy33(const double* A, double* B);

void copy31(const double* A, double* B);

void axisAngleGetR(const double* w, double* R);

void R_boxplus(const double* R, const double* w, double* R2);

void E_boxplus(const double* R, const double* TR, const double* dx, double* R2, double* TR2);

void RT_getE(const double* R, const double* TR, double* t, double* E);

void getE_diff(const double* R, const double* TR, double* E, double* deltaE);

void normalizeLine_diff(const double* L, const double* dL, double* LH, double* dLH);

void normalizeLine_diffOnly(const double* L, const double* dL, double* dLH);

void normalizeLine(const double* L, double* LH);

double residual_diff(const double* R, const double* TR, const double* E, const double* deltaE, 
		  const Eigen::Vector2d p1, const Eigen::Vector2d p2, double* deltaR);

double residual(const double* E, const Eigen::Vector2d p1, const Eigen::Vector2d p2);

class GNHypothesis
{
public:
	GNHypothesis();
	GNHypothesis(Eigen::Matrix3d R0, Eigen::Vector3d t0);
	
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_map;
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> TR_map;
	Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> E_map;
	Eigen::Map<Eigen::Matrix<double, 3, 1>> t_map;
	double R[3*3];
	double TR[3*3];
	double E[3*3];
	double t[3*1];
	double cost;
};

void getSubset(const common::scan_t& pts1, const common::scan_t& pts2, common::scan_t& subset1, common::scan_t& subset2, int modelPoints, 
	std::uniform_int_distribution<>& dist, std::default_random_engine& rng);

Eigen::Matrix3d findEssentialMatGN(common::scan_t pts1, common::scan_t pts2,
		Eigen::Matrix3d& R0, Eigen::Vector3d& t0, Eigen::Matrix3d& R2, Eigen::Vector3d& t2,
		int n_hypotheses, int n_GNiters,
		bool withNormalization = true, bool optimizedCost = false);

enum_str(optimizer_t, optimizer_t_vec, optimizer_GN, optimizer_LM)

enum_str(cost_function_t, cost_function_t_vec, cost_algebraic, cost_single, cost_sampson)

enum_str(implementation_t, implementation_t_vec, impl_eig, impl_ptr)

enum_str(consensus_t, consensus_t_vec, consensus_RANSAC, consensus_LMEDS)

enum_str(initial_guess_t, initial_guess_t_vec, init_random, init_previous, init_truth)

enum_str(pose_disambig_t, pose_disambig_t_vec, disambig_none, disambig_chierality, disambig_trace)

class GNSAC_Solver : public common::ESolver
{
public:
	GNSAC_Solver(std::string yaml_filename, YAML::Node node);

public:
	void generate_hypotheses(const common::scan_t& subset1, const common::scan_t& subset2, const common::EHypothesis& initial_guess, std::vector<common::EHypothesis>& hypotheses);

	void refine_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& best_hypothesis, common::EHypothesis& result);

	void find_best_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const Eigen::Matrix4d& RT_truth, common::EHypothesis& result);

	double score_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& hypothesis);

private:
	double step(const common::scan_t& pts1, const common::scan_t& pts2, 
		const GNHypothesis& h1, GNHypothesis& h2, double& lambda, bool last_iteration, double residual_norm);

	int optimize(const common::scan_t& pts1, const common::scan_t& pts2, const GNHypothesis& h1, GNHypothesis& h2);

	double score_single_ptr(const Eigen::Vector2d& pt1, const Eigen::Vector2d& pt2, const GNHypothesis& hypothesis);

	double score_sampson_eig(const Eigen::Vector2d& pt1, const Eigen::Vector2d& pt2, const GNHypothesis& hypothesis);

	double score_sampson_ptr(const Eigen::Vector2d& pt1, const Eigen::Vector2d& pt2, const GNHypothesis& hypothesis);

	double score(const common::scan_t& pts1, const common::scan_t& pts2, GNHypothesis hypothesis, double best_cost);

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
	GNHypothesis previous_result;

private:
	Eigen::Matrix4d RT_truth;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif //GNSAC_PTR_EIG_H
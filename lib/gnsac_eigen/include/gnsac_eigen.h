#ifndef GNSAC_EIGEN_H
#define GNSAC_EIGEN_H

#include "comm_loaders.h"
#include <yaml-cpp/yaml.h>
#include "common.h"
#include "solvers.h"
#include <eigen3/Eigen/Eigen>
#include <fstream>
#include <memory>

namespace gnsac_eigen
{

enum_str(optimizer_t, optimizer_t_vec, optimizer_GN, optimizer_LM)

enum_str(cost_function_t, cost_function_t_vec, cost_algebraic, cost_single, cost_sampson)

enum_str(implementation_t, implementation_t_vec, impl_eig, impl_ptr)

enum_str(consensus_t, consensus_t_vec, consensus_RANSAC, consensus_LMEDS)

enum_str(initial_guess_t, initial_guess_t_vec, init_random, init_previous, init_truth)

enum_str(pose_disambig_t, pose_disambig_t_vec, disambig_none, disambig_chierality, disambig_trace)

const int MAX_PTS = 2000;

///////////////////////
// Manifold Elements //
///////////////////////

class SO3
{
public:
	SO3();
	SO3(const Eigen::Matrix3d& R);
	SO3(const SO3&) = delete;
	Eigen::Matrix3d R;
	SO3& operator= (const SO3& other);
	void boxplus(const Eigen::Vector3d& delta, SO3& result) const;
	void derivative(int i, Eigen::Matrix3d& result) const;
};

class SO2
{
public:
	SO2();
	SO2(const Eigen::Matrix3d& R);
	SO2(const Eigen::Vector3d& v);
	SO2(const SO2&) = delete;
	Eigen::Matrix3d R;
	SO2& operator= (const SO2& other);

	// To get the vector, we use v = R'*[0; 0; 1]. This is the 3rd row of the matrix,
	// Hence to initialize the map we need to use v(R.data() + 2).
	// Changing it directly isn't allowed, because it would alter R.
	const Eigen::Map<const Eigen::Matrix<double, 3, 1>, Eigen::Unaligned, Eigen::Stride<1, 3> > v;
	void boxplus(const Eigen::Vector2d& delta, SO2& result) const;
	void derivative(int i, Eigen::Vector3d& result) const;
};

class EManifold
{

public:
	EManifold();
	EManifold(const Eigen::Matrix3d& R, const Eigen::Matrix3d& TR);
	EManifold(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	EManifold(const EManifold&) = delete;
	EManifold& operator= (const EManifold& other);
	void setR(Eigen::Matrix3d R);
	void setT(Eigen::Vector3d t);
	void setTR(Eigen::Matrix3d TR);
	void boxplus(const Eigen::Matrix<double, 5, 1>& delta, EManifold& result) const;
	void derivative(int i, Eigen::Matrix3d& result) const;
private:
	void updateE();
	
// Note: This is not as appealing to put private member variables before the public ones,
// but if we want to initialize our eigen references in the constructor, their dependencies 
// have to be listed first so they will be initialized first.
// See https://stackoverflow.com/questions/4037219/order-of-execution-in-constructor-initialization-list
protected:
	SO3 rot;
	SO2 vec;
	Eigen::Matrix3d E_;

public:
	const Eigen::Matrix3d& E;
	const Eigen::Matrix3d& R;
	const Eigen::Matrix3d& TR;
	const Eigen::Map<const Eigen::Matrix<double, 3, 1>, Eigen::Unaligned, Eigen::Stride<1, 3> >& t;
};

////////////////////////////////
// Residual or Cost Functions //
////////////////////////////////

class DifferentiableResidual
{
public:
	virtual void residual(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err) = 0;

	virtual void residual_sqr(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err) = 0;

	virtual void residual_diff(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err, Eigen::Map<Eigen::MatrixXd>& jacobian) = 0;

	static std::shared_ptr<DifferentiableResidual> from_enum(cost_function_t cost_fcn);

protected:
	DifferentiableResidual();
};

class AlgebraicResidual : public DifferentiableResidual
{
public:
	AlgebraicResidual();

	void residual(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err);

	void residual_sqr(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err);

	void residual_diff(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err, Eigen::Map<Eigen::MatrixXd>& jacobian);
};

class SingleImageResidual : public DifferentiableResidual
{
public:
	SingleImageResidual();

	void residual(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err);

	void residual_sqr(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err);

	void residual_diff(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& err, Eigen::Map<Eigen::MatrixXd>& jacobian);
};

class SampsonResidual : public DifferentiableResidual
{
public:
	SampsonResidual();

	void residual(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& residual);

	void residual_sqr(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& residual);

	void residual_diff(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Eigen::Map<Eigen::VectorXd>& residual, Eigen::Map<Eigen::MatrixXd>& jacobian);
};

template <typename _Function, typename _X, int _Rows, int _Cols>
void numericalDerivative(_Function fcn, _X& x, Eigen::Matrix<double, _Rows, _Cols>& J)
{
	double h = 1e-8;
	for(int i = 0; i < _Cols; i++)
	{
		Eigen::Matrix<double, _Cols, 1> dx = Eigen::Matrix<double, _Cols, 1>::Zero();
		dx(i) = h/2;
		_X x1, x2;
		x.boxplus(-dx, x1);
		x.boxplus(dx, x2);
		J.col(i) = (fcn(x2) - fcn(x1)) / h;
	}
}

template <int _InputRows, typename _Function, typename _X, int _Rows, int _Cols>
void numericalDerivative_i(_Function fcn, _X& x, Eigen::Matrix<double, _Rows, _Cols>& J, int i)
{
	double h = 1e-8;
	Eigen::Matrix<double, _InputRows, 1> dx = Eigen::Matrix<double, _InputRows, 1>::Zero();
	dx(i) = h/2;
	_X x1, x2;
	x.boxplus(-dx, x1);
	x.boxplus(dx, x2);
	J = (fcn(x2) - fcn(x1)) / h;
}

////////////////
// Optimizers //
////////////////

class Optimizer
{
protected:
	Optimizer(std::shared_ptr<DifferentiableResidual> residual, int maxIterations, int exitTolerance);

public:
	virtual void optimize(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& initialGuess, EManifold& result) = 0;

protected:
	std::shared_ptr<DifferentiableResidual> residual_fcn;
	int maxIterations;
public:
	int exitTolerance;
};

class GaussNewton : public Optimizer
{
public:
	GaussNewton(std::shared_ptr<DifferentiableResidual> residual, int maxIterations, int exitTolerance);

	void optimize(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& initialGuess, EManifold& result);

private:
	double r_buf[MAX_PTS*1];
	double J_buf[MAX_PTS*5];
};

class LevenbergMarquardt : public Optimizer
{
public:
	LevenbergMarquardt(std::shared_ptr<DifferentiableResidual> residual, int maxIterations, int exitTolerance, double lambda0);

	void optimize(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& initialGuess, EManifold& result);

private:
	double lambda0;
	double r_buf[MAX_PTS*1];
	double J_buf[MAX_PTS*5];
};

//////////////////////////
// Consensus Algorithms //
//////////////////////////

class ConsensusAlgorithm
{
protected:
	ConsensusAlgorithm(std::shared_ptr<Optimizer> optimizer, int n_subsets, bool seedWithBestHypothesis);

public:
	void run(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& initialGuess, EManifold& bestModel);

	virtual double score(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& hypothesis, double best_cost) = 0;

private:
	void getSubset(const common::scan_t& pts1, const common::scan_t& pts2, common::scan_t& subset1, common::scan_t& subset2, int modelPoints, 
		std::uniform_int_distribution<>& dist, std::default_random_engine& rng);
	int n_subsets;
	bool seedWithBestHypothesis;
	std::shared_ptr<Optimizer> optimizer;
};

class RANSAC_Algorithm : public ConsensusAlgorithm
{
public:
	RANSAC_Algorithm(std::shared_ptr<Optimizer> optimizer, int n_subsets, bool seedWithBestHypothesis, std::shared_ptr<DifferentiableResidual> cost_fcn, double threshold);

	double score(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& hypothesis, double best_cost);

private:
	std::shared_ptr<DifferentiableResidual> residual_fcn;
	double threshold;
	double err_buf[MAX_PTS*1];
};

class LMEDS_Algorithm : public ConsensusAlgorithm
{
public:
	LMEDS_Algorithm(std::shared_ptr<Optimizer> optimizer, int n_subsets, bool seedWithBestHypothesis, std::shared_ptr<DifferentiableResidual> cost_fcn);

	double score(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& hypothesis, double best_cost);

private:
	std::shared_ptr<DifferentiableResidual> residual_fcn;
	double err_buf[MAX_PTS*1];
};

////////////////////////////
// Top-level GNSAC_Solver //
////////////////////////////

class GNSAC_Solver : public common::ESolver
{
public:
	GNSAC_Solver(std::string yaml_filename, YAML::Node node);

	void generate_hypotheses(const common::scan_t& subset1, const common::scan_t& subset2, const common::EHypothesis& initial_guess, std::vector<common::EHypothesis>& hypotheses);

	void refine_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& best_hypothesis, common::EHypothesis& result);

	void find_best_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const Eigen::Matrix4d& RT_truth, common::EHypothesis& result);

	double score_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& hypothesis);

public:
	std::shared_ptr<Optimizer> optimizer;
	std::shared_ptr<ConsensusAlgorithm> consensusAlg;
	std::shared_ptr<DifferentiableResidual> optimizerCost;
	std::shared_ptr<DifferentiableResidual> scoringCost;
	initial_guess_t initialGuessMethod;
	pose_disambig_t poseDisambigMethod;

private:
	std::ofstream optimizer_log_file;
	std::ofstream accuracy_log_file;
	std::ofstream comparison_tr_log_file;
	std::ofstream comparison_gn_log_file;
	//Eigen::Matrix4d RT_truth;
	EManifold prevResult;
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW	
};

void run_tests();

void run_jacobian_tests();

void run_optimizer_tests();

}

#endif //GNSAC_EIGEN_H
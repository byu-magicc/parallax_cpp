#include <eigen3/Eigen/Dense>
#include <vector>
#include "gnsac_eigen.h"
#include <iostream>
#include <random>
#include <chrono>
#include <experimental/filesystem>
#include "common.h"

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

SO3& SO3::operator= (const SO3& other)
{
	if(&other == this)
		return *this;
	R = other.R;
	return *this;
}

void SO3::boxplus(const Eigen::Vector3d& delta, SO3& result) const
{
	Matrix3d dR;
	axisAngleGetR(delta, dR);
	result.R = dR * R;
}

void SO3::derivative(int i, Matrix3d& result) const
{
	// genR = skew(elementaryVector(3, i));
	Vector3d delta = Vector3d::Zero();
	delta(i) = 1;
	Matrix3d genR;
	skew(delta, genR);

	// result = genR * R
	result = genR * R;
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

SO2& SO2::operator= (const SO2& other)
{
	if(&other == this)
		return *this;
	R = other.R;
	return *this;
}

void SO2::boxplus(const Eigen::Vector2d& delta_, SO2& result) const
{
	Matrix3d dR;
	Vector3d delta;
	delta << delta_(0), delta_(1), 0;
	axisAngleGetR(delta, dR);
	result.R = dR * R;
}

void SO2::derivative(int i, Vector3d& result) const
{
	// genR = skew(elementaryVector(3, i));
	Vector3d delta = Vector3d::Zero();
	delta(i) = 1;
	Matrix3d genR;
	skew(delta, genR);

	// result = (genTR * R).' * [0; 0; 1]
	result = (genR * R).transpose().col(2);
}

EManifold::EManifold() : rot(), vec(), E(E_), R(rot.R), TR(vec.R), t(vec.v)
{
	release_assert(t.data() == vec.v.data());
	updateE();
}

// Note: The order of initialization is very important here, but it has nothing to do with
// the order the variables appear in the member initializer list. We have to make sure we
// have the correct order in the class definition!
// See https://stackoverflow.com/questions/4037219/order-of-execution-in-constructor-initialization-list
EManifold::EManifold(const Eigen::Matrix3d& _R, const Eigen::Matrix3d& _TR) : rot(_R), vec(_TR), E(E_), R(rot.R), TR(vec.R), t(vec.v)
{
	updateE();
}

EManifold::EManifold(const Eigen::Matrix3d& _R, const Eigen::Vector3d& _t) : rot(_R), vec(_t), E(E_), R(rot.R), TR(vec.R), t(vec.v)
{
	updateE();
}

EManifold& EManifold::operator= (const EManifold& other)
{
	if(&other == this)
		return *this;
	E_ = other.E_;
	rot = other.rot;
	vec = other.vec;
	return *this;
}

void EManifold::setR(Matrix3d R)
{
	rot.R = R;
	updateE();
}

void EManifold::setT(Vector3d t)
{
	SO2 vec2(t);
	vec.R = vec2.R;
	updateE();
}

void EManifold::setTR(Matrix3d TR)
{
	vec.R = TR;
	updateE();
}

void EManifold::boxplus(const Matrix<double, 5, 1>& delta, EManifold& result) const
{
	rot.boxplus(delta.head(3), result.rot);
	vec.boxplus(delta.tail(2), result.vec);
	result.updateE();
}

void EManifold::updateE()
{
	release_assert(R.data() == rot.R.data());
	release_assert(TR.data() == vec.R.data());
	release_assert(t.data() == vec.v.data());

	//E = Tx*R;
	Matrix3d Tx;
	skew(t, Tx);
	E_ = Tx*R;
}

void EManifold::derivative(int i, Matrix3d& result) const
{
	if(i >= 0 && i < 3)
	{
		// deltaE = Tx * derivR;
		Matrix3d Tx;
		skew(t, Tx);
		Matrix3d derivR;
		rot.derivative(i, derivR);
		result = Tx * derivR;
	}
	else if(i >= 3 && i < 5)
	{
		// deltaE = skew(derivTR)*R;
		Vector3d derivT;
		vec.derivative(i - 3, derivT);
		Matrix3d derivT_skew;
		skew(derivT, derivT_skew);
		result = derivT_skew * R;
	}
	else
	{
		printf("Error: Cannot calculate derivative in direction %d.", i);
		exit(EXIT_FAILURE);
	}
}

////////////////////////////////
// Residual or Cost Functions //
////////////////////////////////

DifferentiableResidual::DifferentiableResidual()
{
	
}

shared_ptr<DifferentiableResidual> DifferentiableResidual::from_enum(cost_function_t cost_fcn)
{
	if (cost_fcn == cost_algebraic)
	{
		shared_ptr<AlgebraicResidual> ptr1 = make_shared<AlgebraicResidual>();
		return dynamic_pointer_cast<DifferentiableResidual>(ptr1);
	}
	else if (cost_fcn == cost_single)
	{
		shared_ptr<SingleImageResidual> ptr1 = make_shared<SingleImageResidual>();
		return dynamic_pointer_cast<DifferentiableResidual>(ptr1);
	}
	else if (cost_fcn == cost_sampson)
	{
		shared_ptr<SampsonResidual> ptr1 = make_shared<SampsonResidual>();
		return dynamic_pointer_cast<DifferentiableResidual>(ptr1);
	}
	else
	{
		cout << "Cost function enum " << cost_fcn << " not recognized" << endl;
		exit(EXIT_FAILURE);
	}	
}

// Algebraic Residual
AlgebraicResidual::AlgebraicResidual()
{

}

void AlgebraicResidual::residual(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err)
{
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = eManifold.E * x1;
		double x2t_E_x1 = x2.dot(E_x1);
		err(i) = x2t_E_x1;
	}
}

void AlgebraicResidual::residual_sqr(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err)
{
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = eManifold.E * x1;
		double x2t_E_x1 = x2.dot(E_x1);
		err(i) = x2t_E_x1 * x2t_E_x1;
	}
}

void AlgebraicResidual::residual_diff(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err, Map<MatrixXd>& jacobian)
{
	const Matrix3d& E = eManifold.E;
	Matrix3d deltaE[5];
	for(int j = 0; j < 5; j++)
		eManifold.derivative(j, deltaE[j]);
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = E * x1;
		double x2t_E_x1 = x2.dot(E_x1);
		err(i) = x2t_E_x1;
		for(int j = 0; j < 5; j++)
		{
			Vector3d dE_x1 = deltaE[j] * x1;
			Vector3d dEt_x2 = deltaE[j].transpose() * x2;
			double x2t_dE_x1 = x2.dot(dE_x1);
			jacobian(i, j) = x2t_dE_x1;
		}
	}
}

// Single Image Residual
SingleImageResidual::SingleImageResidual()
{

}

void SingleImageResidual::residual(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err)
{
	const Matrix3d& E = eManifold.E;
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = E * x1;
		double x2t_E_x1 = x2.dot(E_x1);
		double a = E_x1(0);
		double b = E_x1(1);
		double sum = a*a + b*b;
		double sqrt_sum = sqrt(sum);
		err(i) = x2t_E_x1 / sqrt_sum;
	}
}

void SingleImageResidual::residual_sqr(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err)
{
	const Matrix3d& E = eManifold.E;
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = E * x1;
		double x2t_E_x1 = x2.dot(E_x1);
		double a = E_x1(0);
		double b = E_x1(1);
		double sum = a*a + b*b;
		err(i) = x2t_E_x1 * x2t_E_x1 / sum;
	}
}

void SingleImageResidual::residual_diff(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err, Map<MatrixXd>& jacobian)
{
	const Matrix3d& E = eManifold.E;
	Matrix3d deltaE[5];
	for(int j = 0; j < 5; j++)
		eManifold.derivative(j, deltaE[j]);
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = E * x1;
		double x2t_E_x1 = x2.dot(E_x1);
		double a = E_x1(0);
		double b = E_x1(1);
		double sum = a*a + b*b;
		double sqrt_sum = sqrt(sum);
		err(i) = x2t_E_x1 / sqrt_sum;
		for(int j = 0; j < 5; j++)
		{
			// Observe that the bottom part of the fraction is a scalar, so we can take the derivative using the quotient rule
			Vector3d dE_x1 = deltaE[j] * x1;
			double x2t_dE_x1 = x2.dot(dE_x1);
			double dA = dE_x1(0);
			double dB = dE_x1(1);
			double dSum = a*dA + b*dB;
			jacobian(i, j) = (sqrt_sum * x2t_dE_x1 - x2t_E_x1 / sqrt_sum * dSum) / sum;
		}
	}

}

// Sampson Residual
SampsonResidual::SampsonResidual()
{

}

void SampsonResidual::residual(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err)
{
	const Matrix3d& E = eManifold.E;
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = E * x1;
		Vector3d Et_x2 = E.transpose() * x2;
		double x2t_E_x1 = x2.dot(E_x1);
		double a = E_x1(0);
		double b = E_x1(1);
		double c = Et_x2(0);
		double d = Et_x2(1);
		double sum = a*a + b*b + c*c + d*d;
		double sqrt_sum = sqrt(sum);
		err(i) = x2t_E_x1 / sqrt_sum;
	}
}

void SampsonResidual::residual_sqr(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err)
{
	const Matrix3d& E = eManifold.E;
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = E * x1;
		Vector3d Et_x2 = E.transpose() * x2;
		double x2t_E_x1 = x2.dot(E_x1);
		double a = E_x1(0);
		double b = E_x1(1);
		double c = Et_x2(0);
		double d = Et_x2(1);
		double sum = a*a + b*b + c*c + d*d;
		err(i) = x2t_E_x1*x2t_E_x1 / sum;
	}
}

void SampsonResidual::residual_diff(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& eManifold, Map<VectorXd>& err, Map<MatrixXd>& jacobian)
{
	const Matrix3d& E = eManifold.E;
	Matrix3d deltaE[5];
	for(int j = 0; j < 5; j++)
		eManifold.derivative(j, deltaE[j]);
	for(int i = 0; i < pts1.size(); i++)
	{
		Vector3d x1, x2;
		x1 << pts1[i](0), pts1[i](1), 1;
		x2 << pts2[i](0), pts2[i](1), 1;
		Vector3d E_x1 = E * x1;
		Vector3d Et_x2 = E.transpose() * x2;
		double x2t_E_x1 = x2.dot(E_x1);
		double a = E_x1(0);
		double b = E_x1(1);
		double c = Et_x2(0);
		double d = Et_x2(1);
		double sum = a*a + b*b + c*c + d*d;
		double sqrt_sum = sqrt(sum);
		err(i) = x2t_E_x1 / sqrt_sum;
		for(int j = 0; j < 5; j++)
		{
			// Observe that the bottom part of the fraction is a scalar, so we can take the derivative using the quotient rule
			Vector3d dE_x1 = deltaE[j] * x1;
			Vector3d dEt_x2 = deltaE[j].transpose() * x2;
			double x2t_dE_x1 = x2.dot(dE_x1);
			double dA = dE_x1(0);
			double dB = dE_x1(1);
			double dC = dEt_x2(0);
			double dD = dEt_x2(1);
			double dSum = a*dA + b*dB + c*dC + d*dD;
			jacobian(i, j) = (sqrt_sum * x2t_dE_x1 - x2t_E_x1 / sqrt_sum * dSum) / sum;
		}
	}
}

////////////////
// Optimizers //
////////////////

Optimizer::Optimizer(std::shared_ptr<DifferentiableResidual> _residual, int _maxIterations, int _exitTolerance) :
	residual_fcn(_residual), maxIterations(_maxIterations), exitTolerance(_exitTolerance)
{

}

GaussNewton::GaussNewton(std::shared_ptr<DifferentiableResidual> _residual, int _maxIterations, int _exitTolerance) : Optimizer(_residual, _maxIterations, _exitTolerance)
{

}

void GaussNewton::optimize(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& initialGuess, EManifold& result)
{
	Map<VectorXd> r = Map<VectorXd>(r_buf, (int)pts1.size());
	Map<MatrixXd> J = Map<MatrixXd>(J_buf, (int)pts2.size(), 5);
	result = initialGuess;
	for(int i = 0; i < maxIterations; i++)
	{
		time_cat_verbose(common::TimeCatVerboseMakeJ);
		residual_fcn->residual_diff(pts1, pts2, result, r, J);

		// Todo: Figure out how to use matrix workspaces
		time_cat_verbose(common::TimeCatVerboseSolveMatrix);
		Matrix<double, 5, 1> dx = -J.fullPivLu().solve(r);

		time_cat_verbose(common::TimeCatVerboseManifoldUpdate);
		result.boxplus(dx, result);

		time_cat_verbose(common::TimeCatVerboseCalcResidual);
		residual_fcn->residual(pts1, pts2, result, r);
		double residual_norm = r.norm();
		double dx_norm = dx.norm();

		// time_cat(common::TimeCatNone);
		// double lambda = -1;
		// int attempts = -1;
		// common::write_log(common::log_optimizer, (char*)&residual_norm, sizeof(double));
		// common::write_log(common::log_optimizer, (char*)&dx_norm, sizeof(double));
		// common::write_log(common::log_optimizer, (char*)&lambda, sizeof(double));
		// common::write_log(common::log_optimizer, (char*)&attempts, sizeof(double));
		// time_cat(common::TimeCatHypoGen);

		if (residual_norm < exitTolerance)
			break;
	}
}

LevenbergMarquardt::LevenbergMarquardt(std::shared_ptr<DifferentiableResidual> _residual, int _maxIterations, int _exitTolerance, double _lambda0) : Optimizer(_residual, _maxIterations, _exitTolerance), lambda0(_lambda0)
{

}

void LevenbergMarquardt::optimize(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& initialGuess, EManifold& result)
{
	Map<VectorXd> r = Map<VectorXd>(r_buf, (int)pts1.size());
	Map<MatrixXd> J = Map<MatrixXd>(J_buf, (int)pts2.size(), 5);
	Matrix<double, 5, 1> dx;
	double r_norm;
	double lambda = lambda0;
	result = initialGuess;
	EManifold tmp;
	for(int i = 0; i < maxIterations; i++)
	{
		time_cat_verbose(common::TimeCatVerboseMakeJ);
		residual_fcn->residual_diff(pts1, pts2, result, r, J);
		r_norm = r.norm();

		// JtJ = J'*J;
		// Jtr = J'*r;
		Matrix<double, 5, 5> JtJ = J.transpose()*J;
		Matrix<double, 5, 1> Jtr = J.transpose()*r;
		int attempts = 0;
		while(true)
		{		
			// dx = -(JtJ + lambda*diag(diag(JtJ)))\J'*r;
			// Todo: Figure out how to use matrix workspaces
			time_cat_verbose(common::TimeCatVerboseSolveMatrix);
			Matrix<double, 5, 5> JtJ_diag_only = JtJ.diagonal().asDiagonal();
			dx = -(JtJ + lambda*JtJ_diag_only).fullPivLu().solve(Jtr);

			time_cat_verbose(common::TimeCatVerboseManifoldUpdate);
			result.boxplus(dx, tmp);

			time_cat_verbose(common::TimeCatVerboseCalcResidual);
			residual_fcn->residual(pts1, pts2, tmp, r);
			double r_norm_tmp = r.norm();
			if(r_norm_tmp <= r_norm || lambda > 1e30)
			{
				// If the new error is lower, keep new values and move onto the next iteration.
				// Decrease lambda, which makes the algorithm behave more like Gauss-Newton.
				// Gauss-Newton gives very fast convergence when the function is well-behaved.
				result = tmp;
				lambda = lambda / 2;
				r_norm = r_norm_tmp;
				break;
			}
			else
			{
				// If the new error is higher, discard new values and try again. 
				// This time increase lambda, which makes the algorithm behave more
				// like gradient descent and decreases the step size.
				// Keep trying until we succeed at decreasing the error.
				lambda = lambda * 2;
				attempts++;
			}
		}
		double dx_norm = dx.norm();

		// double attempts_double = attempts;
		// common::write_log(common::log_optimizer, (char*)&r_norm, sizeof(double));
		// common::write_log(common::log_optimizer, (char*)&dx_norm, sizeof(double));
		// common::write_log(common::log_optimizer, (char*)&lambda, sizeof(double));
		// common::write_log(common::log_optimizer, (char*)&attempts_double, sizeof(double));
		// time_cat(common::TimeCatHypoGen);

		if (r_norm < exitTolerance)
			break;
	}
}

//////////////////////////
// Consensus Algorithms //
//////////////////////////

ConsensusAlgorithm::ConsensusAlgorithm(std::shared_ptr<Optimizer> _optimizer, int _n_subsets, bool _seedWithBestHypothesis) : optimizer(_optimizer), n_subsets(_n_subsets), seedWithBestHypothesis(_seedWithBestHypothesis)
{

}

void ConsensusAlgorithm::run(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& initialGuess, EManifold& bestModel)
{
	std::default_random_engine rng(0);
	std::uniform_int_distribution<> dist(0, pts1.size() - 1);

	// Fully score initial hypothesis
	time_cat(common::TimeCatHypoScoring);
	double bestCost = score(pts1, pts2, initialGuess, 1e10);
	EManifold model;
	for(int i = 0; i < n_subsets; i++)
	{
		// Get subset
		time_cat(common::TimeCatHypoGen);
		common::scan_t subset1;
		common::scan_t subset2;
		getSubset(pts1, pts2, subset1, subset2, 5, dist, rng);

		// Generate a hypothesis using subset
		if(seedWithBestHypothesis)
			optimizer->optimize(subset1, subset2, bestModel, model);
		else
			optimizer->optimize(subset1, subset2, initialGuess, model);

		// Partially score hypothesis (terminate early if cost exceeds lowest cost)
		time_cat(common::TimeCatHypoScoring);
		double cost = score(pts1, pts2, model, bestCost);
		if(cost < bestCost)
		{
			bestModel = model;
			bestCost = cost;
		}
	}
}

void ConsensusAlgorithm::getSubset(const common::scan_t& pts1, const common::scan_t& pts2, common::scan_t& subset1, common::scan_t& subset2, int modelPoints, 
	std::uniform_int_distribution<>& dist, std::default_random_engine& rng)
{
	int count = pts1.size();
	vector<int> idxs;
	for (int i = 0; i < modelPoints; i++)
	{
		int idx = 0;
		bool unique = false;
		while(!unique)
		{
			// randomly pick an element index to add to the subset
			idx = dist(rng);

			// ensure element is unique
			unique = true;
			for (int j = 0; j < i; j++)
				if (idx == idxs[j])
				{
					unique = false;
					break;
				}
		}
		
		// add element at index
		idxs.push_back(idx);
		subset1.push_back(pts1[idx]);
		subset2.push_back(pts2[idx]);
	}
}

RANSAC_Algorithm::RANSAC_Algorithm(std::shared_ptr<Optimizer> _optimizer, int _n_subsets, bool _seedWithBestHypothesis, std::shared_ptr<DifferentiableResidual> _cost_fcn, double _threshold) :
	ConsensusAlgorithm(_optimizer, _n_subsets, _seedWithBestHypothesis), residual_fcn(_cost_fcn), threshold(_threshold)
{

}

double RANSAC_Algorithm::score(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& hypothesis, double best_cost)
{
	int n_pts = pts1.size();
	Map<VectorXd> err(err_buf, n_pts, 1);
	double RANSAC_threshold_sqr = threshold * threshold;
	residual_fcn->residual_sqr(pts1, pts2, hypothesis, err);
	double cost = 0;
	for(int i = 0; i < n_pts; i++)
		if(err_buf[i] > RANSAC_threshold_sqr)
			cost++;
	return cost;
}

LMEDS_Algorithm::LMEDS_Algorithm(std::shared_ptr<Optimizer> _optimizer, int _n_subsets, bool _seedWithBestHypothesis, std::shared_ptr<DifferentiableResidual> _cost_fcn) :
	ConsensusAlgorithm(_optimizer, _n_subsets, _seedWithBestHypothesis), residual_fcn(_cost_fcn)
{

}

double LMEDS_Algorithm::score(const common::scan_t& pts1, const common::scan_t& pts2, const EManifold& hypothesis, double best_cost)
{
	int n_pts = pts1.size();
	Map<VectorXd> err(err_buf, n_pts, 1);
	residual_fcn->residual_sqr(pts1, pts2, hypothesis, err);
	int medianIdx = (int)n_pts / 2;
	std::nth_element(err_buf, err_buf + medianIdx, err_buf + n_pts);
	return err_buf[medianIdx];
}

////////////////////////////
// Top-level GNSAC_Solver //
////////////////////////////

GNSAC_Solver::GNSAC_Solver(std::string yaml_filename, YAML::Node node, std::string result_directory) : common::ESolver(yaml_filename, node, result_directory)
{
	// Optimizer cost
	string optimizer_cost_str;
	common::get_yaml_node("optimizer_cost", yaml_filename, node, optimizer_cost_str);
	cost_function_t opt_cost = (cost_function_t)common::get_enum_from_string(cost_function_t_vec, optimizer_cost_str);
	optimizerCost = DifferentiableResidual::from_enum(opt_cost);

	// Optimizer
	string optimizer_str;
	int maxIterations;
	double exitTolerance;
	common::get_yaml_node("optimizer", yaml_filename, node, optimizer_str);
	common::get_yaml_node("max_iterations", yaml_filename, node, maxIterations);
	common::get_yaml_node("exit_tolerance", yaml_filename, node, exitTolerance);
	optimizer_t optim = (optimizer_t)common::get_enum_from_string(optimizer_t_vec, optimizer_str);
	if(optim == optimizer_GN)
		optimizer = make_shared<GaussNewton>(optimizerCost, maxIterations, exitTolerance);
	else if(optim == optimizer_LM)
	{
		double lambda0;
		common::get_yaml_node("LM_lambda", yaml_filename, node, lambda0);	
		optimizer = make_shared<LevenbergMarquardt>(optimizerCost, maxIterations, exitTolerance, lambda0);
	}
	else
	{
		printf("Optimizer enum value %d not recognized.", (int)optim);
		exit(EXIT_FAILURE);
	}

	// Consensus cost
	string scoring_cost_str;
	common::get_yaml_node("scoring_cost", yaml_filename, node, scoring_cost_str);
	cost_function_t score_cost = (cost_function_t)common::get_enum_from_string(cost_function_t_vec, scoring_cost_str);
	scoringCost = DifferentiableResidual::from_enum(score_cost);

	// Consensus Algorithm
	string consensus_alg_str;
	int n_subsets;
	bool consensus_seed_best;
	common::get_yaml_node("n_subsets", yaml_filename, node, n_subsets);
	common::get_yaml_node("consensus_alg", yaml_filename, node, consensus_alg_str);
	common::get_yaml_node("consensus_seed_best", yaml_filename, node, consensus_seed_best);
	consensus_t consen_alg = (consensus_t)common::get_enum_from_string(consensus_t_vec, consensus_alg_str);
	if(consen_alg == consensus_RANSAC)
	{
		double RANSAC_threshold;
		common::get_yaml_node("RANSAC_threshold", yaml_filename, node, RANSAC_threshold);
		consensusAlg = make_shared<RANSAC_Algorithm>(optimizer, n_subsets, consensus_seed_best, scoringCost, RANSAC_threshold);
	}
	else if(consen_alg == consensus_LMEDS)
		consensusAlg = make_shared<LMEDS_Algorithm>(optimizer, n_subsets, consensus_seed_best, scoringCost);
	else
	{
		printf("Consensus algorithm enum value %d not recognized.", (int)optim);
		exit(EXIT_FAILURE);
	}

	// Other parameters
	string initial_guess_method_str, pose_disambig_str;
	common::get_yaml_node("initial_guess", yaml_filename, node, initial_guess_method_str);
	common::get_yaml_node("pose_disambig", yaml_filename, node, pose_disambig_str);
	common::get_yaml_node("log_optimizer", yaml_filename, node, log_optimizer);
	common::get_yaml_node("log_comparison", yaml_filename, node, log_comparison);
	initialGuessMethod = (initial_guess_t)common::get_enum_from_string(initial_guess_t_vec, initial_guess_method_str);
	poseDisambigMethod = (pose_disambig_t)common::get_enum_from_string(pose_disambig_t_vec, pose_disambig_str);
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
	optimizer->exitTolerance = 0; // Data is all garbled if each run has a different number of iterations...
	optimizer_log_file.open(fs::path(result_directory) / "optimizer.bin");
}

void GNSAC_Solver::init_comparison_log(string result_directory)
{
	optimizer->exitTolerance = 0; // Data is all garbled if each run has a different number of iterations...
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
	// Init
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(0);
	std::uniform_int_distribution<> dist(0, pts1.size() - 1);
	EManifold initialGuess;
	if(initialGuessMethod == init_previous)
		initialGuess = prevResult;
	else if(initialGuessMethod == init_random)
	{
		Matrix3d R0 = Matrix3d::Identity();
		Vector3d t0;
		std::normal_distribution<double> dist_normal(0.0, 1.0);
		t0 << dist_normal(rng), dist_normal(rng), dist_normal(rng);
		initialGuess = EManifold(R0, t0);
	}
	else if(initialGuessMethod == init_truth)
	{
		Matrix3d R_truth = RT_truth.block<3, 3>(0, 0);
		Vector3d t_truth = RT_truth.block<3, 1>(0, 3);
		initialGuess = EManifold(R_truth, t_truth);
	}
	
	// Run RANSAC or LMEDS
	EManifold bestModel;
	consensusAlg->run(pts1, pts2, initialGuess, bestModel);

	// Disambiguate rotation and translation
	time_cat(common::TimeCatNone);
	time_cat_verbose(common::TimeCatVerboseNone);
	Vector3d t = bestModel.t;
	Matrix3d R1 = bestModel.R;
	Matrix3d R2 = common::R1toR2(R1, t);
	if(poseDisambigMethod == disambig_trace)
	{
		if(R2.trace() > R1.trace())
			bestModel.setR(R2);
		Matrix3d R = bestModel.R;
		if(common::chierality(pts1, pts2, R, -t) > 
		   common::chierality(pts1, pts2, R, t))
		   bestModel.setT(-t);
	}
	else if(poseDisambigMethod == disambig_chierality)
	{
		int num_pos_depth[4];
		Matrix3d R12[4] = {R1, R1, R2, R2};
		Vector3d t12[4] = {t,  -t,  t, -t};
		for(int i = 0; i < 4; i++)
			num_pos_depth[i] = common::chierality(pts1, pts2, R12[i], t12[i]);
		int max_idx = max_element(num_pos_depth, num_pos_depth + 4) - num_pos_depth;
		bestModel.setR(R12[max_idx]);
		bestModel.setT(t12[max_idx]);
	}
	result.R = bestModel.R;
	result.t = bestModel.t;
	result.E = bestModel.E;
	result.has_RT = true;
	
	// Save best hypothesis for next time
	prevResult = bestModel;
}

double GNSAC_Solver::score_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const common::EHypothesis& hypothesis)
{
	return 0;
}

void run_tests()
{
	run_jacobian_tests();
	run_optimizer_tests();
}

class EManifoldTest : public EManifold
{
public:
	EManifoldTest() : EManifold() {}
	EManifoldTest(const Matrix3d& R, const Matrix3d& TR) : EManifold(R, TR) {}
	EManifoldTest(const Matrix3d& R, const Vector3d& t) : EManifold(R, t) {}
	SO3& getRot() {return rot;}
	SO2& getVec() {return vec;}
	Matrix3d& getE_() {return E_;}
};

void run_jacobian_tests()
{
	std::default_random_engine rng(0);
	std::normal_distribution<double> dist(0.0, 1.0);
	for(int ii = 0; ii < 100; ii++)
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

		// SO3
		// Boxplus negation
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

		// Test copy assignment operator (make sure it copies by value, not reference)
		SO3 rot_copy;
		rot_copy = rot;
		release_assert(&rot_copy != &rot);
		release_assert(&rot_copy.R != &rot.R);
		common::checkMatrices("rot_copy.R", "rot.R", rot_copy.R, rot.R);
		rot_copy.R = Matrix3d::Zero();
		release_assert(rot_copy.R.determinant() == 0);
		release_assert(fabs(rot.R.determinant() - 1) < 1e-8);

		// Unit vector
		Vector3d v;
		common::fill_rnd(v, dist, rng);
		v = unit(v);
		SO2 vec(v);
		common::checkMatrices("SO2(v).v", "v", vec.v, v);

		// Test copy assignment operator (make sure it copies by value, not reference)
		SO2 vec_copy;
		vec_copy = vec;
		release_assert(&vec_copy != &vec);
		release_assert(&vec_copy.R != &vec.R);
		release_assert(&vec_copy.v != &vec.v);
		common::checkMatrices("vec_copy.R", "vec.R", vec_copy.R, vec.R);
		common::checkMatrices("vec_copy.v", "vec.v", vec_copy.v, vec.v);
		vec_copy.R = Matrix3d::Zero();
		release_assert(vec_copy.R.determinant() == 0);
		release_assert(vec_copy.v.sum() == 0);
		release_assert(fabs(vec.R.determinant() - 1) < 1e-8);
		release_assert(vec.v.sum() != 0);

		// Ensure pointer is initialized correctly
		release_assert(vec.v.data() == vec.R.data() + 2);
		Vector3d z_dir;
		z_dir << 0, 0, 1;
		common::checkMatrices("vec.v", "vec.R'*[0; 0; 1]", vec.v, vec.R.transpose()*z_dir);

		// Ensure no rotation for unit vector in z direction
		rot = SO3(vec.R);
		w << 0, 0, 1;
		rot.boxplus(w, rot);
		SO2 vec2(rot.R);
		common::checkMatrices("v", "v.boxplus([0; 0; 1])", v, vec2.v);

		// Essential Matrix
		common::fill_rnd(w1, dist, rng);
		common::fill_rnd(w2, dist, rng);
		axisAngleGetR(w1, R1);
		axisAngleGetR(w2, R2);
		EManifoldTest E(R1, R2);

		// Check pointers
		common::checkMatrices("R", "E.R", R1, E.R);
		common::checkMatrices("TR", "E.TR", R2, E.TR);
		common::checkMatrices("E.R", "E.rot.R", E.R, E.getRot().R);
		common::checkMatrices("E.TR", "E.vec.R", E.TR, E.getVec().R);
		common::checkMatrices("E.t", "E.vec.v", E.t, E.getVec().v);
		common::checkMatrices("E.E", "E.E_", E.E, E.getE_());
		release_assert(&E.R == &E.getRot().R);
		release_assert(&E.TR == &E.getVec().R);
		release_assert(&E.t == &E.getVec().v);
		release_assert(&E.E == &E.getE_());

		// Test copy assignment operator (make sure it copies by value, not reference)
		EManifoldTest E_copy;
		E_copy = E;
		release_assert(&E_copy != &E);
		release_assert(&E_copy.E != &E.E);
		release_assert(&E_copy.R != &E.R);
		release_assert(&E_copy.t != &E.t);
		release_assert(&E_copy.TR != &E.TR);
		common::checkMatrices("E_copy.E", "E.E", E_copy.E, E.E);
		common::checkMatrices("E_copy.R", "E.R", E_copy.R, E.R);
		common::checkMatrices("E_copy.t", "E.t", E_copy.t, E.t);
		common::checkMatrices("E_copy.TR", "E.TR", E_copy.TR, E.TR);

		////////// Jacobian tests for Manifold Elements ////////////
		// SO3
		common::fill_rnd(w, dist, rng);
		axisAngleGetR(w, R);
		rot = SO3(R);
		for(int i = 0; i < 3; i++)
		{
			Matrix3d deriv, deriv_num;

			// Analytical derivative
			rot.derivative(i, deriv);

			// Numerical derivative
			numericalDerivative_i<3>([](const SO3& rot_so3) -> Matrix3d {
					return rot_so3.R;
				}, rot,	deriv_num, i);

			// Check result
			common::checkMatrices("SO3 deriv", "SO3 deriv_num", deriv, deriv_num, i);
		}

		// SO2
		common::fill_rnd(w, dist, rng);
		axisAngleGetR(w, R);
		vec = SO2(R);
		for(int i = 0; i < 2; i++)
		{
			Vector3d deriv, deriv_num;

			// Analytical derivative
			vec.derivative(i, deriv);

			// Numerical derivative
			numericalDerivative_i<2>([](const SO2& vec_so2) -> Vector3d {
					return vec_so2.v;
				}, vec,	deriv_num, i);

			// Check result
			common::checkMatrices("SO2 deriv", "SO2 deriv_num", deriv, deriv_num, i);
		}

		// EManifold
		common::fill_rnd(w1, dist, rng);
		common::fill_rnd(w2, dist, rng);
		axisAngleGetR(w1, R1);
		axisAngleGetR(w2, R2);
		EManifold eManifold(R1, R2);
		for(int i = 0; i < 5; i++)
		{
			Matrix3d deriv, deriv_num;

			// Analytical derivative
			eManifold.derivative(i, deriv);

			// Numerical derivative
			numericalDerivative_i<5>([](const EManifold& e_manifold) -> Matrix3d {
					return e_manifold.E;
				}, eManifold, deriv_num, i);

			// Check result
			common::checkMatrices("SO2 deriv", "SO2 deriv_num", deriv, deriv_num, i);
		}

		////////// Jacobian tests for Cost Functions ////////////
		common::scan_t pts1;
		common::scan_t pts2;
		const int N_PTS = 1;
		for(int i = 0; i < N_PTS; i++)
		{
			Vector2d pt1, pt2;
			common::fill_rnd(pt1, dist, rng);
			common::fill_rnd(pt2, dist, rng);
			pts1.push_back(unit(pt1));
			pts2.push_back(unit(pt2));
		}
		vector<cost_function_t> cost_functions = {cost_algebraic, cost_single, cost_sampson};
		vector<string> cost_function_names = {"Algebraic", "Single", "Sampson"};
		for(int i = 0; i < cost_functions.size(); i++)
		{
			shared_ptr<DifferentiableResidual> cost_function = DifferentiableResidual::from_enum(cost_functions[i]);
			string name = cost_function_names[i];
			Matrix<double, N_PTS, 5> J, J_num;
			Matrix<double, N_PTS, 1> err;
			Matrix<double, N_PTS, 1> err2;
			Matrix<double, N_PTS, 1> err_sqr;
			Map<VectorXd> err_map = Map<VectorXd>(err.data(), N_PTS, 1);
			Map<VectorXd> err2_map = Map<VectorXd>(err2.data(), N_PTS, 1);
			Map<VectorXd> err_sqr_map = Map<VectorXd>(err_sqr.data(), N_PTS, 1);
			Map<MatrixXd> J_map = Map<MatrixXd>(J.data(), N_PTS, 5);

			// Analytical derivative
			cost_function->residual(pts1, pts2, eManifold, err_map);
			cost_function->residual_sqr(pts1, pts2, eManifold, err_sqr_map);
			cost_function->residual_diff(pts1, pts2, eManifold, err2_map, J_map);

			// Numerical derivative
			numericalDerivative([&](const EManifold& e_manifold) -> Matrix<double, N_PTS, 1> {
					Matrix<double, N_PTS, 1> err;
					Map<VectorXd> err_map = Map<VectorXd>(err.data(), N_PTS);
					cost_function->residual(pts1, pts2, e_manifold, err_map);
					return err;
				}, eManifold, J_num);

			// Check result
			common::checkMatrices(name + "_err", name + "_err2", err, err2);
			common::checkMatrices(name + "_err^2", name + "err_sqr", err.array().square().matrix(), err_sqr);
			common::checkMatrices(name + "_J", name + "_J_Num", J, J_num, -1, 1e-5);
		}
	}
}

void run_optimizer_tests()
{
	common::init_logs("../params/solvers/gn_eigen.yaml", "../logs/test");	
	std::default_random_engine rng(0);
	std::normal_distribution<double> dist(0.0, 1.0);

	////////// Test each optimizer with each cost function ////////////
	// Randomly generate some three-dimensional points on the surface of a unit sphere
	const int n_pts = 50;
	vector<Vector3d> Pts;
	std::uniform_real_distribution<double> dist_uniform(0, 1);
	for(int i = 0; i < n_pts; i++)
	{
		Vector3d pt;
		common::fill_rnd(pt, dist, rng);
		Pts.push_back(unit(pt));
	}
	common::write_log(common::log_unit_test_pts_world, (char*)&Pts[0], sizeof(double) * 3 * n_pts);

	// Place the camera 2 units away from the center of the sphere, so that the maximum
	// FOV angle is approx 45 degrees.
	// We'll use a little trick to do this. We'll do this using the SO2 unit vector object
	// to get the position (just multiply by 2), and the SO2 derivatives to get the unit vectors
	// in the other directions.
	Vector3d w;
	Matrix3d R;
	common::fill_rnd(w, dist, rng);
	axisAngleGetR(w, R);
	SO2 vec(R);
	Vector3d pos_0_1 = -vec.v * 2;

	// These vectors are the basis vectors of the camera in frame 0 (global frame)
	// They should really be written as x_sup0_sub1 or sup0_x_sub1.
	Vector3d d1, d2;
	vec.derivative(0, d1);
	vec.derivative(1, d2);
	Vector3d x_0_1 = -unit(d2);
	Vector3d y_0_1 = unit(d1);
	Vector3d z_0_1 = unit(vec.v);

	// This is the rotation from 1 to 0, which should be written as R_sup0_sub1 or sup0_R_sub1.
	// We'll have to invert it later to get the actual rotation we want.
	Matrix3d R_0_1;
	R_0_1 << x_0_1, y_0_1, z_0_1;

	// Make sure that the basis vectors are orthogonal and that it is a right-hand coordinate frame.
	release_assert(fabs(R_0_1.determinant() - 1) < 1e-8);
	release_assert((R_0_1*R_0_1.inverse() - Matrix3d::Identity()).norm() < 1e-8);

	// Let's also create the position and rotation of the second camera, 
	// by moving the position and rotation.
	double move_rot_deg = 2;
	double move_pos = 0.1;
	common::fill_rnd(w, dist, rng);
	w = unit(w) * move_rot_deg * M_PI/180;
	Matrix3d dR;
	axisAngleGetR(w, dR);
	Vector3d d_pos;
	common::fill_rnd(d_pos, dist, rng);		
	d_pos = unit(d_pos) * move_pos;
	Matrix3d R_0_2 = R_0_1*dR;
	Vector3d pos_0_2 = pos_0_1 + d_pos;

	// Now construct the complete RT matrices and invert.
	Matrix4d RT_0_1 = common::RT_combine(R_0_1, pos_0_1);
	Matrix4d RT_0_2 = common::RT_combine(R_0_2, pos_0_2);
	Matrix4d RT_1_0 = RT_0_1.inverse();
	Matrix4d RT_2_0 = RT_0_2.inverse();
	Matrix4d RT_2_1 = RT_2_0 * RT_0_1;
	Matrix4d RT_true = RT_2_1;
	Matrix3d R_true;
	Vector3d t_true;
	common::RT_split(RT_true, R_true, t_true);
	t_true = unit(t_true);
	EManifold eManifoldTruth(R_true, t_true);

	// Now perterb the position and rotation by a small ratio of the original
	// movement. This will be used as an initial guess for the optimizer.
	double perterb_rot_deg = move_rot_deg*0.1;
	double perterb_pos = move_pos*0.01;
	Vector3d T_initialGuess;
	common::fill_rnd(w, dist, rng);
	w = unit(w) * perterb_rot_deg * M_PI/180;
	axisAngleGetR(w, dR);
	common::fill_rnd(d_pos, dist, rng);
	d_pos = unit(d_pos) * perterb_pos;
	Matrix3d R_initialGuess = R_true*dR;
	Vector3d t_initialGuess = unit(t_true + d_pos);
	EManifold initialGuess(R_initialGuess, t_initialGuess);

	// Project world points into the camera frame (normalized image plane)
	vector<float> dist1, dist2;
	common::scan_t pts1, pts2;
	Matrix3d cameraMatrixNone = Matrix3d::Identity();
	common::project_points(Pts, pts1, dist1, RT_1_0, cameraMatrixNone);
	common::project_points(Pts, pts2, dist2, RT_2_0, cameraMatrixNone);
	for(int i = 0; i < n_pts; i++)
	{
		release_assert(dist1[i] > 0);
		release_assert(dist2[i] > 0);
		release_assert(fabs(pts1[i](0)) < 1);
		release_assert(fabs(pts1[i](1)) < 1);
		release_assert(fabs(pts2[i](0)) < 1);
		release_assert(fabs(pts2[i](1)) < 1);
	}
	common::write_log(common::log_unit_test_pts_camera, (char*)&pts1[0], sizeof(double) * 2 * n_pts);
	common::write_log(common::log_unit_test_pts_camera, (char*)&pts2[0], sizeof(double) * 2 * n_pts);

	// Run optimizers!
	double maxIterations = 20;
	double exitTolerance = 1e-10;			
	double lambda0 = 1e-4;
	vector<cost_function_t> cost_functions = {cost_algebraic, cost_single, cost_sampson};
	vector<string> cost_function_names = {"Algebraic", "Single", "Sampson"};
	vector<optimizer_t> optimizers = {optimizer_GN, optimizer_LM};
	vector<string> optimizer_names = {"GaussNewton", "LevenbergMarquardt"};
	for(int i = 0; i < optimizers.size(); i++)
	{
		for(int j = 0; j < cost_functions.size(); j++)
		{
			// Create optimizer
			cout << "Optimizer: " << GREEN_TEXT << optimizer_names[i] << BLACK_TEXT << endl;
			cout << "Cost Function: " << GREEN_TEXT << cost_function_names[j] << BLACK_TEXT << endl;
			optimizer_t optim = optimizers[i];
			shared_ptr<DifferentiableResidual> cost_function = DifferentiableResidual::from_enum(cost_functions[j]);
			shared_ptr<Optimizer> optimizer;
			if(optim == optimizer_GN)
				optimizer = make_shared<GaussNewton>(cost_function, maxIterations, exitTolerance);
			else if(optim == optimizer_LM)
				optimizer = make_shared<LevenbergMarquardt>(cost_function, maxIterations, exitTolerance, lambda0);

			// Setup
			EManifold eManifold;
			eManifold = initialGuess;
			double cost;
			Vector4d err_truth;
			VectorXd err_vec = VectorXd(n_pts);
			Map<VectorXd> err_map = Map<VectorXd>(err_vec.data(), n_pts);

			// Err before
			cost_function->residual(pts1, pts2, eManifold, err_map);
			cost = err_vec.norm();
			err_truth = common::err_truth(eManifold.R, eManifold.t, RT_true);
			printf("             %-8s %-8s %-8s %-8s %-8s\n", "Cost", "R", "t", "ch_R", "ch_t");
			printf("Initial err: %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n", cost, err_truth(0), err_truth(1), err_truth(2), err_truth(3));

			// Optimize 
			optimizer->optimize(pts1, pts2, eManifold, eManifold);

			// Err after
			cost_function->residual(pts1, pts2, eManifold, err_map);
			cost = err_vec.norm();
			err_truth = common::err_truth(eManifold.R, eManifold.t, RT_true);
			printf("Final err:   %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n", cost, err_truth(0), err_truth(1), err_truth(2), err_truth(3));

			// Err truth (should be zero)
			cost_function->residual(pts1, pts2, eManifoldTruth, err_map);
			cost = err_vec.norm();
			err_truth = common::err_truth(eManifoldTruth.R, eManifoldTruth.t, RT_true);
			printf("Truth err:   %-8.3f %-8.3f %-8.3f %-8.3f %-8.3f\n", cost, err_truth(0), err_truth(1), err_truth(2), err_truth(3));
			printf("\n");				
		}
	}
	common::close_logs();
}

}
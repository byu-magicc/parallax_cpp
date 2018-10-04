#ifndef GNSAC_PTR_EIG_H
#define GNSAC_PTR_EIG_H

#include "common.h"
#include <eigen3/Eigen/Dense>

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

void GN_step(const common::scan_t pts1, const common::scan_t pts2, const double* R, const double* TR,
		  double* E2, double* R2, double* TR2, double* t2, int method, bool withNormalization = true, bool useLM = false);

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

Eigen::Matrix3d findEssentialMatGN(common::scan_t pts1, common::scan_t pts2,
		Eigen::Matrix3d& R0, Eigen::Vector3d& t0, Eigen::Matrix3d& R2, Eigen::Vector3d& t2,
		int n_hypotheses, int n_GNiters,
		bool withNormalization = true, bool optimizedCost = false);
		
}

#endif //GNSAC_PTR_EIG_H
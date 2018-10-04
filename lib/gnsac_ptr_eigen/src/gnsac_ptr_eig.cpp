#include <eigen3/Eigen/Dense>
#include <vector>
#include "gnsac_ptr_eig.h"
#include <random>
#include <chrono>

using namespace std;
using namespace Eigen;

namespace gnsac_ptr_eigen
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

void skew(const double* w, double* T)
{
	T[0] = 0;     T[1] = -w[2]; T[2] = w[1];
	T[3] = w[2];  T[4] = 0;     T[5] = -w[0];
	T[6] = -w[1]; T[7] = w[0];  T[8] = 0;
}

void mult3pt(const double* A, const Vector2d pt, double* y)
{
	y[0] = A[0]*pt(0) + A[1]*pt(1) + A[2];
	y[1] = A[3]*pt(0) + A[4]*pt(1) + A[5];
	y[2] = A[6]*pt(0) + A[7]*pt(1) + A[8];
}

void mult31(const double* A, const double* x, double* y)
{
	y[0] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2];
	y[1] = A[3]*x[0] + A[4]*x[1] + A[5]*x[2];
	y[2] = A[6]*x[0] + A[7]*x[1] + A[8]*x[2];
}

double mult3vtv(const double* x, const double* y)
{
	return x[0]*y[0] + x[1]*y[1] + x[2]*y[2];
}

double multptv(const Vector2d pt, const double* v)
{
	return pt(0)*v[0] + pt(1)*v[1] + v[2];
}

void mult33(const double* A, const double* B, double* C)
{
	C[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
	C[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
	C[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8];
	C[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
	C[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
	C[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8];
	C[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6];
	C[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7];
	C[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
}

double normsqr31(const double* v)
{
	return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

double norm31(const double* v)
{
	return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

void unit31(const double* v, double* v2)
{
	double norm = norm31(v);
	v2[0] = v[0] / norm;
	v2[1] = v[1] / norm;
	v2[2] = v[2] / norm;
}

void tran33(const double* A, double* B)
{
	B[0] = A[0]; B[1] = A[3]; B[2] = A[6];
	B[3] = A[1]; B[4] = A[4]; B[5] = A[7];
	B[6] = A[2]; B[7] = A[5]; B[8] = A[8];
}

void tran_nxm(const double* A, int n, int m, double* B)
{
	for(int i = 0; i < n; i++)
		for(int j = 0; j < m; j++)
			B[j*n + i] = A[i*m + j];
}

void unitvector_getV(const double* R, double* v)
{
	//v = R'*[0; 0; 1];
	v[0] = R[6];
	v[1] = R[7];
	v[2] = R[8];
}

void t2TR(const double* t, double* TR)
{
	// Note that Matlab and C++ give slightly different values for acos (different by 1e-15)
	// This small difference can cause the GN algorithm to diverge and find a valid but different solution.
	double t_unit[3];
	unit31(t, t_unit);
	double theta = acos(t_unit[2]);
	double snc = sinc(theta);
	double w[3];
	w[0] = t_unit[1]/snc;
	w[1] = -t_unit[0]/snc;
	w[2] = 0;
	axisAngleGetR(w, TR);
}

void elementaryVector3(int idx, double* v)
{
	v[0] = 0;
	v[1] = 0;
	v[2] = 0;
	v[idx] = 1;
}

//void printArray(const char* s, const double* p, int n, int m, int sfAfter = 4, int sfBefore = 5);

void printArray(const char* s, const double* p, int n, int m, int sfAfter, int sfBefore)
{
	printf("%s: (%dx%d)\n", s, n, m);
	char format[40];
	sprintf(format, "%%%d.%df", sfBefore + sfAfter, sfAfter);
	for (int r = 0; r < n; r++)
	{
		if (r == 0)
			printf("[");
		else
			printf("\n ");
		for (int c = 0; c < m; c++)
		{
			if (c > 0)
				printf(" ");
			printf(format, p[r*m + c]);
		}
	}
	printf("]\n");
}

//#define RED_TEXT "\033[31m[49m"
#define RED_TEXT "\033[101m"
#define GREEN_TEXT "\033[32m"
#define BLACK_TEXT "\033[0m\033[49m"

void printArrayComp(const char* s, const double* p, const double* p2, int n, int m, float eps = 1e-6, int sfAfter = 4, int sfBefore = 5)
{
	printf("%s: (%dx%d)\n", s, n, m);
	char format[40];
	sprintf(format, "%%%d.%df", sfBefore + sfAfter, sfAfter);
	for (int r = 0; r < n; r++)
	{
		if (r == 0)
			printf("[");
		else
			printf("\n ");
		for (int c = 0; c < m; c++)
		{
			if (c > 0)
				printf(" ");
			bool equal = fabs(p[r*m + c] - p2[r*m + c]) <= eps;
			if(!equal)
				printf(RED_TEXT);
			printf(format, p[r*m + c]);
			if(!equal)
				printf(BLACK_TEXT);
		}
	}
	printf("]\n");
}

void checkArrays(const char* s1, const char* s2, const double* p1, const double* p2, int n, int m, float eps = 1e-6, int sfAfter = 4, int sfBefore = 5)
{
	// First, find out if the arrays are equal
	bool equal = true;
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c < m; c++)
		{
			if (fabs(p1[r*m + c] - p2[r*m + c]) > eps)
			{
				equal = false;
				break;
			}
		}
		if (!equal)
			break;
	}
	if(equal)
		return;

	// If the arrays are not equal, print them both out
	printArrayComp(s1, p1, p2, n, m, eps, sfAfter, sfBefore);
	printArrayComp(s2, p2, p1, n, m, eps, sfAfter, sfBefore);
	printf("\n");
}


void printPoints(const char* s, common::scan_t pts)
{
	printf("%s: (2x%d)\n", s, (int)pts.size());
	printf("[");
	for(int i = 0; i < pts.size(); i++)
		printf("%10.4f ", pts[i](0));
	printf("\n ");
	for(int i = 0; i < pts.size(); i++)
		printf("%10.4f ", pts[i](1));
	printf("]\n");
}

void zero5(double* r)
{
	r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0; r[4] = 0;
}

void eye3(double* I)
{
	I[0] = 1; I[1] = 0; I[2] = 0;
	I[3] = 0; I[4] = 1; I[5] = 0;
	I[6] = 0; I[7] = 0; I[8] = 1;
}

void mult3s(const double* A, double s, double* B)
{
	B[0] = s*A[0]; B[1] = s*A[1]; B[2] = s*A[2];
	B[3] = s*A[3]; B[4] = s*A[4]; B[5] = s*A[5];
	B[6] = s*A[6]; B[7] = s*A[7]; B[8] = s*A[8]; 
}

void mult31s(const double* A, double s, double* B)
{
	B[0] = s*A[0];
	B[1] = s*A[1];
	B[2] = s*A[2];
}

void add33(const double* A, const double* B, double* C)
{
	C[0] = A[0] + B[0]; C[1] = A[1] + B[1]; C[2] = A[2] + B[2];
	C[3] = A[3] + B[3]; C[4] = A[4] + B[4]; C[5] = A[5] + B[5];
	C[6] = A[6] + B[6]; C[7] = A[7] + B[7]; C[8] = A[8] + B[8];
}

void copy33(const double* A, double* B)
{
	B[0] = A[0]; B[1] = A[1]; B[2] = A[2];
	B[3] = A[3]; B[4] = A[4]; B[5] = A[5];
	B[6] = A[6]; B[7] = A[7]; B[8] = A[8];
}

void copy31(const double* A, double* B)
{
	B[0] = A[0];
	B[1] = A[1];
	B[2] = A[2];
}

// Note: This is the Rodriguez equation, but to avoid numerical underflow, the 
// "1 - cos(theta)" has been replaced with "2*sin(theta/2)^2", (a trig identity).
// This avoids underflow when theta is small.
void axisAngleGetR(const double* w, double* R)
{
	// theta = norm(w);
	double theta = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
	
	// wx = skew(w);
	double wx[9];
	skew(w, wx);
	
	// R = eye(3) + sinc(theta)*wx + 0.5*sinc(theta/2)^2*wx^2;	
	double I[9];
	eye3(I);
	double tmp1[9];
	mult3s(wx, sinc(theta), tmp1);
	double wx2[9];
	mult33(wx, wx, wx2);
	double st2 = sinc(theta/2);
	double tmp2[9];
	mult3s(wx2, st2*st2/2, tmp2);
	double tmp3[9];
	add33(I, tmp1, tmp3);
	add33(tmp3, tmp2, R);
}

// can operate in-place
void R_boxplus(const double* R, const double* w, double* R2)
{
	double dR[9];
	axisAngleGetR(w, dR);
	if(R != R2)
		mult33(dR, R, R2);
	else
	{
		double R3[9];
		mult33(dR, R, R3);
		copy33(R3, R2);
	}
}

void E_boxplus(const double* R, const double* TR, const double* dx, double* R2, double* TR2)
{
	const double* w1 = dx;
	double w2[3];
	w2[0] = dx[3];
	w2[1] = dx[4];
	w2[2] = 0;
	R_boxplus(R, w1, R2);
	R_boxplus(TR, w2, TR2);
}

void RT_getE(const double* R, const double* TR, double* t, double* E)
{
	//Tx = skew(unitvector_getV(TR));
	unitvector_getV(TR, t);
	double Tx[9];
	skew(t, Tx);
	
	//E = Tx*R;
	mult33(Tx, R, E);
}

// calculate residual and jacobian of residual for a single measurement
void getE_diff(const double* R, const double* TR, double* E, double* deltaE)
{
	//Tx = skew(unitvector_getV(TR));
	double t[3];
	unitvector_getV(TR, t);
	double Tx[9];
	skew(t, Tx);
	
	//E = Tx*R;
	mult33(Tx, R, E);
	for(int i = 0; i < 5; i++)
	{
		// find deltaE using generators of Lie Algebra
		double* deltaE_i = &deltaE[9*i];
		if(i < 3)
		{
			// genR = skew(elementaryVector(3, i));
			double ev[3];
			elementaryVector3(i, ev);
			double genR[9];
			skew(ev, genR);

			// deltaE = Tx * genR * R;
			double tmp[9];
			mult33(genR, R, tmp);
			mult33(Tx, tmp, deltaE_i);
		}
		else
		{
			// genTR = skew([elementaryVector(2, i - 3); 0]);
			double ev[3];
			elementaryVector3(i - 3, ev);
			double genTR[9];
			skew(ev, genTR);
			
			// deltaE = skew((genTR * TR).' * [0; 0; 1])*R;
			double tmp[9];
			mult33(genTR, TR, tmp);
			double t_[3];
			unitvector_getV(tmp, t_);
			double Tx_[9];
			skew(t_, Tx_);
			mult33(Tx_, R, deltaE_i);
		}
	}
}

void normalizeLine_diff(const double* L, const double* dL, double* LH, double* dLH)
{
	double a = L[0];
	double b = L[1];
	double c = L[2];
	double s = sqrt(a*a + b*b);
	double sinv = 1./s;
	double s3inv = 1./(s*s*s);
	LH[0] = sinv*L[0];
	LH[1] = sinv*L[1];
	LH[2] = sinv*L[2];
	dLH[0] = (sinv - a*a*s3inv)*dL[0] -         a*b*s3inv *dL[1];
	dLH[1] =        -a*b*s3inv *dL[0] + (sinv - b*b*s3inv)*dL[1];
	dLH[2] =        -a*c*s3inv *dL[0] -         b*c*s3inv *dL[1] + sinv*dL[2];
}

void normalizeLine_diffOnly(const double* L, const double* dL, double* dLH)
{
	double a = L[0];
	double b = L[1];
	double c = L[2];
	double s = sqrt(a*a + b*b);
	double sinv = 1./s;
	double s3inv = 1./(s*s*s);
	dLH[0] = (sinv - a*a*s3inv)*dL[0] -         a*b*s3inv *dL[1];
	dLH[1] =        -a*b*s3inv *dL[0] + (sinv - b*b*s3inv)*dL[1];
	dLH[2] =        -a*c*s3inv *dL[0] -         b*c*s3inv *dL[1] + sinv*dL[2];
}

void normalizeLine(const double* L, double* LH)
{
	double a = L[0];
	double b = L[1];
	double c = L[2];
	double s = sqrt(a*a + b*b);
	double sinv = 1./s;
	LH[0] = sinv*L[0];
	LH[1] = sinv*L[1];
	LH[2] = sinv*L[2];
}

// calculate residual and jacobian of residual for a single measurement
double residual_diff(const double* R, const double* TR, const double* E, const double* deltaE, 
		  const Vector2d p1, const Vector2d p2, double* deltaR)
{
	// r = p2'*normalizeLine(E*p1);
	double L[3];
	mult3pt(E, p1, L);
	double r; //see below
	
	// deltaR = zeros(1, 5);
	zero5(deltaR);
	
	for(int i = 0; i < 5; i++)
	{
		const double* deltaE_i = &deltaE[9*i];
		
		// calculate derivative of the residual
		// line = E*p1; (already calculated above)
		// deltaLine = deltaE*p1;
		double dL[3];
		mult3pt(deltaE_i, p1, dL);
		
		// [~, deltaLHat] = normalizeLine_diff(line, deltaLine);
		double dLH[3];
		if(i == 0)
		{
			double LH[3];
			normalizeLine_diff(L, dL, LH, dLH);
			
			// r = p2'*normalizeLine(E*p1);
			r = multptv(p2, LH);
		}
		else
			normalizeLine_diffOnly(L, dL, dLH);
		
		// deltaR(:, i) = p2'*deltaLHat;
		deltaR[i] = multptv(p2, dLH);
	}
	return r;
}

// calculate residual for a single measurement
double residual(const double* E, const Vector2d p1, const Vector2d p2)
{
	// r = p2'*normalizeLine(E*p1);
	double L[3];
	mult3pt(E, p1, L);
	double LH[3];
	normalizeLine(L, LH);
	double r = multptv(p2, LH);	
	return r;
}

// calculate residual and jacobian of residual for a single measurement
double residual_diff_without_normalization(
		const double* R, const double* TR, const double* E, const double* deltaE, 
		const Vector2d p1, const Vector2d p2, double* deltaR)
{
	// r = p2'*normalizeLine(E*p1);
	double L[3];
	mult3pt(E, p1, L);
	double r; //see below
	
	// deltaR = zeros(1, 5);
	zero5(deltaR);
	
	for(int i = 0; i < 5; i++)
	{
		const double* deltaE_i = &deltaE[9*i];
		
		// calculate derivative of the residual
		// line = E*p1; (already calculated above)
		// deltaLine = deltaE*p1;
		double dL[3];
		mult3pt(deltaE_i, p1, dL);
		
		if(i == 0)
			r = multptv(p2, L);
		
		// deltaR(:, i) = p2'*deltaL;
		deltaR[i] = multptv(p2, dL);
	}
	return r;
}

// calculate residual for a single measurement
double residual_without_normalization(const double* E, const Vector2d p1, const Vector2d p2)
{
	// r = p2'*E*p1;
	double L[3];
	mult3pt(E, p1, L);
	double r = multptv(p2, L);	
	return r;
}

#define MAX_PTS 2000

//function [E, R, TR] = GN_step(pts1, pts2, R, TR)
void GN_step(const common::scan_t pts1, const common::scan_t pts2, const double* R, const double* TR,
		  double* E2, double* R2, double* TR2, double* t2, int method, bool withNormalization, bool useLM)
{
	// Gauss-Newton
	// N = size(pts1, 2);
	// r = zeros(N, 1);
	// J = zeros(N, 5);
	double lambda = 1e-4;
	int N = (int)pts1.size();
	assert(N < MAX_PTS);
	static double r[MAX_PTS*1];
	static double r2[MAX_PTS*1];
	static double J[MAX_PTS*5];
	static double dx[5*1];
	Map<Matrix<double, -1, -1, RowMajor>> r_map = Map<Matrix<double, -1, -1, RowMajor>>(r, N, 1);
	Map<Matrix<double, -1, -1, RowMajor>> r2_map = Map<Matrix<double, -1, -1, RowMajor>>(r2, N, 1);
	Map<Matrix<double, -1, -1, RowMajor>> J_map = Map<Matrix<double, -1, -1, RowMajor>>(J, N, 5);
	Map<Matrix<double, -1, -1, RowMajor>> dx_map = Map<Matrix<double, -1, -1, RowMajor>>(dx, 5, 1);

	double E[3*3];
	double deltaE[15*3];
	getE_diff(R, TR, E, deltaE);
	for(int i = 0; i < N; i++)
	{
		// [r(i), J(i, :)] = residual_diff(R, TR, pts1(:, i), pts2(:, i));
		if(withNormalization)
			r[i] = residual_diff(R, TR, E, deltaE, pts1[i], pts2[i], &J[i*5]);
		else
			r[i] = residual_diff_without_normalization(R, TR, E, deltaE, pts1[i], pts2[i], &J[i*5]);
	}
	
	if(useLM)
	{
		// JtJ = J'*J;
		// Jtr = J'*r;
		Matrix<double, 5, 5> JtJ = J_map.transpose()*J_map;
		Matrix<double, 5, 1> Jtr = J_map.transpose()*r_map;
		while(true)
		{
			// dx = -(JtJ + lambda*diag(diag(JtJ)))\J'*r;
			Matrix<double, 5, 5> JtJ_diag_only = JtJ.diagonal().asDiagonal();
			dx_map = -(JtJ + lambda*JtJ_diag_only).fullPivLu().solve(Jtr);
	
			// [R2, TR2] = E_boxplus(R, TR, dx);
			E_boxplus(R, TR, dx, R2, TR2);
			
			// r2 = residual_batch(R2, TR2, pts1, pts2);
			for(int i = 0; i < N; i++)
			{
				if(withNormalization)
					r2[i] = residual(E, pts1[i], pts2[i]);
				else
					r2[i] = residual_without_normalization(E, pts1[i], pts2[i]);
			}
			
			if(r2_map.norm() <= r_map.norm())
			{
				// If the new error is lower, keep new values and move onto the next iteration.
				// Decrease lambda, which makes the algorithm behave more like Gauss-Newton.
				// Gauss-Newton gives very fast convergence when the function is well-behaved.
				R = R2;
				TR = TR2;
				lambda = lambda / 2;
				break;
			}
			else
				// If the new error is higher, discard new values and try again. 
				// This time increase lambda, which makes the algorithm behave more
				// like gradient descent and decreases the step size.
				// Keep trying until we succeed at decreasing the error.
				lambda = lambda * 2;
		}
	}
	else
	{
		// dx = -J\r;
		dx_map = -J_map.fullPivLu().solve(r_map);
		
		// [R, TR] = E_boxplus(R, TR, dx);
		E_boxplus(R, TR, dx, R2, TR2);
	}
	
	// t = unitvector_getV(TR);
	unitvector_getV(TR2, t2);
	
	// E = skew(t)*R;
	double tx[9];
	skew(t2, tx);
	mult33(tx, R2, E2);
}

GNHypothesis::GNHypothesis() : R_map(R), TR_map(TR), E_map(E), t_map(t), cost(0)
{
	R_map = Matrix<double, 3, 3, RowMajor>::Identity();
	TR_map = Matrix<double, 3, 3, RowMajor>::Identity();
	RT_getE(R, TR, t, E);
}
	
GNHypothesis::GNHypothesis(Matrix3d R0, Vector3d t0) : R_map(R), TR_map(TR), E_map(E), t_map(t), cost(0)
{
	R_map = R0;
	t_map = t0;
	t2TR(t, TR);
	RT_getE(R, TR, t, E);
}

void getSubset(common::scan_t& pts1, common::scan_t& pts2, common::scan_t& subset1, common::scan_t& subset2, int modelPoints, 
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

double score_LMEDS(common::scan_t& pts1, common::scan_t& pts2, double* E, double maxMedian)
{
	int costsAboveMedian = 0;
	int n_pts = (int)pts1.size();
	int medianIdx = (int)n_pts/2;
	vector<double> err = vector<double>(n_pts, 0);
	Map<Matrix<double, 3, 3, RowMajor>> E_map = Map<Matrix<double, 3, 3, RowMajor>>(E);
	for (int i = 0; i < n_pts; i++)
	{
		Vector3d x1;
		x1 << pts1[i](0), pts1[i](1), 1.;
		Vector3d x2;
		x2 << pts2[i](0), pts2[i](1), 1.;
		Vector3d Ex1 = E_map * x1;
		Vector3d Etx2 = E_map.transpose() * x2;
		double x2tEx1 = x2.dot(Ex1);
		
		double a = Ex1[0] * Ex1[0];
		double b = Ex1[1] * Ex1[1];
		double c = Etx2[0] * Etx2[0];
		double d = Etx2[1] * Etx2[1];

		double err_i = x2tEx1 * x2tEx1 / (a + b + c + d);

// 		double a, b, c, dotprod, distsqr;
// 		
// 		// Find the equation of the line 
// 		// l = [a b c]' E*x2
// 		// ax + by + c = 0
// 		a = E[0]*pts1[i](0) + E[1]*pts1[i](1) + E[2];
// 		b = E[3]*pts1[i](0) + E[4]*pts1[i](1) + E[5];
// 		c = E[6]*pts1[i](0) + E[7]*pts1[i](1) + E[8];
// 		
// 		// distance to the line can be found be taking the dot product
// 		// of the normalized line and the homogeneous coordinate of the point.
// 		// To normalize the line, divide a, b, and c by sqrt(a^2 + b^2), 
// 		// so that vector [a_new b_new] is a unit vector.
// 		// We normalize when the square of the distance is computed, thus avoiding a sqrt.
// 		dotprod = (pts2[i](0)*a + pts2[i](1)*b + c);
// 		distsqr = dotprod*dotprod / (a*a + b*b);
//			err_i = distsqr;
		
		err[i] = err_i;
		
		// Preempt scoring if median is guaranteed to be above a certain amount
		if(err_i > maxMedian)
		{
			costsAboveMedian++;
			if(costsAboveMedian > medianIdx)
				return maxMedian;
		}
	}
	std::nth_element(err.begin(), err.begin() + medianIdx, err.end());
	return err[medianIdx];
}

double score_LMEDS2(common::scan_t& pts1, common::scan_t& pts2, double* E, double maxMedian)
{
	int costsAboveMedian = 0;
	int n_pts = (int)pts1.size();
	int medianIdx = (int)n_pts/2;
	vector<double> err = vector<double>(n_pts, 0);
	for (int i = 0; i < n_pts; i++)
	{
		double a1, b1, c1, a2, b2, c2, dotprod, distsqr, err_i;
		
		// Find the equation of the line 
		// l = [a b c]' = E*x1
		// ax + by + c = 0
		a2 = E[0]*pts1[i](0) + E[1]*pts1[i](1) + E[2];
		b2 = E[3]*pts1[i](0) + E[4]*pts1[i](1) + E[5];
		c2 = E[6]*pts1[i](0) + E[7]*pts1[i](1) + E[8];
		
		a1 = E[0]*pts2[i](0) + E[3]*pts2[i](1) + E[6];
		b1 = E[1]*pts2[i](0) + E[4]*pts2[i](1) + E[7];
		c1 = E[2]*pts2[i](0) + E[5]*pts2[i](1) + E[8];
		
		// Distance to the line can be found be taking the dot product
		// of the normalized line and the homogeneous coordinate of the point.
		// To normalize the line, divide a, b, and c by sqrt(a^2 + b^2), 
		// so that vector [a_new b_new] is a unit vector.
		// We normalize when the square of the distance is computed, thus avoiding a sqrt.
		dotprod = (pts2[i](0)*a2 + pts2[i](1)*b2 + c2);
		//distsqr = dotprod*dotprod / (a2*a2 + b2*b2);
		distsqr = dotprod*dotprod / (a2*a2 + b2*b2 + a1*a1 + b1*b1);
		err_i = distsqr;
		err[i] = err_i;
		
		// Preempt scoring if median is guaranteed to be above a certain amount
		if(err_i > maxMedian)
		{
			costsAboveMedian++;
			if(costsAboveMedian > medianIdx)
				return maxMedian;
		}
	}
	std::nth_element(err.begin(), err.begin() + medianIdx, err.end());
	return err[medianIdx];
}

void copyHypothesis(const GNHypothesis& h1, GNHypothesis& h2)
{
	copy33(h1.E, h2.E);
	copy33(h1.R, h2.R);
	copy33(h1.TR, h2.TR);
	copy31(h1.t, h2.t);
	h2.cost = h1.cost;
}

Matrix3d findEssentialMatGN(common::scan_t pts1, common::scan_t pts2, 
		Matrix3d& R0, Vector3d& t0, Matrix3d& R2, Vector3d& t2,
		int n_hypotheses, int n_GNiters, 
		bool withNormalization, bool optimizedCost)
{
	// Init
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(0);
	std::uniform_int_distribution<> dist(0, pts1.size() - 1);
	GNHypothesis bestModel(R0, t0);
	
	// Fully score initial hypothesis
	time_cat_verbose(common::TimeCatHypoScoring);
	if(optimizedCost)
		bestModel.cost = score_LMEDS2(pts1, pts2, bestModel.E, 1e10);
	else
		bestModel.cost = score_LMEDS(pts1, pts2, bestModel.E, 1e10);

	//if(record_all_hypotheses)
	//	all_hypotheses.push_back(bestModel.E_.clone());
	GNHypothesis model;
	for(int i = 0; i < n_hypotheses; i++)
	{
		// Get subset
		common::scan_t subset1;
		common::scan_t subset2;
		getSubset(pts1, pts2, subset1, subset2, 5, dist, rng);

		// Initialize GN algorithm with best model and then perform 10 GN iterations
		time_cat_verbose(common::TimeCatHypoGen);
		copyHypothesis(bestModel, model);
		for(int j = 0; j < n_GNiters; j++)
			GN_step(subset1, subset2, model.R, model.TR, model.E, model.R, model.TR, model.t, 1, withNormalization);

		// Partially score hypothesis (terminate early if cost exceeds lowest cost)
		time_cat_verbose(common::TimeCatHypoScoring);
		if(optimizedCost)
			model.cost = score_LMEDS2(pts1, pts2, model.E, bestModel.cost);
		else
			model.cost = score_LMEDS(pts1, pts2, model.E, bestModel.cost);
		if(model.cost < bestModel.cost)
			copyHypothesis(model, bestModel);
		//if(record_all_hypotheses)
		//	all_hypotheses.push_back(bestModel.E_.clone());
	}
	time_cat_verbose(common::TimeCatNone);
	R2 = bestModel.R_map;
	t2 = bestModel.t_map;
	return bestModel.E_map;
}

}
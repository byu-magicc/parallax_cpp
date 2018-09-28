#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include <vector>
#include "ptr/GN_step.h"


using namespace std;

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

void mult3pt(const double* A, const cv::Point2d pt, double* y)
{
	y[0] = A[0]*pt.x + A[1]*pt.y + A[2];
	y[1] = A[3]*pt.x + A[4]*pt.y + A[5];
	y[2] = A[6]*pt.x + A[7]*pt.y + A[8];
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

double multptv(const cv::Point2d pt, const double* v)
{
	return pt.x*v[0] + pt.y*v[1] + v[2];
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

void printPoints(const char* s, vector<cv::Point2f> pts)
{
	printf("%s: (2x%d)\n", s, pts.size());
	printf("[");
	for(int i = 0; i < pts.size(); i++)
		printf("%10.4f ", pts[i].x);
	printf("\n ");
	for(int i = 0; i < pts.size(); i++)
		printf("%10.4f ", pts[i].y);
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
	
	// R = eye(3) + sinc(theta)*wx + 1/2*sinc(theta/2)^2*wx^2;	
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
		  const cv::Point2d p1, const cv::Point2d p2, double* deltaR)
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
double residual(const double* E, const cv::Point2d p1, const cv::Point2d p2)
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
		const cv::Point2d p1, const cv::Point2d p2, double* deltaR)
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
double residual_without_normalization(const double* E, const cv::Point2d p1, const cv::Point2d p2)
{
	// r = p2'*E*p1;
	double L[3];
	mult3pt(E, p1, L);
	double r = multptv(p2, L);	
	return r;
}

//function [E, R, TR] = GN_step(pts1, pts2, R, TR)
void GN_step(const vector<cv::Point2d> pts1, const vector<cv::Point2d> pts2, const double* R, const double* TR,
		  double* E2, double* R2, double* TR2, double* t2, int method, bool withNormalization, bool useLM)
{
	// Gauss-Newton
	// N = size(pts1, 2);
	// r = zeros(N, 1);
	// J = zeros(N, 5);
	double lambda = 1e-4;
	int N = (int)pts1.size();
	cv::Mat r_mat = cv::Mat::zeros(N, 1, CV_64F);
	cv::Mat r2_mat = cv::Mat::zeros(N, 1, CV_64F);
	cv::Mat J_mat = cv::Mat::zeros(N, 5, CV_64F);
	cv::Mat dx_mat = cv::Mat::zeros(5, 1, CV_64F);	
	double* r = r_mat.ptr<double>();
	double* r2 = r2_mat.ptr<double>();
	double* J = J_mat.ptr<double>();
	double* dx = dx_mat.ptr<double>();
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
		cv::Mat JtJ_mat = J_mat.t()*J_mat;
		cv::Mat Jtr_mat = J_mat.t()*r_mat;
		while(true)
		{
			// dx = -(JtJ + lambda*diag(diag(JtJ)))\J'*r;
			cv::solve(-(JtJ_mat + lambda*cv::Mat::diag(JtJ_mat.diag())), Jtr_mat, dx_mat, cv::DECOMP_NORMAL);
	
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
			
			if(cv::norm(r2_mat) <= cv::norm(r_mat))
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
		// QR takes about twice as long on Windows, all the other methods have similar computation times.
		if(method == 1)
			cv::solve(-J_mat, r_mat, dx_mat, cv::DECOMP_NORMAL);
		else if(method == 2)
			cv::solve(-J_mat, r_mat, dx_mat, cv::DECOMP_NORMAL | cv::DECOMP_QR);
		else if(method == 3)
			cv::solve(-J_mat, r_mat, dx_mat, cv::DECOMP_NORMAL | cv::DECOMP_SVD);
		else if(method == 4)
			cv::solve(-J_mat.t()*J_mat, J_mat.t()*r_mat, dx_mat, cv::DECOMP_CHOLESKY);
		else if(method == 5);
		else
			CV_Assert(0);
		
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

GNHypothesis::GNHypothesis() : cost(0), 
	E_(cv::Mat::zeros(3, 3, CV_64F)), R_(cv::Mat::eye(3, 3, CV_64F)), 
	TR_(cv::Mat::eye(3, 3, CV_64F)), t_(cv::Mat::zeros(3, 1, CV_64F))
{
	R = R_.ptr<double>();
	TR = TR_.ptr<double>();
	E = E_.ptr<double>();
	t = t_.ptr<double>();
	RT_getE(R, TR, t, E);
}
	
GNHypothesis::GNHypothesis(cv::Mat R0, cv::Mat t0) : cost(0), 
	E_(cv::Mat::zeros(3, 3, CV_64F)), R_(R0.clone()), 
	TR_(cv::Mat::eye(3, 3, CV_64F)), t_(cv::Mat::zeros(3, 1, CV_64F))
{
	R = R_.ptr<double>();
	TR = TR_.ptr<double>();
	E = E_.ptr<double>();
	t = t_.ptr<double>();
	t2TR(t0.ptr<double>(), TR);
	RT_getE(R, TR, t, E);
}

void getSubset(vector<cv::Point2d>& pts1, vector<cv::Point2d>& pts2, vector<cv::Point2d>& subset1, vector<cv::Point2d>& subset2, int modelPoints, cv::RNG& rng)
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
			idx = rng.uniform(0, count);

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

double score_LMEDS(vector<cv::Point2d>& pts1, vector<cv::Point2d>& pts2, double* E, double maxMedian)
{
	int costsAboveMedian = 0;
	int n_pts = (int)pts1.size();
	int medianIdx = (int)n_pts/2;
	vector<double> err = vector<double>(n_pts, 0);
	cv::Matx33d E_matx(E);
	for (int i = 0; i < n_pts; i++)
	{
		cv::Vec3d x1(pts1[i].x, pts1[i].y, 1.);
		cv::Vec3d x2(pts2[i].x, pts2[i].y, 1.);
		cv::Vec3d Ex1 = E_matx * x1;
		cv::Vec3d Etx2 = E_matx.t() * x2;
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
// 		a = E[0]*pts1[i].x + E[1]*pts1[i].y + E[2];
// 		b = E[3]*pts1[i].x + E[4]*pts1[i].y + E[5];
// 		c = E[6]*pts1[i].x + E[7]*pts1[i].y + E[8];
// 		
// 		// distance to the line can be found be taking the dot product
// 		// of the normalized line and the homogeneous coordinate of the point.
// 		// To normalize the line, divide a, b, and c by sqrt(a^2 + b^2), 
// 		// so that vector [a_new b_new] is a unit vector.
// 		// We normalize when the square of the distance is computed, thus avoiding a sqrt.
// 		dotprod = (pts2[i].x*a + pts2[i].y*b + c);
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

double score_LMEDS2(vector<cv::Point2d>& pts1, vector<cv::Point2d>& pts2, double* E, double maxMedian)
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
		a2 = E[0]*pts1[i].x + E[1]*pts1[i].y + E[2];
		b2 = E[3]*pts1[i].x + E[4]*pts1[i].y + E[5];
		c2 = E[6]*pts1[i].x + E[7]*pts1[i].y + E[8];
		
		a1 = E[0]*pts2[i].x + E[3]*pts2[i].y + E[6];
		b1 = E[1]*pts2[i].x + E[4]*pts2[i].y + E[7];
		c1 = E[2]*pts2[i].x + E[5]*pts2[i].y + E[8];
		
		// Distance to the line can be found be taking the dot product
		// of the normalized line and the homogeneous coordinate of the point.
		// To normalize the line, divide a, b, and c by sqrt(a^2 + b^2), 
		// so that vector [a_new b_new] is a unit vector.
		// We normalize when the square of the distance is computed, thus avoiding a sqrt.
		dotprod = (pts2[i].x*a2 + pts2[i].y*b2 + c2);
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

cv::Mat findEssentialMatGN(vector<cv::Point2d> pts1, vector<cv::Point2d> pts2, 
		cv::Mat& R0, cv::Mat& t0, cv::Mat& R2, cv::Mat& t2, std::vector<cv::Mat>& all_hypotheses,
		int n_hypotheses, int n_GNiters, 
		bool withNormalization, bool optimizedCost, bool record_all_hypotheses)
{
	// Init
	cv::RNG rng(cv::getCPUTickCount());
	GNHypothesis bestModel(R0, t0);
	
	// Fully score initial hypothesis
	if(optimizedCost)
		bestModel.cost = score_LMEDS2(pts1, pts2, bestModel.E, 1e10);
	else
		bestModel.cost = score_LMEDS(pts1, pts2, bestModel.E, 1e10);
	if(record_all_hypotheses)
		all_hypotheses.push_back(bestModel.E_.clone());
	GNHypothesis model;
	for(int i = 0; i < n_hypotheses; i++)
	{
		// Get subset
		vector<cv::Point2d> subset1;
		vector<cv::Point2d> subset2;
		getSubset(pts1, pts2, subset1, subset2, 5, rng);

		// Initialize GN algorithm with best model and then perform 10 GN iterations
		copyHypothesis(bestModel, model);
		for(int j = 0; j < n_GNiters; j++)
			GN_step(subset1, subset2, model.R, model.TR, model.E, model.R, model.TR, model.t, 1, withNormalization);

		// Partially score hypothesis (terminate early if cost exceeds lowest cost)
		if(optimizedCost)
			model.cost = score_LMEDS2(pts1, pts2, model.E, bestModel.cost);
		else
			model.cost = score_LMEDS(pts1, pts2, model.E, bestModel.cost);
		if(model.cost < bestModel.cost)
			copyHypothesis(model, bestModel);
		if(record_all_hypotheses)
			all_hypotheses.push_back(bestModel.E_.clone());
	}
	R2 = bestModel.R_;
	t2 = bestModel.t_;
	return bestModel.E_;
}
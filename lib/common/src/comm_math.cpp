#include "comm_math.h"
#include <eigen3/Eigen/Eigen>
#include <string>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;
using namespace Eigen;

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/eigen.hpp"

//#define RED_TEXT "\033[31m[49m"
#define RED_TEXT "\033[101m"
#define GREEN_TEXT "\033[32m"
#define BLACK_TEXT "\033[0m\033[49m"

void common::printMatricesComp(string s, const MatrixXd A1, const MatrixXd A2, float eps, int sfAfter, int sfBefore)
{
	if(A1.rows() != A2.rows() || A1.cols() != A2.cols())
	{
		printf("Error: Cannot compare arrays of different sizes (%dx%d and %dx%d)\n", (int)A1.rows(), (int)A1.cols(), (int)A2.rows(), (int)A2.cols());
		exit(EXIT_FAILURE);
	}
	printf("%s: (%dx%d)\n", s.c_str(), (int)A1.rows(), (int)A1.cols());
	char format[40];
	sprintf(format, "%%%d.%df", sfBefore + sfAfter, sfAfter);
	for (int r = 0; r < A1.rows(); r++)
	{
		if (r == 0)
			printf("[");
		else
			printf("\n ");
		for (int c = 0; c < A1.cols(); c++)
		{
			if (c > 0)
				printf(" ");
			bool equal = fabs(A1(r, c) - A2(r, c)) <= eps;
			if(!equal)
				printf(RED_TEXT);
			printf(format, A1(r, c));
			if(!equal)
				printf(BLACK_TEXT);
		}
	}
	printf("]\n");
}

void common::checkMatrices(string s1, string s2, const MatrixXd A1, const MatrixXd A2, int dir, float eps, int sfAfter, int sfBefore, bool block)
{
	// First, find out if the arrays are equal
	MatrixXd err = (A1 - A2).cwiseAbs();
	if(err.maxCoeff() < eps)
		return;
	if(block)
	{
		if(dir != -1)
			printf(RED_TEXT "Error: Matrix comparison failed in direction %d." BLACK_TEXT "\n", dir);
		else
			printf(RED_TEXT "Error: Matrix comparison failed." BLACK_TEXT "\n");
	}

	// If the arrays are not equal, print them both out
	printMatricesComp(s1, A1, A2, eps, sfAfter, sfBefore);
	printMatricesComp(s2, A2, A1, eps, sfAfter, sfBefore);
	printf("\n");
	if(block)
		exit(EXIT_FAILURE);
}

Matrix3d common::skew(Vector3d v)
{
	Matrix3d Tx;
	Tx << 0,    -v(2),   v(1),
		  v(2),  0,     -v(0),
		 -v(1),  v(0),   0;
	return Tx;
}

Vector3d common::vex(Matrix3d Tx)
{
	Vector3d w;
	w << Tx(2, 1), Tx(0, 2), Tx(1, 0);
	return w;
}

double common::sinc(double x)
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

Matrix3d common::R1toR2(Matrix3d R1, Vector3d t_)
{
	// We need to flip all unit vectors of R1 = [r1 r2 r3] across the translation direction.
	// It is easier to do this in two steps:
	// 1. Flip the vectors [r1 r2 r3] across the plane perpendicular to the translation direction. 
	//    This is done by subtracting the projection of [r1 r2 r3] onto t twice. This is the
	//    householder transformation.
	// 2. Negate the vectors [r1 r2 r3]. In the end, the components parallel to the translation
	//    direction have been flipped twice, but the components perpendicular to t have only
	//    been flipped once.
	Vector3d t = unit(t_);
	Matrix3d dotprod_rep = (t.transpose() * R1).replicate<3, 1>();
	Matrix3d translation_rep = t.replicate<1, 3>();
	Matrix3d R2 = -(R1 - 2 * translation_rep.cwiseProduct(dotprod_rep));
	return R2;
}

Matrix3d common::vecToR(Vector3d w)
{
	double theta = w.norm();
	Matrix3d wx = skew(w);

	// R = eye(3) + sinc(theta)*wx + 0.5*sinc(theta/2)^2*wx^2;	
	double sinc2 = sinc(theta / 2);
	Matrix3d R = Matrix3d::Identity() + sinc(theta)*wx + 0.5 * sinc2 * sinc2 * wx * wx;
	return R;
}

Vector3d common::RtoVec(Matrix3d R)
{
	// WARNING: BROKEN for angles near pi
	// The rodrigues formula gives us
	// R = I + sin(theta)*wx_hat + (1 - cos(theta))*wx_hat^2
	// Notice that the first and third terms are symmetric,
	// while the second term is skew-symmetric and has no diagonal components.
	// The diagonal components of the matrix are are an easy way to get the "I + (1 - cos)" terms
	// We can cancel out symmetric terms using (R - R') / 2. This allows us to get the "sin" term.
	double cos_theta = (R.trace() - 1) / 2;
	Vector3d sin_theta_w_hat = vex(R - R.transpose()) / 2;
	double sin_theta = sin_theta_w_hat.norm();

	// To get theta, we could use atan2, but it is slightly more accurate to 
	// use acos or asin, depending on which area of the unit circle we are in.
	// For best accuracy, we should avoid using acos or asin if the value is near 1.
	// An easy way to prevent this is to simply use whichever value is smallest.
	// (the multiplication by 2 slightly alters the asin/acos regions and was determined hueristically)
	// We need to handle theta = [0, pi]. (quadrant 1 and 2)
	// Theta by definition is norm(w), so it can't be negative. Angles larger than pi
	// end up wrapping around the other way, whichever is shortest.
	// theta = atan2(sin_theta, cos_theta);
	double theta;
	if (fabs(cos_theta) < fabs(sin_theta)*2)
		theta = acos(cos_theta);
	else if (cos_theta > 0)
		theta = asin(sin_theta);
	else //if (cos_theta < 0)
		theta = M_PI - asin(sin_theta);
	Vector3d w = sin_theta_w_hat / sinc(theta);
	return w;
}

double common::R_norm(Matrix3d R)
{
	// The rodrigues formula gives us
	// R = I + sin(theta)*wx_hat + (1 - cos(theta))*wx_hat^2
	// Notice that the first and third terms are symmetric,
	// while the second term is skew-symmetric and has no diagonal components.
	// The diagonal components of the matrix are are an easy way to get the "I + (1 - cos)" terms
	// We can cancel out symmetric terms using (R - R') / 2. This allows us to get the "sin" term.
	double cos_theta = (R.trace() - 1) / 2;
	Vector3d sin_theta_w_hat = vex(R - R.transpose()) / 2;
	double sin_theta = sin_theta_w_hat.norm();

	// To get theta, we could use atan2, but it is slightly more accurate to 
	// use acos or asin, depending on which area of the unit circle we are in.
	// For best accuracy, we should avoid using acos or asin if the value is near 1.
	// An easy way to prevent this is to simply use whichever value is smallest.
	// (the multiplication by 2 slightly alters the asin/acos regions and was determined hueristically)
	// We need to handle theta = [0, pi]. (quadrant 1 and 2)
	// Theta by definition is norm(w), so it can't be negative. Angles larger than pi
	// end up wrapping around the other way, whichever is shortest.
	// theta = atan2(sin_theta, cos_theta);
	double theta;
	if (fabs(cos_theta) < fabs(sin_theta)*2)
		theta = acos(cos_theta);
	else if (cos_theta > 0)
		theta = asin(sin_theta);
	else //if (cos_theta < 0)
		theta = M_PI - asin(sin_theta);
	return theta;
}

Matrix3d common::decomposeEssentialMat(const Matrix3d& E, Matrix3d& R1, Matrix3d& R2, Vector3d& t)
{
	cv::Mat E_cv, R1_cv, R2_cv, t_cv;
	cv::eigen2cv(E, E_cv);
	cv::decomposeEssentialMat(E_cv, R1_cv, R2_cv, t_cv);
	cv::cv2eigen(R1_cv, R1);
	cv::cv2eigen(R2_cv, R2);
	cv::cv2eigen(t_cv, t);
}

Vector4d common::err_truth(const Matrix3d& R1, const Matrix3d& R2, const Vector3d& t, const Matrix4d& RT)
{
	// Extract R and T from the 4x4 homogenous transformation
	Matrix3d R_truth = RT.block<3, 3>(0, 0);
	Vector3d t_truth = RT.block<3, 1>(0, 3);
	t_truth = unit(t_truth);

	// Calculate errors for R1, R2, t, and -t
	Vector3d t_unit = unit(t);
	double err_t1 = acos(t_unit.dot(t_truth));
	double err_t2 = acos(-t_unit.dot(t_truth));
	double err_R1 = R_norm(R1 * R_truth.transpose());
	double err_R2 = R_norm(R2 * R_truth.transpose());
	double err_t_min = min(err_t1, err_t2);
	double err_R_min = min(err_R1, err_R2);

	// See if the first rotation and translation are the correct one
	bool correct_R1 = err_R1 < err_R2;
	bool correct_t1 = err_t1 < err_t2;

	// Output result
	Vector4d err;
	err << err_R_min, err_t_min, (int)correct_R1, (int)correct_t1;
	return err;
}

Vector4d common::err_truth(const Matrix3d& R1, const Vector3d& t, const Matrix4d& RT)
{
	Matrix3d R2 = R1toR2(R1, t);
	return err_truth(R1, R2, t, RT);
}

Vector4d common::err_truth(const Matrix3d& E, const Matrix4d& RT)
{
	Matrix3d R1, R2;
	Vector3d t;
	decomposeEssentialMat(E, R1, R2, t);
	return err_truth(R1, R2, t, RT);
}

Vector2d common::dist_E(const Matrix3d& E1, const Matrix3d& E2)
{
	Matrix3d R11, R12, R21, R22;
	Vector3d t1, t2;
	decomposeEssentialMat(E1, R11, R12, t1);
	decomposeEssentialMat(E2, R21, R22, t2);
	double err1_T = acos(t1.dot(t2));
	double err2_T = acos(-t1.dot(t2));
	double err1_R = R_norm(R11 * R21.transpose());
	double err2_R = R_norm(R11 * R22.transpose());
	double err3_R = R_norm(R12 * R21.transpose());
	double err4_R = R_norm(R12 * R22.transpose());
	double err_t_min = min(err1_T, err2_T);
	double err_R_min = min(min(err1_R, err2_R), min(err3_R, err4_R));
	Vector2d err;
	err << err_R_min, err_t_min;
	return err;
}

void common::undistort_points(const scan_t& pts, scan_t& pts_u, Matrix3d camera_matrix)
{
	// Note: We aren't inverting actually the actual camera matrix. We assume 
	// the camera matrix is formatted as expected:
	// [fx 0  cx
	//  0  fy cy
	//  0  0  1]
	double inv_fx = 1./camera_matrix(0, 0);
	double inv_fy = 1./camera_matrix(1, 1);
	double cx = camera_matrix(0, 2);
	double cy = camera_matrix(1, 2);
	pts_u = scan_t(pts.size());
	for(int i = 0; i < pts.size(); i++)
		pts_u[i] << (pts[i](0) - cx)*inv_fx, (pts[i](1) - cy)*inv_fy;
}

Vector2d common::sampson_err(const Matrix3d& E, const scan_t& pts1, const scan_t& pts2)
{
	int n_pts = pts1.size();
	int medianIdx = (int)n_pts / 2;
	vector<double> err = vector<double>(n_pts, 0);
	double err_total = 0;
	for(int i = 0; i < n_pts; i++)
	{
		Vector3d x1;
		x1 << pts1[i](0), pts1[i](1), 1.;
		Vector3d x2;
		x2 << pts2[i](0), pts2[i](1), 1.;
		Vector3d Ex1 = E * x1;
		Vector3d Etx2 = E.transpose() * x2;
		double x2tEx1 = x2.dot(Ex1);
		
		double a = Ex1[0] * Ex1[0];
		double b = Ex1[1] * Ex1[1];
		double c = Etx2[0] * Etx2[0];
		double d = Etx2[1] * Etx2[1];
		double err_i = x2tEx1 * x2tEx1 / (a + b + c + d);
		err[i] = err_i;
		err_total += err_i;
	}
	std::nth_element(err.begin(), err.begin() + medianIdx, err.end());
	double med_err = err[medianIdx];
	double mean_err = err_total / n_pts;
	Vector2d err_result;
	err_result << med_err, mean_err;
	return err_result;
}

void common::five_point(const scan_t& subset1, const scan_t& subset2, vector<Matrix3d>& hypotheses)
{
	// Convert to Point2f
	int n_pts = subset1.size();
	if(n_pts != 5)
	{
		printf("The five-point algorithm doesn't accept %d points\n", n_pts);
		exit(EXIT_FAILURE);
	}	
	vector<cv::Point2d> subset1_cv = vector<cv::Point2d>(n_pts);
	vector<cv::Point2d> subset2_cv = vector<cv::Point2d>(n_pts);
	for (int i = 0; i < n_pts; i++)
	{
		subset1_cv[i].x = subset1[i](0);
		subset1_cv[i].y = subset1[i](1);
		subset2_cv[i].x = subset2[i](0);
		subset2_cv[i].y = subset2[i](1);
	}

	// Calc (multiple) hypotheses using 5-point algorithm
	cv::Mat E_cv = findEssentialMat(subset1_cv, subset2_cv, cv::Mat::eye(3, 3, CV_64F));
	if(E_cv.rows % 3 != 0 || (E_cv.rows > 0 && E_cv.cols != 3))
	{
		printf("Invalid essential matrix size: [%d x %d]\n", E_cv.rows, E_cv.cols);
		exit(EXIT_FAILURE);
	}
	int n_hypotheses = E_cv.rows / 3;
	hypotheses.resize(n_hypotheses);
	for(int i = 0; i < n_hypotheses; i++)
	{
		Map<Matrix<double, 3, 3, RowMajor>> E_i = Map<Matrix<double, 3, 3, RowMajor>>(&E_cv.at<double>(i * 3, 0));
		hypotheses[i] = E_i;
	}
}

void common::perspectiveTransform(const scan_t& pts1, scan_t& pts2, const Matrix3d& H)
{
	int n_pts = pts1.size();
	pts2.resize(pts1.size());
	vector<cv::Point2d> pts1_cv = vector<cv::Point2d>(n_pts);
	vector<cv::Point2d> pts2_cv = vector<cv::Point2d>(n_pts);
	for (int i = 0; i < n_pts; i++)
	{
		pts1_cv[i].x = pts1[i](0);
		pts1_cv[i].y = pts1[i](1);
	}
	cv::Mat H_cv;
	cv::eigen2cv(H, H_cv);
	cv::perspectiveTransform(pts1_cv, pts2_cv, H_cv);
	for(int i = 0; i < n_pts; i++)
		pts2[i] << pts2_cv[i].x, pts2_cv[i].y;
}

void common::getParallaxField(const Matrix3d& E, const Vector2d& loc, Vector2d& perpendicular, Vector2d& parallel)
{
	Vector3d pt;
	pt << loc(0), loc(1), 1;
	Vector3d line = E * pt;
	perpendicular << line(0), line(1);
	perpendicular = unit(perpendicular);
	parallel << perpendicular(1), -perpendicular(0);
}

int common::chierality(const scan_t& pts1, const scan_t& pts2, const Matrix3d& R, const Vector3d& t)
{
	// Derotate points
	// Use rotation as a euclidian homography matrix to transform points to 2nd frame
	// H_e = R + T_x*n
	const Matrix3d& H_e = R;
	scan_t pts1_warped;
	perspectiveTransform(pts1, pts1_warped, H_e);

	// Calculate the point velocities (velocity = parallax + actual velocity)
	scan_t pointVelocities = scan_t(pts1.size());
	for (int i = 0; i < pts1.size(); i++)
		pointVelocities[i] = (pts2[i] - pts1_warped[i]);

	// Use the translation as a skew symmetric matrix to calculate the field normal and parallel vectors
	scan_t fieldPerpendicular = scan_t(pts1.size());
	scan_t fieldParallel = scan_t(pts1.size());
	for (int i = 0; i < pts1.size(); i++)
		getParallaxField(skew(t), pts1_warped[i], fieldPerpendicular[i], fieldParallel[i]);

	// Count how many points have a positive parallax component
	int numPosDotProduct = 0;
	for (int i = 0; i < pts1.size(); i++)
		if (pointVelocities[i].dot(fieldParallel[i]) > 0)
			numPosDotProduct++;
}
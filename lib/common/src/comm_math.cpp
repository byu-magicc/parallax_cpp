#include "comm_math.h"
#include <eigen3/Eigen/Eigen>
#include <string>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;
using namespace Eigen;

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

Vector2d common::err_truth(const Matrix3d& R1, const Vector3d& t_, const Matrix4d& RT)
{
	// Extract R and T from the 4x4 homogenous transformation
	Matrix3d R_truth = RT.block<3, 3>(0, 0);
	Vector3d t_truth = RT.block<3, 1>(0, 3);
	t_truth = unit(t_truth);

	// Calculate errors for R1, R2, t, and -t
	Matrix3d R2 = R1toR2(R1, t_);
	Vector3d t = unit(t_);
	double err_t1 = acos(t.dot(t_truth));
	double err_t2 = acos(-t.dot(t_truth));
	double err_R1 = RtoVec(R1 * R_truth.transpose()).norm();
	double err_R2 = RtoVec(R2 * R_truth.transpose()).norm();
	double err_t_min = min(err_t1, err_t2);
	double err_R_min = min(err_R1, err_R2);

	// Output result
	Vector2d err;
	err << err_R_min, err_t_min;
	return err;
}
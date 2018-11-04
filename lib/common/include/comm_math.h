#ifndef COMM_MATH_H
#define COMM_MATH_H

#include <eigen3/Eigen/Eigen>

namespace common
{

#define unit(vec)  ((vec) / (vec).norm())

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > scan_t;
typedef std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > scan4_t;

template <typename T, typename T2>
void fill_rnd(Eigen::MatrixBase<T>& A, T2 dist, std::default_random_engine& rng)
{
	for(int i = 0; i < A.rows(); i++)
		for(int j = 0; j < A.cols(); j++)
			A(i, j) = dist(rng);
}

// Define colors (https://misc.flogisoft.com/bash/tip_colors_and_formatting#background)
#define RED_TEXT "\033[91m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[93m"
#define BLACK_TEXT "\033[0m\033[49m"

void printMatricesComp(std::string s, const Eigen::MatrixXd A1, const Eigen::MatrixXd A2, float eps = 1e-6, int sfAfter = 4, int sfBefore = 5);

void checkMatrices(std::string s1, std::string s2, const Eigen::MatrixXd A1, const Eigen::MatrixXd A2, int dir = -1, float eps = 1e-6, int sfAfter = 4, int sfBefore = 5, bool block = true);

Eigen::Matrix3d skew(Eigen::Vector3d v);

Eigen::Vector3d vex(Eigen::Matrix3d Tx);

double sinc(double x);

Eigen::Matrix3d R1toR2(Eigen::Matrix3d R1, Eigen::Vector3d t);

Eigen::Matrix3d vecToR(Eigen::Vector3d v);

Eigen::Vector3d RtoVec(Eigen::Matrix3d R);

double R_norm(Eigen::Matrix3d R);

Eigen::Matrix3d decomposeEssentialMat(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t);

void RT_split(const Eigen::Matrix4d& RT, Eigen::Matrix3d& R, Eigen::Vector3d& t);

Eigen::Matrix4d RT_combine(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

Eigen::Vector4d err_truth(const Eigen::Matrix3d& R1_est, const Eigen::Matrix3d& R2_est, const Eigen::Vector3d& t_est, const Eigen::Matrix4d& RT_truth);

Eigen::Vector4d err_truth(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, const Eigen::Matrix4d& RT_truth);

Eigen::Vector4d err_truth(const Eigen::Matrix3d& E, const Eigen::Matrix4d& RT_truth);

Eigen::Vector2d dist_E(const Eigen::Matrix3d& E1, const Eigen::Matrix3d& E2);

void undistort_points(const scan_t& pts, scan_t& pts1_u, Eigen::Matrix3d& camera_matrix);

void project_points(const std::vector<Eigen::Vector3d>& Pts, scan_t& pts, std::vector<float>& dist, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, const Eigen::Matrix3d& camera_matrix);

void project_points(const std::vector<Eigen::Vector3d>& Pts, scan_t& pts, std::vector<float>& dist, const Eigen::Matrix4d& RT, const Eigen::Matrix3d& camera_matrix);

Eigen::Vector2d sampson_err(const Eigen::Matrix3d& E, const scan_t& pts1, const scan_t& pts2);

void five_point(const scan_t& subset1, const scan_t& subset2, std::vector<Eigen::Matrix3d>& hypotheses);

void perspectiveTransform(const scan_t& pts1, scan_t& pts2, const Eigen::Matrix3d& H);

void getParallaxField(const Eigen::Matrix3d& E, const Eigen::Vector2d& loc, Eigen::Vector2d& perpendicular, Eigen::Vector2d& parallel);

void lineIntersection(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, Eigen::Vector3d& n, double& err);

void triangulatePoints(const scan_t& pts1, const scan_t& pts2, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, std::vector<Eigen::Vector3d>& Pts1, std::vector<Eigen::Vector3d>& Pts2, std::vector<double>& err);

int chierality(const scan_t& pts1, const scan_t& pts2, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);

}

#endif // COMM_MATH_H
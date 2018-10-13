#ifndef COMM_MATH_H
#define COMM_MATH_H

#include <eigen3/Eigen/Eigen>

namespace common
{

#define unit(vec)  ((vec) / (vec).norm())

// template <typename T>
// Eigen::MatrixBase<T> unit(Eigen::MatrixBase<T>& val)
// {
//     return val / val.norm();
// }

Eigen::Matrix3d skew(Eigen::Vector3d v);

Eigen::Vector3d vex(Eigen::Matrix3d Tx);

double sinc(double x);

Eigen::Matrix3d R1toR2(Eigen::Matrix3d R1, Eigen::Vector3d t);

Eigen::Matrix3d vecToR(Eigen::Vector3d v);

Eigen::Vector3d RtoVec(Eigen::Matrix3d R);

double R_norm(Eigen::Matrix3d R);

Eigen::Matrix3d decomposeEssentialMat(const Eigen::Matrix3d& E, Eigen::Matrix3d& R1, Eigen::Matrix3d& R2, Eigen::Vector3d& t);

Eigen::Vector2d err_truth(const Eigen::Matrix3d& R1_est, const Eigen::Matrix3d& R2_est, const Eigen::Vector3d& t_est, const Eigen::Matrix4d& RT_truth);

Eigen::Vector2d err_truth(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, const Eigen::Matrix4d& RT_truth);

Eigen::Vector2d err_truth(const Eigen::Matrix3d& E, const Eigen::Matrix4d& RT_truth);

Eigen::Vector2d dist_E(const Eigen::Matrix3d& E1, const Eigen::Matrix3d& E2);

}

#endif // COMM_MATH_H
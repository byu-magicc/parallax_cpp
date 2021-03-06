#ifndef GNSAC_PTR_OCV_H
#define GNSAC_PTR_OCV_H

#include "opencv2/core/core.hpp"
#include "common/common.h"
#include <eigen3/Eigen/Dense>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/eigen.hpp"
#include <vector>
#include <iostream>

namespace gnsac_ptr_opencv
{

double sinc(double x);

void skew(const double* w, double* T);

void mult3pt(const double* A, const cv::Point2d pt, double* y);

void mult31(const double* A, const double* x, double* y);

double mult3vtv(const double* x, const double* y);

double multptv(const cv::Point2d pt, const double* v);

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

void printPoints(const char* s, std::vector<cv::Point2f> pts);

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
		  const cv::Point2d p1, const cv::Point2d p2, double* deltaR);

double residual(const double* E, const cv::Point2d p1, const cv::Point2d p2);

void GN_step(const std::vector<cv::Point2d> pts1, const std::vector<cv::Point2d> pts2, const double* R, const double* TR,
		  double* E2, double* R2, double* TR2, double* t2, int method, bool withNormalization = true, bool useLM = false);

class GNHypothesis
{
public:
	GNHypothesis();
	GNHypothesis(cv::Mat R0, cv::Mat t0);
	
	cv::Mat E_;
	cv::Mat R_;
	cv::Mat TR_;
	cv::Mat t_;
	double* R;
	double* TR;
	double* E;
	double* t;
	double cost;
};

void getSubset(std::vector<cv::Point2d>& pts1, std::vector<cv::Point2d>& pts2, std::vector<cv::Point2d>& subset1, std::vector<cv::Point2d>& subset2, int modelPoints, cv::RNG& rng);

double score_LMEDS(std::vector<cv::Point2d>& pts1, std::vector<cv::Point2d>& pts2, double* E, double maxMedian);

double score_LMEDS2(std::vector<cv::Point2d>& pts1, std::vector<cv::Point2d>& pts2, double* E, double maxMedian);

void copyHypothesis(const GNHypothesis& h1, GNHypothesis& h2);

cv::Mat findEssentialMatGN(std::vector<cv::Point2d> pts1, std::vector<cv::Point2d> pts2,
		cv::Mat& R0, cv::Mat& t0, cv::Mat& R2, cv::Mat& t2,
		int n_hypotheses, int n_GNiters,
		bool withNormalization = true, bool optimizedCost = false);

Eigen::Matrix3d findEssentialMatGN(common::scan_t pts1, common::scan_t pts2,
		Eigen::Matrix3d& R0, Eigen::Vector3d& t0, Eigen::Matrix3d& R2, Eigen::Vector3d& t2,
		int n_hypotheses, int n_GNiters,
		bool withNormalization = true, bool optimizedCost = false);

}

#endif //GNSAC_PTR_OCV_H
#ifndef __OPENCV_CALIB3D__HPP__
#define __OPENCV_CALIB3D__HPP__

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"

namespace cv_
{

	//! @addtogroup calib3d
	//! @{

	//! type of the robust estimation algorithm
	enum {
		LMEDS = 4, //!< least-median algorithm
		RANSAC = 8, //!< RANSAC algorithm
		RHO = 16, //!< RHO algorithm
	};

	enum {
		SOLVEPNP_ITERATIVE = 0,
		SOLVEPNP_EPNP = 1, //!< EPnP: Efficient Perspective-n-Point Camera Pose Estimation @cite lepetit2009epnp
		SOLVEPNP_P3P = 2, //!< Complete Solution Classification for the Perspective-Three-Point Problem @cite gao2003complete
		SOLVEPNP_DLS = 3, //!< A Direct Least-Squares (DLS) Method for PnP  @cite hesch2011direct
		SOLVEPNP_UPNP = 4  //!< Exhaustive Linearization for Robust Camera Pose and Focal Length Estimation @cite penate2013exhaustive

	};

	enum {
		CALIB_CB_ADAPTIVE_THRESH = 1,
		CALIB_CB_NORMALIZE_IMAGE = 2,
		CALIB_CB_FILTER_QUADS = 4,
		CALIB_CB_FAST_CHECK = 8
	};

	enum {
		CALIB_CB_SYMMETRIC_GRID = 1,
		CALIB_CB_ASYMMETRIC_GRID = 2,
		CALIB_CB_CLUSTERING = 4
	};

	enum {
		CALIB_USE_INTRINSIC_GUESS = 0x00001,
		CALIB_FIX_ASPECT_RATIO = 0x00002,
		CALIB_FIX_PRINCIPAL_POINT = 0x00004,
		CALIB_ZERO_TANGENT_DIST = 0x00008,
		CALIB_FIX_FOCAL_LENGTH = 0x00010,
		CALIB_FIX_K1 = 0x00020,
		CALIB_FIX_K2 = 0x00040,
		CALIB_FIX_K3 = 0x00080,
		CALIB_FIX_K4 = 0x00800,
		CALIB_FIX_K5 = 0x01000,
		CALIB_FIX_K6 = 0x02000,
		CALIB_RATIONAL_MODEL = 0x04000,
		CALIB_THIN_PRISM_MODEL = 0x08000,
		CALIB_FIX_S1_S2_S3_S4 = 0x10000,
		CALIB_TILTED_MODEL = 0x40000,
		CALIB_FIX_TAUX_TAUY = 0x80000,
		// only for stereo
		CALIB_FIX_INTRINSIC = 0x00100,
		CALIB_SAME_FOCAL_LENGTH = 0x00200,
		// for stereo rectification
		CALIB_ZERO_DISPARITY = 0x00400,
		CALIB_USE_LU = (1 << 17), //!< use LU instead of SVD decomposition for solving. much faster but potentially less precise
	};

	//! the algorithm for finding fundamental matrix
	enum {
		FM_7POINT = 1, //!< 7-point algorithm
		FM_8POINT = 2, //!< 8-point algorithm
		FM_LMEDS = 4, //!< least-median algorithm
		FM_RANSAC = 8  //!< RANSAC algorithm
	};

	int fivepoint(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model, bool time = false);

	void fivepoint_getCoeffMat(double *e, double *A);

	class EMEstimatorCallback : public PointSetRegistrator::Callback
	{
	public:
		int runKernel(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model) const;

	protected:
		void computeError(cv::InputArray _m1, cv::InputArray _m2, cv::InputArray _model, cv::OutputArray _err) const;

		float computeCost(cv::InputArray _m1, cv::InputArray _m2, cv::InputArray _model, int i1, int i2, float threshold) const;
	};

	CV_EXPORTS_W cv::Mat findEssentialMat(cv::InputArray points1, cv::InputArray points2,
		cv::InputArray cameraMatrix, int method = RANSAC,
		double prob = 0.999, double threshold = 1.0,
		cv::OutputArray mask = cv::noArray());

	CV_EXPORTS_W cv::Mat findEssentialMatPreempt(cv::InputArray points1, cv::InputArray points2, cv::InputArray cameraMatrix,
		float threshold = 1.0, int n_iters = 200, int blocksize = 30, cv::OutputArray mask = cv::noArray());
} // cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/calib3d/calib3d_c.h"
#endif

#endif

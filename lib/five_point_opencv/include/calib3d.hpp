#ifndef __CUSTOM_CALIB3D__HPP__
#define __CUSTOM_CALIB3D__HPP__

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "common.h"
#include "solvers.h"

namespace five_point_opencv
{
	class CV_EXPORTS LMSolver : public cv::Algorithm
	{
	public:
		class CV_EXPORTS Callback
		{
		public:
			virtual ~Callback() {}
			virtual bool compute(cv::InputArray param, cv::OutputArray err, cv::OutputArray J) const = 0;
		};

		virtual void setCallback(const cv::Ptr<LMSolver::Callback>& cb) = 0;
		virtual int run(cv::InputOutputArray _param0) const = 0;
	};

	CV_EXPORTS cv::Ptr<LMSolver> createLMSolver(const cv::Ptr<LMSolver::Callback>& cb, int maxIters);

	class CV_EXPORTS PointSetRegistrator : public cv::Algorithm
	{
	public:
		class CV_EXPORTS Callback
		{
		public:
			virtual ~Callback() {}
			virtual int runKernel(cv::InputArray m1, cv::InputArray m2, cv::OutputArray model) const = 0;
			virtual void computeError(cv::InputArray m1, cv::InputArray m2, cv::InputArray model, cv::OutputArray err) const = 0;
			virtual float computeCost(cv::InputArray m1, cv::InputArray m2, cv::InputArray model, int i1, int i2, float threshold) const = 0;			
			virtual bool checkSubset(cv::InputArray, cv::InputArray, int) const { return true; }
		};

		virtual void setCallback(const cv::Ptr<PointSetRegistrator::Callback>& cb) = 0;
		virtual bool run(cv::InputArray m1, cv::InputArray m2, cv::OutputArray model, cv::OutputArray mask) const = 0;
	};

	class FivePointSolver;

	CV_EXPORTS cv::Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const cv::Ptr<PointSetRegistrator::Callback>& cb,
		FivePointSolver* solver,
		int modelPoints, double threshold,
		double confidence = 0.99, int maxIters = 1000);

	CV_EXPORTS cv::Ptr<PointSetRegistrator> createLMeDSPointSetRegistrator(const cv::Ptr<PointSetRegistrator::Callback>& cb,
		FivePointSolver* solver,
		int modelPoints, double confidence = 0.99, int maxIters = 1000);

	CV_EXPORTS cv::Ptr<PointSetRegistrator> createPreemtiveRANSACPointSetRegistrator(const cv::Ptr<PointSetRegistrator::Callback>& cb,
		FivePointSolver* solver,
		int modelPoints, double threshold, int niters, int blocksize);

	template<typename T> inline int compressElems(T* ptr, const uchar* mask, int mstep, int count)
	{
		int i, j;
		for (i = j = 0; i < count; i++)
			if (mask[i*mstep])
			{
				// JHW: Replace element (unless they have the same index, in which case nothing would happen)
				if (i > j)
					ptr[j] = ptr[i];
				j++;
			}
		return j;
	}

	void swapElements(cv::Mat& m1, cv::Mat& m2, int i, int j);

	// Randomly permute the ordering of pairs of points using the Fisher-Yates shuffle algorithm
	// This algorithm is O(n) and for a certain number of elements always takes the same time to execute.
	void shuffleElements(cv::Mat& m1, cv::Mat& m2, cv::RNG& rng);


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

	int fivepoint(cv::InputArray _m1, cv::InputArray _m2, cv::OutputArray _model);

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
		int niters = 100, cv::OutputArray mask = cv::noArray());

	CV_EXPORTS_W cv::Mat findEssentialMatPreempt(cv::InputArray points1, cv::InputArray points2, cv::InputArray cameraMatrix,
		float threshold = 1.0, int n_iters = 200, int blocksize = 30, cv::OutputArray mask = cv::noArray());

	enum_str(consensus_t, consensus_t_str, consensus_RANSAC, consensus_LMEDS)

	class FivePointSolver : public common::ESolver
	{
	public:
		FivePointSolver(std::string yaml_filename, YAML::Node node, std::string result_directory);

	public:
		void generate_hypotheses(const common::scan_t& subset1, const common::scan_t& subset2, const common::EHypothesis& initial_guess, std::vector<common::EHypothesis>& hypotheses);

		void find_best_hypothesis(const common::scan_t& pts1, const common::scan_t& pts2, const Eigen::Matrix4d& RT_truth, common::EHypothesis& result);

		static FivePointSolver* getInstance();

	public:
		consensus_t consensus_alg;
		int n_subsets;
		double RANSAC_threshold;
		Eigen::Matrix4d RT_truth;

	private:
		static FivePointSolver* instance;
	};


	
} // cv

#ifndef DISABLE_OPENCV_24_COMPATIBILITY
#include "opencv2/calib3d/calib3d_c.h"
#endif

#endif

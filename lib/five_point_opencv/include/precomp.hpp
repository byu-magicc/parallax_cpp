/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#ifndef __OPENCV_PRECOMP__H__
#define __OPENCV_PRECOMP__H__

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/core/ocl.hpp"

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/calib3d/calib3d_tegra.hpp"
#else
#define GET_OPTIMIZED(func) (func)
#endif


namespace cv_
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

	CV_EXPORTS cv::Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const cv::Ptr<PointSetRegistrator::Callback>& cb,
		int modelPoints, double threshold,
		double confidence = 0.99, int maxIters = 1000);

	CV_EXPORTS cv::Ptr<PointSetRegistrator> createLMeDSPointSetRegistrator(const cv::Ptr<PointSetRegistrator::Callback>& cb,
		int modelPoints, double confidence = 0.99, int maxIters = 1000);

	CV_EXPORTS cv::Ptr<PointSetRegistrator> createPreemtiveRANSACPointSetRegistrator(const cv::Ptr<PointSetRegistrator::Callback>& cb,
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
}

#endif

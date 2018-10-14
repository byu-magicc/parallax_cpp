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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "calib3d.hpp"
#include "common.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <iostream>

using namespace cv;
using namespace std;

namespace five_point_opencv
{
	class Hypothesis
	{
	public: 
		Mat model;
		float cost;

		Hypothesis(Mat _model, int _cost = 0):
			model(_model), cost(_cost)
		{

		}

		//Compares two hypotheses. Returns true if the first argument is less than (i.e. is ordered before) the second.
		bool operator() (const Hypothesis h1, const Hypothesis h2)
		{
			return h1.cost < h2.cost;
		}
	};

	// JHW The equation can be derived as follows. Let
	// P(I) = P(point j is an inlier)
	// P(M) = P(model i is composed of only inliers)
	// P(A) = P(out of all the models i = 1:n, at least one is correct) = desired confidence
	// m    = model points
	// n    = number of iterations
	// 
	// Applying equations for repeated trials of a binomial random variable,
	// P(M) = P(I)^m
	// P(A) = 1 - (1 - P(M))^n
	// 
	// We can then solve for n:
	// 1 - P(A) = (1 - P(M))^n
	// log(1 - P(A)) = n*log(1 - P(M))
	// n = log(1 - P(A)) / (1 - P(M))
	int RANSACUpdateNumIters(double p, double ep, int modelPoints, int maxIters)
	{
		if (modelPoints <= 0)
			CV_Error(Error::StsOutOfRange, "the number of model points should be positive");

		p = MAX(p, 0.);
		p = MIN(p, 1.);
		ep = MAX(ep, 0.);
		ep = MIN(ep, 1.);

		// avoid inf's & nan's
		double num = MAX(1. - p, DBL_MIN);
		double denom = 1. - std::pow(1. - ep, modelPoints);
		if (denom < DBL_MIN)
			return 0;

		num = std::log(num);
		denom = std::log(denom);

		return denom >= 0 || -num >= maxIters*(-denom) ? maxIters : cvRound(num / denom);
	}


	class RANSACPointSetRegistrator : public PointSetRegistrator
	{
	public:
		FivePointSolver* solver;

		RANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb = Ptr<PointSetRegistrator::Callback>(),
			FivePointSolver* _solver = NULL,
			int _modelPoints = 0, double _threshold = 0, double _confidence = 0.99, int _niters = 100)
			: cb(_cb), modelPoints(_modelPoints), threshold(_threshold), confidence(_confidence), niters(_niters), solver(_solver)
		{
			checkPartialSubsets = false;
		}

		//JHW: Note that the threshold is squared inside the function, so we don't need to square it before passing it into the function.
		int findInliers(const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh) const
		{
			cb->computeError(m1, m2, model, err);
			mask.create(err.size(), CV_8U);

			CV_Assert(err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
			const float* errptr = err.ptr<float>();
			uchar* maskptr = mask.ptr<uchar>();
			float t = (float)(thresh*thresh);
			int i, n = (int)err.total(), nz = 0;
			for (i = 0; i < n; i++)
			{
				int f = errptr[i] <= t;
				maskptr[i] = (uchar)f;
				nz += f;
			}
			return nz;
		}

		bool getSubset(const Mat& m1, const Mat& m2,
			Mat& ms1, Mat& ms2, RNG& rng,
			int maxAttempts = 1000) const
		{
			cv::AutoBuffer<int> _idx(modelPoints);
			int* idx = _idx;
			int i = 0, j, k, iters = 0;
			int esz1 = (int)m1.elemSize(), esz2 = (int)m2.elemSize();
			int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
			int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
			int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
			const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

			ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
			ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

			int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

			CV_Assert(count >= modelPoints && count == count2);
			CV_Assert((esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0);
			esz1 /= sizeof(int);
			esz2 /= sizeof(int);

			for (; iters < maxAttempts; iters++)
			{
				for (i = 0; i < modelPoints && iters < maxAttempts; )
				{
					// JHW - Keep picking new random numbers until you get one that is unique
					int idx_i = 0;
					for (;;)
					{
						idx_i = idx[i] = rng.uniform(0, count);
						for (j = 0; j < i; j++)
							if (idx_i == idx[j])
								break;
						if (j == i)
							break;
					}
					for (k = 0; k < esz1; k++)
						ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
					for (k = 0; k < esz2; k++)
						ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
					if (checkPartialSubsets && !cb->checkSubset(ms1, ms2, i + 1))
					{
						// we may have selected some bad points;
						// so, let's remove some of them randomly
						// JHW - What this really means is only keep the first k points (where k is generated randomly)
						i = rng.uniform(0, i + 1);
						iters++;
						continue;
					}
					i++;
				}
				// JHW - Of course, we always check the subset once complete
				if (!checkPartialSubsets && i == modelPoints && !cb->checkSubset(ms1, ms2, i))
					continue;
				break;
			}

			return i == modelPoints && iters < maxAttempts;
		}

		bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const
		{
			time_cat(common::TimeCatHypoGen);
			bool result = false;
			Mat m1 = _m1.getMat(), m2 = _m2.getMat();
			Mat err, mask, model, bestModel, ms1, ms2;

			int iter; //, niters = MAX(maxIters, 1);
			int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
			int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
			int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

			// write checksum

			// JHW: It looks like they are always initializing the random number generator with the same seed, for repeatability.
			// This means multiple runs with the same data points will give the exact same answer. The only way to get a different
			// answer is to randomize the input points.
			RNG rng((uint64)-1);

			CV_Assert(cb);
			CV_Assert(confidence > 0 && confidence < 1);

			CV_Assert(count >= 0 && count2 == count);
			if (count < modelPoints)
				return false;

			Mat bestMask0, bestMask;

			if (_mask.needed())
			{
				_mask.create(count, 1, CV_8U, -1, true);
				bestMask0 = bestMask = _mask.getMat();
				CV_Assert((bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count);
			}
			else
			{
				bestMask.create(count, 1, CV_8U);
				bestMask0 = bestMask;
			}

			if (count == modelPoints)
			{
				if (cb->runKernel(m1, m2, bestModel) <= 0)
					return false;
				time_cat_verbose(common::TimeCatVerboseNone);
				bestModel.copyTo(_model);
				bestMask.setTo(Scalar::all(1));
				return true;
			}
			
			// JHW: niters is now constant
			// const double outlierRatio = 0.45;
			// niters = RANSACUpdateNumIters(confidence, outlierRatio, modelPoints, maxIters);
			// niters = MAX(niters, 3);
			int totalModels = 0;			

			for (iter = 0; iter < niters; iter++)
			{
				time_cat(common::TimeCatHypoGen);
				int i, goodCount, nmodels;
				if (count > modelPoints)
				{
					bool found = getSubset(m1, m2, ms1, ms2, rng, 10000);
					if (!found)
					{
						if (iter == 0)
							return false;
						break;
					}
				}

				// JHW: It appears that some subsets generate more than one hypothesis. The 5-point algorithm
				// falls in this category. When this occurs, every model that is generated will be
				// individually scored.
				// Futhermore, there is no preemptive RANSAC scoring.
				nmodels = cb->runKernel(ms1, ms2, model);
				time_cat_verbose(common::TimeCatVerboseNone);
				if (nmodels <= 0)
					continue;
				CV_Assert(model.rows % nmodels == 0);
				Size modelSize(model.cols, model.rows / nmodels);
				totalModels += nmodels;

				time_cat(common::TimeCatHypoScoring);
				for (i = 0; i < nmodels; i++)
				{					
					Mat model_i = model.rowRange(i*modelSize.height, (i + 1)*modelSize.height);
					goodCount = findInliers(m1, m2, model_i, err, mask, threshold);

					// JHW - If for some odd reason one or more of the minimum subset points from which the model was
					// generated are not inliers, and not a single other point is an inlier to the model, the model
					// won't even be stored as the best model! This would result in an empty matrix. However,
					// the likelihood of this happening is pretty small, unless the threshold is set to a really low number.
					if (goodCount > MAX(maxGoodCount, modelPoints - 1))
					{
						std::swap(mask, bestMask);
						model_i.copyTo(bestModel);
						maxGoodCount = goodCount;
						// JHW - The interesting thing here is that the number of interations is constantly being updated
						// Fortunately, there is a max number of iterations (default is 1000) that prevents things
						// from going through the roof if the threshold is accidentally set too low.
						// It appears that count is the number of points and goodCount is the number of inliers.
						// This means that the ratio being passed in is the percentage of outliers.
						// Also note that the sequence is monotonicly non-increasing. In other words, an update will never increase the number
						// of iterations required.
						// JHW - Temporarily removing for more accurate timing results
						//niters = RANSACUpdateNumIters(confidence, (double)(count - goodCount) / count, modelPoints, niters);
					}
				}
			}
			//printf("Total models: %d\n", totalModels);

			if (maxGoodCount > 0)
			{
				if (bestMask.data != bestMask0.data)
				{
					if (bestMask.size() == bestMask0.size())
						bestMask.copyTo(bestMask0);
					else
						transpose(bestMask, bestMask0);
				}
				bestModel.copyTo(_model);
				result = true;
			}
			else
				_model.release();

			time_cat(common::TimeCatNone);
			return result;
		}

		void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) { cb = _cb; }

		Ptr<PointSetRegistrator::Callback> cb;
		int modelPoints;
		bool checkPartialSubsets;
		double threshold;
		double confidence;
		int niters;
	};

	class LMeDSPointSetRegistrator : public RANSACPointSetRegistrator
	{
	public:
		FivePointSolver* solver;

		LMeDSPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb = Ptr<PointSetRegistrator::Callback>(),
			FivePointSolver* _solver = NULL,
			int _modelPoints = 0, double _confidence = 0.99, int _niters = 100)
			: RANSACPointSetRegistrator(_cb, _solver, _modelPoints, 0, _confidence, _niters), solver(_solver) {}

		bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const
		{
			time_cat(common::TimeCatHypoGen);
			const double outlierRatio = 0.45;
			bool result = false;
			Mat m1 = _m1.getMat(), m2 = _m2.getMat();
			Mat ms1, ms2, err, errf, model, bestModel, mask, mask0;

			int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
			int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
			int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
			double minMedian = DBL_MAX, sigma;

			RNG rng((uint64)-1);

			CV_Assert(cb);
			CV_Assert(confidence > 0 && confidence < 1);

			CV_Assert(count >= 0 && count2 == count);
			if (count < modelPoints)
				return false;

			if (_mask.needed())
			{
				_mask.create(count, 1, CV_8U, -1, true);
				mask0 = mask = _mask.getMat();
				CV_Assert((mask.cols == 1 || mask.rows == 1) && (int)mask.total() == count);
			}

			if (count == modelPoints)
			{
				if (cb->runKernel(m1, m2, bestModel) <= 0)
					return false;
				time_cat_verbose(common::TimeCatVerboseNone);
				bestModel.copyTo(_model);
				mask.setTo(Scalar::all(1));
				return true;
			}

			// JHW:  niters is now constant
			// JHW: Interesting. So we assume that the outlier ratio is always 45 for LMEDS. The ratio is only
			// updated once.
			int iter; //, niters = RANSACUpdateNumIters(confidence, outlierRatio, modelPoints, maxIters);
			//niters = MAX(niters, 3);
			int totalModels = 0;

			for (iter = 0; iter < niters; iter++)
			{
				time_cat(common::TimeCatHypoGen);
				int i, nmodels;
				if (count > modelPoints)
				{
					bool found = getSubset(m1, m2, ms1, ms2, rng);
					if (!found)
					{
						if (iter == 0)
							return false;
						totalModels += nmodels;
						break;
					}
				}

				nmodels = cb->runKernel(ms1, ms2, model);
				time_cat_verbose(common::TimeCatVerboseNone);
				if (nmodels <= 0)
					continue;

				CV_Assert(model.rows % nmodels == 0);
				Size modelSize(model.cols, model.rows / nmodels);

				time_cat(common::TimeCatHypoScoring);
				for (i = 0; i < nmodels; i++)
				{
					Mat model_i = model.rowRange(i*modelSize.height, (i + 1)*modelSize.height);
					cb->computeError(m1, m2, model_i, err);
					if (err.depth() != CV_32F)
						err.convertTo(errf, CV_32F);
					else
						errf = err;
					CV_Assert(errf.isContinuous() && errf.type() == CV_32F && (int)errf.total() == count);
					// JHW: Why not use std::nth_element instead of std::sort? It's a lot more efficient than sorting the entire array.
					std::sort(errf.ptr<int>(), errf.ptr<int>() + count);

					double median = count % 2 != 0 ?
						errf.at<float>(count / 2) : (errf.at<float>(count / 2 - 1) + errf.at<float>(count / 2))*0.5;

					if (median < minMedian)
					{
						minMedian = median;
						model_i.copyTo(bestModel);
					}
				}
			}

			if (minMedian < DBL_MAX)
			{
				// Interesting. So the threshold is based on 
				sigma = 2.5*1.4826*(1 + 5. / (count - modelPoints))*std::sqrt(minMedian);
				sigma = MAX(sigma, 0.001);

				count = findInliers(m1, m2, bestModel, err, mask, sigma);
				if (_mask.needed() && mask0.data != mask.data)
				{
					if (mask0.size() == mask.size())
						mask.copyTo(mask0);
					else
						transpose(mask, mask0);
				}
				bestModel.copyTo(_model);
				result = count >= modelPoints; //returns false if there aren't enough inliers
			}
			else
				_model.release();

			time_cat(common::TimeCatNone);
			return result;
		}

	};

	class PreemtiveRANSACPointSetRegistrator : public PointSetRegistrator
	{
	public:
		Ptr<PointSetRegistrator::Callback> cb;
		int modelPoints;
		int niters;
		bool checkPartialSubsets;
		double threshold;
		int blocksize;
		FivePointSolver* solver;

		PreemtiveRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb = Ptr<PointSetRegistrator::Callback>(),
			FivePointSolver* _solver = NULL,
			int _modelPoints = 0, double _threshold = 0, int _niters = 100, int _blocksize = 30)
			: cb(_cb), modelPoints(_modelPoints), threshold(_threshold), niters(_niters), blocksize(_blocksize), solver(_solver)
		{
			checkPartialSubsets = false;
		}

		//JHW: Note that the threshold is squared inside the function, so we don't need to square it before passing it into the function.
		int findInliers(const Mat& m1, const Mat& m2, const Mat& model, Mat& err, Mat& mask, double thresh) const
		{
			cb->computeError(m1, m2, model, err);
			mask.create(err.size(), CV_8U);

			CV_Assert(err.isContinuous() && err.type() == CV_32F && mask.isContinuous() && mask.type() == CV_8U);
			const float* errptr = err.ptr<float>();
			uchar* maskptr = mask.ptr<uchar>();
			float t = (float)(thresh*thresh);
			int i, n = (int)err.total(), nz = 0;
			for (i = 0; i < n; i++)
			{
				int f = errptr[i] <= t;
				maskptr[i] = (uchar)f;
				nz += f;
			}
			return nz;
		}

		bool getSubset(const Mat& m1, const Mat& m2, Mat& ms1, Mat& ms2, RNG& rng, int maxAttempts = 1000) const
		{
			cv::AutoBuffer<int> _idx(modelPoints);
			int* idx = _idx;
			int i = 0, j, k, iters = 0;
			int esz1 = (int)m1.elemSize(), esz2 = (int)m2.elemSize();
			int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
			int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
			int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
			const int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();

			ms1.create(modelPoints, 1, CV_MAKETYPE(m1.depth(), d1));
			ms2.create(modelPoints, 1, CV_MAKETYPE(m2.depth(), d2));

			int *ms1ptr = ms1.ptr<int>(), *ms2ptr = ms2.ptr<int>();

			CV_Assert(count >= modelPoints && count == count2);
			CV_Assert((esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0);
			esz1 /= sizeof(int);
			esz2 /= sizeof(int);

			for (; iters < maxAttempts; iters++)
			{
				for (i = 0; i < modelPoints && iters < maxAttempts; )
				{
					// JHW - Keep picking new random numbers until you get one that is unique
					int idx_i = 0;
					for (;;)
					{
						idx_i = idx[i] = rng.uniform(0, count);
						for (j = 0; j < i; j++)
							if (idx_i == idx[j])
								break;
						if (j == i)
							break;
					}
					for (k = 0; k < esz1; k++)
						ms1ptr[i*esz1 + k] = m1ptr[idx_i*esz1 + k];
					for (k = 0; k < esz2; k++)
						ms2ptr[i*esz2 + k] = m2ptr[idx_i*esz2 + k];
					if (checkPartialSubsets && !cb->checkSubset(ms1, ms2, i + 1))
					{
						// we may have selected some bad points;
						// so, let's remove some of them randomly
						// JHW - What this really means is only keep the first k points (where k is generated randomly)
						i = rng.uniform(0, i + 1);
						iters++;
						continue;
					}
					i++;
				}
				// JHW - Of course, we always check the subset once complete
				if (!checkPartialSubsets && i == modelPoints && !cb->checkSubset(ms1, ms2, i))
					continue;
				break;
			}

			return i == modelPoints && iters < maxAttempts;
		}

		bool run(InputArray _m1, InputArray _m2, OutputArray _model, OutputArray _mask) const
		{
			//printf("Threshold: %f\n", threshold);

			bool result = false;
			Mat m1 = _m1.getMat(), m2 = _m2.getMat();
			Mat err, mask, model, bestModel, ms1, ms2;

			int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
			int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
			int count = m1.checkVector(d1), count2 = m2.checkVector(d2), maxGoodCount = 0;

			// Initialize random number generator with the same seed for repeatability.
			RNG rng((uint64)-1);

			CV_Assert(cb);
			//CV_Assert(confidence > 0 && confidence < 1);

			CV_Assert(count >= 0 && count2 == count);
			if (count < modelPoints)
				return false;

			Mat bestMask0, bestMask;

			if (_mask.needed())
			{
				_mask.create(count, 1, CV_8U, -1, true);
				bestMask0 = bestMask = _mask.getMat();
				CV_Assert((bestMask.cols == 1 || bestMask.rows == 1) && (int)bestMask.total() == count);
			}
			else
			{
				bestMask.create(count, 1, CV_8U);
				bestMask0 = bestMask;
			}

			if (count == modelPoints)
			{
				if (cb->runKernel(m1, m2, bestModel) <= 0)
					return false;
				time_cat_verbose(common::TimeCatVerboseNone);
				bestModel.copyTo(_model);
				bestMask.setTo(Scalar::all(1));
				return true;
			}

			// Randomly permute the input points (observations)
			// This prevents any deterministic ordering (ie. from top to bottom) that could ruin the preemption scheme
			shuffleElements(m1, m2, rng);
			
			// Generate all hypotheses
			vector<Hypothesis> hypotheses;
			for (int iters = 0; iters < niters; iters++)
			{
				// Get a subset from m1 and m2 and store in ms1 and ms2
				// return size: [model points x 1]
				bool found = getSubset(m1, m2, ms1, ms2, rng, 10000);
				if (!found)
					return false;

				// Generate a single hypothesis from subset.
				// In case of the 5-point algorithm, it is assumed that the callback function 
				// will choose between the different hypotheses using a sixth point.
				// However, we still allow more than one hypothesis to be returned.
				int nmodels = cb->runKernel(ms1, ms2, model);
				time_cat_verbose(common::TimeCatVerboseNone);
				if (nmodels <= 0)
					continue;
				
				// Copy models to array
				CV_Assert(model.rows % nmodels == 0);
				Size modelSize(model.cols, model.rows / nmodels);
				for (int j = 0; j < nmodels; j++)
				{
					Mat model_j = model.rowRange(j*modelSize.height, (j + 1)*modelSize.height);
					hypotheses.push_back(Hypothesis(model_j));
				}
			}

			// Score hypothesis using breadth-first, preemtive scoring
			int hypothesesRemaining = hypotheses.size();
			int idx = 0;
			for (int b = 0; idx < count && hypothesesRemaining > 1; b++)
			{
				// Score remaining hypotheses on observations [i1, i2).
				int i1 = idx;
				int i2 = min(idx + blocksize, count);
				idx = i2;
				for (int modelIdx = 0; modelIdx < hypothesesRemaining; modelIdx++)
					hypotheses[modelIdx].cost += cb->computeCost(m1, m2, hypotheses[modelIdx].model, i1, i2, threshold);

				// Keep the best half of the remaining hypotheses. Reorder them so that the best (lowest cost) are first.
				int keep = max(hypothesesRemaining / 2, 1);
				std::nth_element(hypotheses.begin(), hypotheses.begin() + keep, hypotheses.begin() + hypothesesRemaining, hypotheses[0]);
				//printf("Block %d: ", b);
				//for (int i = 0; i < hypotheses.size(); i++)
				//{
				//	if (i == 0)
				//		printf("keep: ");
				//	if (i == keep)
				//		printf("discard: ");
				//	if (i == hypothesesRemaining)
				//		printf("moot: ");
				//	printf("%.1f ", hypotheses[i].cost);
				//}
				//printf("\n");
				hypothesesRemaining = keep;
			}

			// Find the best remaining model
			int keep = 1;
			std::nth_element(hypotheses.begin(), hypotheses.begin() + keep, hypotheses.begin() + hypothesesRemaining, hypotheses[0]);
			//printf("End: ");
			//for (int i = 0; i < keep; i++)
			//{
			//	if (i == 0)
			//		printf("keep: ");
			//	if (i == keep)
			//		printf("discard: ");
			//	if (i == hypothesesRemaining)
			//		printf("moot: ");
			//	printf("%.1f ", hypotheses[i].cost);
			//}
			//printf("\n");
			hypothesesRemaining = keep;

			// Create the mask array and copy to output
			bestModel = hypotheses[0].model;
			findInliers(m1, m2, bestModel, err, bestMask, threshold);
			bestModel.copyTo(_model);
			if (bestMask.size() == bestMask0.size())
				bestMask.copyTo(bestMask0);
			else
				transpose(bestMask, bestMask0);
			return true;
		}

		void setCallback(const Ptr<PointSetRegistrator::Callback>& _cb) { cb = _cb; }
	};

	Ptr<PointSetRegistrator> createRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
		FivePointSolver* solver,
		int _modelPoints, double _threshold,
		double _confidence, int _niters)
	{
		return Ptr<PointSetRegistrator>(
			new RANSACPointSetRegistrator(_cb, solver, _modelPoints, _threshold, _confidence, _niters));
	}


	Ptr<PointSetRegistrator> createLMeDSPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
		FivePointSolver* solver,
		int _modelPoints, double _confidence, int _niters)
	{
		return Ptr<PointSetRegistrator>(
			new LMeDSPointSetRegistrator(_cb, solver, _modelPoints, _confidence, _niters));
	}


	Ptr<PointSetRegistrator> createPreemtiveRANSACPointSetRegistrator(const Ptr<PointSetRegistrator::Callback>& _cb,
		FivePointSolver* solver,
		int _modelPoints, double _threshold, int _niters, int _blocksize)
	{
		return Ptr<PointSetRegistrator>(
			new PreemtiveRANSACPointSetRegistrator(_cb, solver, _modelPoints, _threshold, _niters, _blocksize));
	}

	void swapElements(Mat& m1, Mat& m2, int i, int j)
	{
		int k;
		int esz1 = (int)m1.elemSize(), esz2 = (int)m2.elemSize();
		int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
		int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
		int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
		int *m1ptr = m1.ptr<int>(), *m2ptr = m2.ptr<int>();
		CV_Assert(i < count && j < count && count == count2);

		Mat tmp;
		tmp.create(1, 1, CV_MAKETYPE(m1.depth(), d1));
		int *tmpptr = tmp.ptr<int>();

		// We will be copying the data using integer increments, even though the data
		// might not actually be composed of integers.
		CV_Assert((esz1 % sizeof(int)) == 0 && (esz2 % sizeof(int)) == 0);
		esz1 /= sizeof(int);
		esz2 /= sizeof(int);

		for (k = 0; k < esz1; k++)
			tmpptr[k] = m1ptr[i*esz1 + k];
		for (k = 0; k < esz1; k++)
			m1ptr[i*esz1 + k] = m1ptr[j*esz1 + k];
		for (k = 0; k < esz1; k++)
			m1ptr[j*esz1 + k] = tmpptr[k];

		for (k = 0; k < esz2; k++)
			tmpptr[k] = m2ptr[i*esz2 + k];
		for (k = 0; k < esz2; k++)
			m2ptr[i*esz2 + k] = m2ptr[j*esz2 + k];
		for (k = 0; k < esz2; k++)
			m2ptr[j*esz2 + k] = tmpptr[k];
	}

	// Randomly permute the ordering of pairs of points using the Fisher-Yates shuffle algorithm
	// This algorithm is O(n) and for a certain number of elements always takes the same time to execute.
	void shuffleElements(Mat& m1, Mat& m2, RNG& rng)
	{
		int d1 = m1.channels() > 1 ? m1.channels() : m1.cols;
		int d2 = m2.channels() > 1 ? m2.channels() : m2.cols;
		int count = m1.checkVector(d1), count2 = m2.checkVector(d2);
		for (int i = 0; i < count; i++)
		{
			int j = rng.uniform(i, count);
			if (i != j)
				swapElements(m1, m2, i, j);
		}
	}
}


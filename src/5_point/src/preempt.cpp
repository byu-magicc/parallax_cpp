#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 

#include <time.h>
#include <map>

#include "precomp.hpp"
#include "calib3d.hpp"
#include "preempt.hpp"
#include "mex.h"
#include "OpenCV2Matlab.h"

// This implementation is designed to be (mostly) standalone, and utilizes the standard types

using namespace std;
using namespace cv;
using namespace cv_;

// Fisher-Yates shuffle algorithm, constant O(n)
void shuffleElements(vector<Point2d>& pts1, vector<Point2d>& pts2, RNG& rng)
{
	int count = std::min(pts1.size(), pts2.size());
	for (int i = 0; i < count; i++)
	{
		int j = rng.uniform(i, count);
		if (i != j)
		{
			std::swap(pts1[i], pts1[j]);
			std::swap(pts2[i], pts2[j]);
		}
	}
}

void getSubset(vector<Point2d>& pts1, vector<Point2d>& pts2, vector<Point2d>& subset1, vector<Point2d>& subset2, int modelPoints, RNG& rng)
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

class Hypothesis
{
public:
	Mat model;
	float cost;

	Hypothesis(Mat _model, int _cost = 0) :
		model(_model), cost(_cost)
	{

	}

	//Compares two hypotheses. Returns true if the first argument is less than (i.e. is ordered before) the second.
	bool operator() (const Hypothesis h1, const Hypothesis h2)
	{
		return h1.cost < h2.cost;
	}
};

float computeError(Point2d& point1, Point2d& point2, Matx33d& E) 
{
	Vec3d x1(point1.x, point1.y, 1.);
	Vec3d x2(point2.x, point2.y, 1.);
	Vec3d Ex1 = E * x1;
	Vec3d Etx2 = E.t() * x2;
	double x2tEx1 = x2.dot(Ex1);

	double a = Ex1[0] * Ex1[0];
	double b = Ex1[1] * Ex1[1];
	double c = Etx2[0] * Etx2[0];
	double d = Etx2[1] * Etx2[1];

	float err = x2tEx1 * x2tEx1 / (a + b + c + d);
	return err;
}

float computeCost1(vector<Point2d>& pts1, vector<Point2d>& pts2, Mat& E_, int i1, int i2, float threshold)
{
	Matx33d E(E_.ptr<double>());
	double cost = 0;
	for (int i = i1; i < i2; i++)
	{
		Vec3d x1(pts1[i].x, pts1[i].y, 1.);
		Vec3d x2(pts2[i].x, pts2[i].y, 1.);
		Vec3d Ex1 = E * x1;
		Vec3d Etx2 = E.t() * x2;
		double x2tEx1 = x2.dot(Ex1);

		double a = Ex1[0] * Ex1[0];
		double b = Ex1[1] * Ex1[1];
		double c = Etx2[0] * Etx2[0];
		double d = Etx2[1] * Etx2[1];

		float err = x2tEx1 * x2tEx1 / (a + b + c + d);
		cost += std::min(std::fabs(err) / (threshold*threshold), 1.0f);
	}
	return cost;
}

float computeCost2(vector<Point2d>& pts1, vector<Point2d>& pts2, Mat& E_, int i1, int i2, float threshold)
{
	Matx33d E(E_.ptr<double>());
	double cost = 0;
	for (int i = i1; i < i2; i++)
	{
		Vec3d x1(pts1[i].x, pts1[i].y, 1.);
		Vec3d x2(pts2[i].x, pts2[i].y, 1.);
 		Vec3d Ex1 = E * x1;
 		double a = Ex1[0];
 		double b = Ex1[1];
 		double dotprod = x2.dot(Ex1);
 		double errSqr = dotprod*dotprod / (a*a + b*b);
		cost += std::min(errSqr / (threshold*threshold), 1.0);
	}
	return cost;
}

float computeCost3(vector<Point2d>& pts1, vector<Point2d>& pts2, Mat& E_, int i1, int i2, float threshold)
{
	const double* E = E_.ptr<double>();
	double cost = 0;
	for (int i = i1; i < i2; i++)
	{
		double a, b, c, dotprod, normsqr, distsqr;
		
		// Find the equation of the line 
		// l = [a b c]' E*x2
		// ax + by + c = 0
		a = E[0]*pts1[i].x + E[1]*pts1[i].y + E[2];
		b = E[3]*pts1[i].x + E[4]*pts1[i].y + E[5];
		c = E[6]*pts1[i].x + E[7]*pts1[i].y + E[8];
		
		// distance to the line can be found be taking the dot product
		// of the normalized line and the homogeneous coordinate of the point.
		// To normalize the line, divide a, b, and c by sqrt(a^2 + b^2), 
		// so that vector [a_new b_new] is a unit vector.
		// We normalize when the square of the distance is computed, thus avoiding a sqrt.
		dotprod = (pts2[i].x*a + pts2[i].y*b + c);
		distsqr = dotprod*dotprod / (a*a + b*b);
		cost += std::min(distsqr/(threshold*threshold), 1.0);
	}
	return cost;
}

float computeCost(vector<Point2d>& pts1, vector<Point2d>& pts2, Mat& E, int i1, int i2, float threshold, int method)
{
	if(method == 1)
		return computeCost1(pts1, pts2, E, i1, i2, threshold);
	else if(method == 2)
		return computeCost2(pts1, pts2, E, i1, i2, threshold);
	else //if(method == 3)
		return computeCost3(pts1, pts2, E, i1, i2, threshold);
}

int sixpoint(vector<Point2d>& subset1, vector<Point2d>& subset2, Mat& model, bool subtime)
{
	// Use the first 5 points to generate multiple possible hypotheses
	CV_Assert(subset1.size() == 6 && subset2.size() == 6);
	vector<Point2d> subset1_(subset1.begin(), subset1.begin() + 5);
	vector<Point2d> subset2_(subset2.begin(), subset2.begin() + 5);
	int nmodels = cv_::fivepoint(subset1_, subset2_, model, subtime);
	if (nmodels <= 0)
		return nmodels;

	// Put hypotheses into a vector
	CV_Assert(model.rows % nmodels == 0);
	Size modelSize(model.cols, model.rows / nmodels);
	vector<Hypothesis> hypotheses;
	for (int j = 0; j < nmodels; j++)
	{
		Mat model_j = model.rowRange(j*modelSize.height, (j + 1)*modelSize.height);
		hypotheses.push_back(Hypothesis(model_j));
	}

	// Use the sixth point for disambiguation
	Point2d point1 = subset1[5];
	Point2d point2 = subset2[5];
	for (int i = 0; i < nmodels; i++)
	{
		Matx33d E(hypotheses[i].model.ptr<double>());
		hypotheses[i].cost = computeError(point1, point2, E);
	}

	// Find minumum 
	Hypothesis bestHypothesis = *min_element(hypotheses.begin(), hypotheses.end(), hypotheses[0]);
	model = bestHypothesis.model;
	nmodels = 1;
	return nmodels;
}

// For this flexible function, we assume that the points are already in the normalized image plane. No undistorting points necessary!
Mat findEssentialMatPreempt2(vector<Point2f> pts1_, vector<Point2f> pts2_, float threshold, int niters, int blocksize, string method)
{
	// Options
	bool sixthPoint = (method.find("a6") != std::string::npos);
	bool hypoGenOnly = (method.find("h") != std::string::npos);
	bool subtime = (method.find("ss") != std::string::npos);
	int costMethod = 1;
	if(method.find("c1") != std::string::npos)
		costMethod = 1;
	else if(method.find("c2") != std::string::npos)
		costMethod = 2;
	else if(method.find("c3") != std::string::npos)
		costMethod = 3;
	
	// Convert points to double (the fivepoint function expects doubles)
	vector<Point2d> pts1, pts2;
	for (int i = 0; i < pts1_.size(); i++)
	{
		pts1.push_back(pts1_[i]);
		pts2.push_back(pts2_[i]);
	}

	// Randomly permute the input points (observations)
	// This prevents any deterministic ordering (ie. from top to bottom) that could ruin the preemption scheme
	RNG rng(getCPUTickCount());
	if(!hypoGenOnly)
		shuffleElements(pts1, pts2, rng);

	// Generate all hypotheses up front
	if(subtime)
		tic();
	vector<Hypothesis> hypotheses;
	for (int iters = 0; iters < niters; iters++)
	{
		// Pick a random subset of points
		vector<Point2d> subset1;
		vector<Point2d> subset2;
		int modelPoints = sixthPoint ? 6 : 5;
		getSubset(pts1, pts2, subset1, subset2, modelPoints, rng);

		// Generate a single hypothesis from subset.
		Mat model;
		int nmodels;
		if (sixthPoint)
			nmodels = sixpoint(subset1, subset2, model, subtime);
		else
			nmodels = cv_::fivepoint(subset1, subset2, model, subtime);
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
	if(subtime)
	{
		tic(); toc("setup", 0);
		tic(); toc("SVD", 0);
		tic(); toc("coeffs1", 0);
		tic(); toc("coeffs2", 0);
		tic(); toc("coeffs3", 0);
		tic(); toc("solvePoly", 0);
		tic(); toc("constructE", 0);
		toc("TotalHypoGenTime");
	}	
	if(hypoGenOnly)
	{
		mexPrintf("%d hypotheses generated in %d iterations\n", hypotheses.size(), niters);
		return hypotheses[0].model;
	}

	// Score hypotheses using breadth-first, observation-based preemtive scoring
	int hypothesesRemaining = hypotheses.size();
	int idx = 0;
	int count = std::min(pts1.size(), pts2.size());
	for (int block = 0; idx < count && hypothesesRemaining > 1; block++)
	{
		// Score remaining hypotheses on observations [i1, i2).
		int i1 = idx;
		int i2 = min(idx + blocksize, count);
		idx = i2;
		for (int modelIdx = 0; modelIdx < hypothesesRemaining; modelIdx++)
			hypotheses[modelIdx].cost += computeCost(pts1, pts2, hypotheses[modelIdx].model, i1, i2, threshold, costMethod);

		// Keep the best half of the remaining hypotheses. Reorder them so that the best (lowest cost) are first.
		int keep = max(hypothesesRemaining / 2, 1);
		std::nth_element(hypotheses.begin(), hypotheses.begin() + keep, hypotheses.begin() + hypothesesRemaining, hypotheses[0]);
		//mexPrintf("Block %d: ", b);
		//for (int i = 0; i < hypotheses.size(); i++)
		//{
		//	if (i == 0)
		//		mexPrintf("keep: ");
		//	if (i == keep)
		//		mexPrintf("discard: ");
		//	if (i == hypothesesRemaining)
		//		mexPrintf("moot: ");
		//	mexPrintf("%.1f ", hypotheses[i].cost);
		//}
		//mexPrintf("\n");
		hypothesesRemaining = keep;
	}

	// Find and return the best remaining model
	int keep = 1;
	std::nth_element(hypotheses.begin(), hypotheses.begin() + keep, hypotheses.begin() + hypothesesRemaining, hypotheses[0]);
	//mexPrintf("End: ");
	//for (int i = 0; i < keep; i++)
	//{
	//	if (i == 0)
	//		mexPrintf("keep: ");
	//	if (i == keep)
	//		mexPrintf("discard: ");
	//	if (i == hypothesesRemaining)
	//		mexPrintf("moot: ");
	//	mexPrintf("%.1f ", hypotheses[i].cost);
	//}
	//mexPrintf("\n");
	hypothesesRemaining = keep;
	Mat E = hypotheses[0].model;
	return E;
}
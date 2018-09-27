#include "OpenCV2Matlab.h"
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "precomp.hpp"
#include "calib3d.hpp"
#include "preempt.hpp"
#include <vector>
#include <map>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <time.h>
#else
#include <sys/time.h>
#define sprintf_s sprintf
#endif

using namespace std;
using namespace cv;
using namespace cv_;

// Gateway function
void mexFunction2(int numOutputs, mxArray *outputs[], int numInputs, const mxArray *inputs[])
{
	// Check inputs
	mexAssert(numInputs == 2, "2 input required.");
	mexAssert(numOutputs == 2, "2 outputs required.");
	mexAssert(mxIsDouble(inputs[0]) && mxIsDouble(inputs[1]), "First two input types must be double.");

	// Convert, calculate, and convert back.
	vector<Point2f> pts1 = mxArrayToPoint2f(inputs[0]);
	vector<Point2f> pts2 = mxArrayToPoint2f(inputs[1]);
	InputArray i1 = pts1;
	InputArray i2 = pts2;
	Mat m1 = i1.getMat();
	Mat m2 = i2.getMat();
	RNG rng(getCPUTickCount());
	for (int i = 0; i < 1000; i++)
		volatile int j = rng.uniform(1, 1000);
	shuffleElements(m1, m2, rng);
	outputs[0] = point2fToMxArray(pts1);
	outputs[1] = point2fToMxArray(pts2);
}

// Fundamental Matrix / Homography
void mexFunction(int numOutputs, mxArray *outputs[], int numInputs, const mxArray *inputs[])
{
	// Check inputs
	mexAssert(numInputs == 6, "5 inputs required.");
	mexAssert(numOutputs >= 1, "At least 1 output required.");
	mexAssert(mxIsDouble(inputs[0]) && mxIsDouble(inputs[1]), "First two input types must be double.");
	mexAssert(mxIsChar(inputs[2]), "Third input must be a char array.");
	mexAssert(mxIsDouble(inputs[3]) && mxGetM(inputs[3]) == 1 && mxGetN(inputs[3]) == 1, "Fourth input must be a scalar.");
	mexAssert(mxIsDouble(inputs[4]) && mxGetM(inputs[4]) == 1 && mxGetN(inputs[4]) == 1, "Fifth input must be a scalar.");
	mexAssert(mxIsDouble(inputs[5]) && mxGetM(inputs[5]) == 1 && mxGetN(inputs[5]) == 1, "Sixth input must be a scalar.");
	mexAssert(mxGetM(inputs[0]) == 2 && mxGetM(inputs[1]) == 2, "First and second inputs must be 2xN.");
	mexAssert(mxGetN(inputs[0]) == mxGetN(inputs[1]), "First and second inputs must have the same number of points.");

	// Convert, calculate, and convert back.
	vector<Point2f> pts1 = mxArrayToPoint2f(inputs[0]);
	vector<Point2f> pts2 = mxArrayToPoint2f(inputs[1]);
	
	
	string s = mxArrayToCPPString(inputs[2]);
	double threshold = mxArrayToDouble(inputs[3]);
	double n_iters = mxArrayToDouble(inputs[4]);
	double blocksize = mxArrayToDouble(inputs[5]);
	bool timeit = (s.find('t') != std::string::npos);
	int count = timeit ? 100 : 1;
	Mat E;
	vector<uchar> mask;
	if(timeit)
		tic();
	for(int i = 0; i < count; i++)
	{
		if (s.find("p1") != std::string::npos)
			E = findEssentialMatPreempt(pts1, pts2, Mat::eye(3, 3, CV_64F), threshold, n_iters, blocksize, mask);
		else if (s.find("p2") != std::string::npos)
			E = findEssentialMatPreempt2(pts1, pts2, threshold, n_iters, blocksize, s);
		else
		{
			mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Invalid method - method string must contain 'p1' or 'p2'");
			break;
		}
	}
	if(timeit)
		toc(s, count, 3);
	outputs[0] = matToMxArray(E);
	if (numOutputs == 2)
		outputs[1] = boolToMxArray(mask);
}


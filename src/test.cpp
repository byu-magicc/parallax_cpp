#include <eigen3/Eigen/Dense>
#include <iostream>
#include "GN_step.h"
#include <vector>

using namespace std;
using namespace Eigen;
using namespace cv;

int main(int argc, char *argv[])
{
    cout << "Runs" << endl;
    vector<cv::Point2d> pts1;
    vector<cv::Point2d> pts2;

	cv::RNG rng(cv::getCPUTickCount());
    for(int i = 0; i < 5; i++)
    {
        pts1.push_back(Point2d(rng.gaussian(1), rng.gaussian(1)));
        pts2.push_back(Point2d(rng.gaussian(1), rng.gaussian(1)));
    }
    Mat R0 = Mat::eye(3, 3, CV_64F);
    Mat t0 = (Mat_<double>(8, 8) << rng.gaussian(1), rng.gaussian(1), rng.gaussian(1));
    Mat R2, t2;
    Mat E = ptr_opencv::findEssentialMatGN(pts1, pts2, R0, t0, R2, t2, 100, 10, true, false);
    cout << E << endl;
}
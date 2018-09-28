#include <eigen3/Eigen/Dense>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ptr/GN_step.h"
#include "common.h"
#include <vector>

using namespace std;
using namespace Eigen;
using namespace cv;


void timing(int argc, char *argv[])
{

}

int main(int argc, char *argv[])
{
	// Get rid of first arg (executable name)
	argc--; argv++;

	// Make sure there are sufficient arguments
	if(argc < 1)
	{
		cout << "Usage: ./cli pts_in [timing, comp_mpda, or sweep]" << endl;
		return 0;
	}
    string filename = string(argv[0]);
    vector<vector<Point2f> > pts1;
    vector<vector<Point2f> > pts2;
    tic();
    loadPoints(filename, pts1, pts2);
    toc("loadPoints");

    return 0;


    argc--; argv++;

	string s = string(argv[0]);
	if(s == "timing")
		timing(argc - 1, argv + 1);
	// else if(s == "comp_mpda")
	// 	compare_mpda(argc - 1, argv + 1);
	// else if(s == "sweep")
	// 	sweep_sensor_noise(argc - 1, argv + 1);
	else
		cout << "Usage: cli [timing, comp_mpda, or sweep]" << endl;
}
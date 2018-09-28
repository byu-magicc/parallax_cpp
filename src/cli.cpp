#include <eigen3/Eigen/Dense>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ptr/GN_step.h"
#include "common.h"
#include <vector>
#include <fstream>

using namespace std;
using namespace Eigen;
using namespace cv;

void timing(string yaml_filename)
{



}

void accuracy(int argc, char *argv[])
{
    
}

int main(int argc, char *argv[])
{
	// Get rid of first arg (executable name)
	argc--; argv++;

	// Make sure there are sufficient arguments
	if(argc < 2)
	{
		cout << "Usage: ./cli pts_in [timing, comp_mpda, or sweep]" << endl;
		return 0;
	}

	// Read file
	string yaml_filename = string(argv[0]);
	VideoPointData video_data = VideoPointData(yaml_filename);
	cout << "Video filename: " << video_data.video_filename << endl;
	cout << "Points filename: " << video_data.points_filename << endl;
	cout << "Frames: " << (int)video_data.data1.size() << endl;
	return 0;

    // Write file
    double a = 1;
    double b = 2;
    std::ofstream log_file;
    log_file.open("../logs/log_test.bin");
    log_file.write((char*)&a, sizeof(double));
    log_file.write((char*)&b, sizeof(double));
    log_file.close();    

    return 0;





    string s = string(argv[1]);
	if(s == "timing")
		timing(yaml_filename);
	// else if(s == "comp_mpda")
	// 	compare_mpda(argc - 1, argv + 1);
	// else if(s == "sweep")
	// 	sweep_sensor_noise(argc - 1, argv + 1);
	else
		cout << "Usage: cli [timing, comp_mpda, or sweep]" << endl;
}
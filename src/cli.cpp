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

void timing(int argc, char *argv[])
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

    // Write file
    double a = 1;
    double b = 2;
    std::ofstream log_file;
    log_file.open("../logs/log_test.bin");
    log_file.write((char*)&a, sizeof(double));
    log_file.write((char*)&b, sizeof(double));
    log_file.close();    

    return 0;

    // Load yaml
    string yaml_filename = string(argv[0]);
    string video_filename, points_filename, truth_filename;
    get_yaml_node("video_filename", yaml_filename, video_filename);
    get_yaml_node("points_filename", yaml_filename, points_filename);
    get_yaml_node("truth_filename", yaml_filename, truth_filename);
    Vector2d image_size;
    Matrix3d camera_matrix;
    get_yaml_eigen("image_size", yaml_filename, image_size);
    get_yaml_eigen("camera_matrix", yaml_filename, camera_matrix);
    cout << "video_filename: " << video_filename;
    cout << "camera_matrix: " << camera_matrix;
    return 0;

    // Load points
    string filename = string(argv[0]);
    vector<vector<Point2f> > pts1;
    vector<vector<Point2f> > pts2;
    tic();
    loadPoints(filename, pts1, pts2);
    toc("loadPoints");

    string s = string(argv[1]);
	if(s == "timing")
		timing(argc - 1, argv + 1);
	// else if(s == "comp_mpda")
	// 	compare_mpda(argc - 1, argv + 1);
	// else if(s == "sweep")
	// 	sweep_sensor_noise(argc - 1, argv + 1);
	else
		cout << "Usage: cli [timing, comp_mpda, or sweep]" << endl;
}
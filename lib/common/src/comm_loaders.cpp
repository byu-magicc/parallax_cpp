#include "comm_loaders.h"
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;
using namespace Eigen;

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/eigen.hpp"


///////////////
// Tokenizer //
///////////////

common::Tokenizer::Tokenizer()
{

}

common::Tokenizer::Tokenizer(string& str) : data(&str[0]), length(str.length())
{

}

common::Tokenizer::Tokenizer(char* data_, int length_) : data(data_), length(length_)
{

}

// Get next token and shorten string by token
common::Tokenizer common::Tokenizer::nextToken(char delimiter)
{
	int index = 0;
	while (data[index] != delimiter && index < length)
		index++;
	Tokenizer child = Tokenizer(data, index);
	data += (index + 1);
	length -= (index + 1);
	return child;
}

int common::Tokenizer::countTokens(char delimiter)
{
	int count = 1;
	for (int i = 0; i < length; i++)
		if (data[i] == delimiter)
			count++;
	return count;
}

common::Tokenizer common::Tokenizer::nextLine()
{
	Tokenizer line = nextToken('\n');
	// Remove carriage return. For some reason Windows ignores the carriage return,  
	// In linux the carriage return isn't filtered and breaks the toInt function. 
	// It also causes the terminal cursor to return to beginning of line
	// (when printing), thus overwriting text. 
	if(line.countTokens('\r') > 0)
		line = line.nextToken('\r');
	return line;
}

int common::Tokenizer::countLines()
{
	return countTokens('\n');
}

bool common::Tokenizer::hasToken()
{
	return length > 0;
}

string common::Tokenizer::str()
{
	return string(data, length);
}

int common::Tokenizer::toInt()
{
	int num = 0;
	int sign = 1;
	for (int i = 0; i < length; i++)
	{
		char c = data[i];
		if (c == '-')
			sign = -1;
		else if (c >= '0' && c <= '9')
			num = num * 10 + (c - '0');
		else if (c != ' ' && c != '\r' && c != '\n')
		{
			cout << "Warning: unrecognized character: \"" << c << "\"" << "(" << (int)c << ")" << endl;
			assert(0);
		}
	}
	return sign * num;
}

float common::Tokenizer::toFloat()
{
	int num = 0;
	int divisor = 1;
	bool afterDecimalPoint = false;
	int sign = 1;
	for (int i = 0; i < length; i++)
	{
		char c = data[i];
		if (c == '-')
			sign = -1;
		else if (c >= '0' && c <= '9')
		{
			num = num * 10 + (c - '0');
			if(afterDecimalPoint)
				divisor *= 10;
		}
		else if (c == '.')
			afterDecimalPoint = true;
		else if (c != ' ' && c != '\r' && c != '\n')
		{
			cout << "Warning: unrecognized character: \"" << c << "\"" << "(" << (int)c << ")" << endl;
			assert(0);
		}
	}
	float result = (float)sign * num / divisor;
	return result;
}

//atoi is slow in DEBUG mode because it requires copying the entire string
//int Tokenizer::toInt()
//{
//	return atoi(str().c_str());
//}

common::Tokenizer common::Tokenizer::clone()
{
	return Tokenizer(data, length);
}

bool common::fileExists(string name)
{
	ifstream ifile(name);
	return ifile.good();
}

void common::loadPoints(string filename, sequence_t& data1, sequence_t& data2)
{
	if (! fileExists(filename))
	{
		cout << "File " << filename << " does not exist." << endl;
		assert(0);
	}

	// Init tokenizer
	ifstream myStream(filename);
	stringstream sstr;
	sstr << myStream.rdbuf();
	string str = sstr.str();
	Tokenizer file = Tokenizer(str);

	// Read points
	data1 = sequence_t();
	data2 = sequence_t();
	while(file.hasToken())
	{
		Tokenizer line = file.nextLine();
		int frame = line.nextToken(' ').toInt();
		int npts = line.nextToken(' ').toInt();
		data1.push_back(scan_t(npts));
		data2.push_back(scan_t(npts));
		scan_t& pts1 = data1[data1.size() - 1];
		scan_t& pts2 = data2[data2.size() - 1];
		for(int i = 0; i < npts; i++)
		{
			line = file.nextLine();
			pts1[i](0) = line.nextToken(' ').toFloat();
			pts1[i](1) = line.nextToken(' ').toFloat();
			pts2[i](0) = line.nextToken(' ').toFloat();
			pts2[i](1) = line.nextToken(' ').toFloat();
		}
	}
}

void common::undistort_points(const scan_t& pts, scan_t& pts_u, Matrix3d camera_matrix)
{
	// Note: We aren't inverting actually the actual camera matrix. We assume 
	// the camera matrix is formatted as expected:
	// [fx 0  cx
	//  0  fy cy
	//  0  0  1]
	double inv_fx = 1./camera_matrix(0, 0);
	double inv_fy = 1./camera_matrix(1, 1);
	double cx = camera_matrix(0, 2);
	double cy = camera_matrix(1, 2);
	pts_u = scan_t(pts.size());
	for(int i = 0; i < pts.size(); i++)
		pts_u[i] << (pts[i](0) - cx)*inv_fx, (pts[i](1) - cy)*inv_fy;
}

Vector2d common::sampson_err(const Matrix3d& E, const scan_t& pts1, const scan_t& pts2)
{
	int n_pts = pts1.size();
	int medianIdx = (int)n_pts / 2;
	vector<double> err = vector<double>(n_pts, 0);
	double err_total = 0;
	for(int i = 0; i < n_pts; i++)
	{
		Vector3d x1;
		x1 << pts1[i](0), pts1[i](1), 1.;
		Vector3d x2;
		x2 << pts2[i](0), pts2[i](1), 1.;
		Vector3d Ex1 = E * x1;
		Vector3d Etx2 = E.transpose() * x2;
		double x2tEx1 = x2.dot(Ex1);
		
		double a = Ex1[0] * Ex1[0];
		double b = Ex1[1] * Ex1[1];
		double c = Etx2[0] * Etx2[0];
		double d = Etx2[1] * Etx2[1];
		double err_i = x2tEx1 * x2tEx1 / (a + b + c + d);
		err[i] = err_i;
		err_total += err_i;
	}
	std::nth_element(err.begin(), err.begin() + medianIdx, err.end());
	double med_err = err[medianIdx];
	double mean_err = err_total / n_pts;
	Vector2d err_result;
	err_result << med_err, mean_err;
	return err_result;
}

void common::five_point(const scan_t& subset1, const scan_t& subset2, vector<Matrix3d>& hypotheses)
{
	// Convert to Point2f
	int n_pts = subset1.size();
	if(n_pts != 5)
	{
		printf("The five-point algorithm doesn't accept %d points\n", n_pts);
		exit(EXIT_FAILURE);
	}	
	vector<cv::Point2d> subset1_cv = vector<cv::Point2d>(n_pts);
	vector<cv::Point2d> subset2_cv = vector<cv::Point2d>(n_pts);
	for (int i = 0; i < n_pts; i++)
	{
		subset1_cv[i].x = subset1[i](0);
		subset1_cv[i].y = subset1[i](1);
		subset2_cv[i].x = subset2[i](0);
		subset2_cv[i].y = subset2[i](1);
	}

	// Calc (multiple) hypotheses using 5-point algorithm
	cv::Mat E_cv = findEssentialMat(subset1_cv, subset2_cv, cv::Mat::eye(3, 3, CV_64F));
	if(E_cv.rows % 3 != 0 || (E_cv.rows > 0 && E_cv.cols != 3))
	{
		printf("Invalid essential matrix size: [%d x %d]\n", E_cv.rows, E_cv.cols);
		exit(EXIT_FAILURE);
	}
	int n_hypotheses = E_cv.rows / 3;
	hypotheses.resize(n_hypotheses);
	for(int i = 0; i < n_hypotheses; i++)
	{
		Map<Matrix<double, 3, 3, RowMajor>> E_i = Map<Matrix<double, 3, 3, RowMajor>>(&E_cv.at<double>(i * 3, 0));
		hypotheses[i] = E_i;
	}
}

void common::loadRT(string filename, truth_t& data)
{
	if (! fileExists(filename))
	{
		cout << "File " << filename << " does not exist." << endl;
		assert(0);
	}

	// Init tokenizer
	ifstream myStream(filename);
	stringstream sstr;
	sstr << myStream.rdbuf();
	string str = sstr.str();
	Tokenizer file = Tokenizer(str);

	// Read RT
	data = truth_t();
	while(file.hasToken())
	{
		data.push_back(Matrix4d());
		Matrix4d& RT = data[data.size() - 1];
		Tokenizer line = file.nextLine();
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++)
				RT(i, j) = line.nextToken(' ').toFloat();
	}
}

common::VideoPointData::VideoPointData(string yaml_filename)
{
	YAML::Node node = YAML::LoadFile(yaml_filename);

	// Filenames	
	get_yaml_node("video_filename", yaml_filename, node, video_filename);
	get_yaml_node("points_filename", yaml_filename, node, points_filename);
	get_yaml_eigen("image_size", yaml_filename, node, image_size);

	// Camera matrix
	if(node["camera_matrix"])
		get_yaml_eigen("camera_matrix", yaml_filename, node, camera_matrix);
	else
	{
		// 75 is what we have been doing for DJI video
		// 35 for egtest05
		double diag_dist_pixels = sqrt(image_size[0]*image_size[0] + image_size[1]*image_size[1]);
		double fov_angle_degrees = 50;
		if(node["camera_fov_degrees"])
			get_yaml_node("camera_fov_degrees", yaml_filename, node, fov_angle_degrees);
		else
			cout << "No camera parameters found, defaulting to FOV of 50 degrees." << endl;
		double focal_length_pixels = diag_dist_pixels / 2 / tan((fov_angle_degrees * M_PI / 180) / 2);
		camera_matrix <<
			focal_length_pixels, 0, image_size[0] / 2,
			0, focal_length_pixels, image_size[1] / 2,
			0, 0, 1;
	}
	if(node["dist_coeffs"])
		get_yaml_eigen("dist_coeffs", yaml_filename, node, dist_coeffs);
	else
		dist_coeffs << 0, 0, 0, 0, 0;

	// Point data
	loadPoints(points_filename, pts1, pts2);

	// Truth data
	if(node["truth_filename"])
	{
		get_yaml_node("truth_filename", yaml_filename, node, truth_filename);
		loadRT(truth_filename, RT);
	}
	else
	{
		truth_filename = "";
		RT = truth_t();
	}
}
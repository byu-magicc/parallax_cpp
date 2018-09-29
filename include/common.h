#ifndef COMMON_H
#define COMMON_H

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include <deque>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>

struct AverageTime
{
	double totalCpuTime = 0;
	double totalActualTime = 0;
	int count = 0;
};

struct timeMeasurement
{
	double cpuTime;
	double actualTime;
	double cpuAv;
	double actualAv;
};

// Convenient string format function.
// See https://stackoverflow.com/questions/69738/c-how-to-get-fprintf-results-as-a-stdstring-w-o-sprintf/69911#69911
std::string str_format (const char *fmt, ...);

double get_wall_time();

double get_cpu_time();

void tic();

timeMeasurement toc(std::string s = "", int count = 1, int sigFigs = 2, bool print = true);

timeMeasurement toc_peek();

void resetTimeAverages();

// Note: A Tokenizer class/struct might seem like reinventing the wheel.
// It also may seem like it isn't a good idea to use pointers to strings
// when the underlying string could be deconstructed without warning.
// But the main purpose of this class is to optimize code in DEBUG mode. 
// In DEBUG mode it can decrease the time to read large files like
// campus.txt from 10s to 2s.
struct Tokenizer
{
	Tokenizer();
	Tokenizer(std::string& str);
	Tokenizer(char* data_, int length_);

	// Get next token and shorten string by token
	Tokenizer nextToken(char delimiter);
	int countTokens(char delimiter);
	Tokenizer nextLine();
	int countLines();
	bool hasToken();
	std::string str();
	int toInt();
	float toFloat();
	Tokenizer clone();

	char* data;
	int length;
};

bool fileExists(std::string name);

void printMat(const char* s, cv::Mat p, int sigFigAfterDecimal = 3, int sigFigBeforeDecimal = 2);

void printMatToStream(std::iostream& ss, std::string s, cv::Mat p, int sigFigAfterDecimal = 3, int sigFigBeforeDecimal = 2);

void loadPoints(std::string filename, std::vector<std::vector<cv::Point2f>>& pts1, std::vector<std::vector<cv::Point2f>>& pts2);

void loadRT(std::string filename, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> >& RT);

// Loads scalar parameters from a .yaml file
// Author: James Jackson
// JHW: Note that it is easiest to just put the definition for the template function in the header, otherwise
// you usually have to manually specify each template you want to use.
// See https://stackoverflow.com/questions/115703/storing-c-template-function-definitions-in-a-cpp-file/115735#115735
template <typename T>
bool get_yaml_node(const std::string key, const std::string filename, T& val, bool print_error = true)
{
	YAML::Node node = YAML::LoadFile(filename);
	if (node[key])
	{
		val = node[key].as<T>();
		return true;
	}
	else
	{
		if (print_error)
		{
			printf("Unable to load \"%s\" from %s\n", key.c_str(), filename.c_str());
		}
		return false;
	}
}

// Loads array from a .yaml file into an Eigen-type matrix or vector.
// Author: James Jackson
template <typename T>
bool get_yaml_eigen(const std::string key, const std::string filename, Eigen::MatrixBase<T>& val) 
{
	YAML::Node node = YAML::LoadFile(filename);
	std::vector<double> vec;
	if (node[key])
	{
		vec = node[key].as<std::vector<double>>();
		if (vec.size() == (val.rows() * val.cols()))
		{
			int k = 0;
			for (int i = 0; i < val.rows(); i++)
			{
				for (int j = 0; j < val.cols(); j++)
				{
					val(i,j) = vec[k++];
				}
			}
			return true;
		}
		else
		{
			printf("Eigen Matrix Size does not match parameter size for \"%s\" in %s", key.c_str(), filename.c_str());
			return false;
		}
	}
	else
	{
		printf("Unable to load \"%s\" from %s\n", key.c_str(), filename.c_str());
		return false;
	}
}

template <typename T>
bool get_yaml_node(const std::string key, std::string filename, YAML::Node node, T& val, bool print_error = true)
{
	if (node[key])
	{
		val = node[key].as<T>();
		return true;
	}
	else
	{
		if (print_error)
		{
			printf("Unable to load \"%s\" from %s\n", key.c_str(), filename.c_str());
		}
		return false;
	}
}

// Loads array from a .yaml file into an Eigen-type matrix or vector.
// Author: James Jackson
template <typename T>
bool get_yaml_eigen(const std::string key, std::string filename, YAML::Node node, Eigen::MatrixBase<T>& val)
{
	std::vector<double> vec;
	if (node[key])
	{
		vec = node[key].as<std::vector<double>>();
		if (vec.size() == (val.rows() * val.cols()))
		{
			int k = 0;
			for (int i = 0; i < val.rows(); i++)
			{
				for (int j = 0; j < val.cols(); j++)
				{
					val(i,j) = vec[k++];
				}
			}
			return true;
		}
		else
		{
			printf("Eigen Matrix Size does not match parameter size for \"%s\" in %s", key.c_str(), filename.c_str());
			return false;
		}
	}
	else
	{
		printf("Unable to load \"%s\" from %s\n", key.c_str(), filename.c_str());
		return false;
	}
}

class VideoPointData
{
public:
    VideoPointData(std::string yaml_filename);
    std::string video_filename, points_filename, truth_filename;
    Eigen::Vector2d image_size;
    Eigen::Matrix3d camera_matrix;
    Eigen::Matrix<double, 5, 1> dist_coeffs;
    std::vector<std::vector<cv::Point2f> > pts1;
    std::vector<std::vector<cv::Point2f> > pts2;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > RT;
};

#endif //COMMON_H
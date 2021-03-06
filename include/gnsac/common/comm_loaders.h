#ifndef COMMON_LOADERS_H
#define COMMON_LOADERS_H

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>
#include "common/comm_math.h"
#include <iostream>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/eigen.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

namespace common
{

// Note: A Tokenizer class/struct might seem like reinventing the wheel,
// and the char pointers could have problems if the string is deconstructed.
// But it is really fast, even in DEBUG mode!
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

typedef std::vector<scan_t> sequence_t;
typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > truth_t;

bool fileExists(std::string name);

void loadPoints(std::string filename, sequence_t& pts1, sequence_t& pts2);

void loadRT(std::string filename, truth_t& RT);

void load_E(std::string filename, std::vector<std::vector<scan_t>>& subsets, std::vector<std::vector<std::vector<Eigen::Matrix3d>>>& data, int frames, int n_hypotheses);

// Loads scalar parameters from a .yaml file
// Author: James Jackson
// JHW: Note that it is easiest to just put the definition for the template function in the header, otherwise
// you usually have to manually specify each template you want to use.
// See https://stackoverflow.com/questions/115703/storing-c-template-function-definitions-in-a-cpp-file/115735#115735
template <typename T>
bool get_yaml_node1(const std::string key, const std::string filename, T& val, bool print_error = true)
{
	YAML::Node node = YAML::LoadFile(filename);
	if (node[key])
	{
		val = node[key].as<T>();
		return true;
	}
	else
	{
		printf("Unable to load \"%s\" from %s\n", key.c_str(), filename.c_str());
		exit(EXIT_FAILURE);
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
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		printf("Unable to load \"%s\" from %s\n", key.c_str(), filename.c_str());
		exit(EXIT_FAILURE);
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
		printf("Unable to load \"%s\" from %s\n", key.c_str(), filename.c_str());
		exit(EXIT_FAILURE);
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
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		printf("Unable to load \"%s\" from %s\n", key.c_str(), filename.c_str());
		exit(EXIT_FAILURE);
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
	sequence_t pts1;
	sequence_t pts2;
	truth_t RT;
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif //COMMON_LOADERS_H
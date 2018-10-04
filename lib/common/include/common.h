#ifndef COMMON_H
#define COMMON_H

#include <deque>
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>

namespace common
{

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

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > scan_t;
typedef std::vector<scan_t> sequence_t;
typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d> > truth_t;

// Convenient string format function.
// See https://stackoverflow.com/questions/69738/c-how-to-get-fprintf-results-as-a-stdstring-w-o-sprintf/69911#69911
std::string str_format (const char *fmt, ...);

double get_wall_time();

double get_cpu_time();

void tic();

timeMeasurement toc(std::string s = "", int count = 1, int sigFigs = 2, bool print = true);

timeMeasurement toc_peek();

void resetTimeAverages();

void print_duration(double duration);

std::string repeat_str(std::string s, int reps);

void progress(int iter, int max_iters);

enum TimeCategory { TimeCatNone = -1, TimeCatHypoGen, TimeCatHypoScoring, TIME_CATS_COUNT };

void cat_timer_reset();

// Fast category timer, overhead of aprox 0.1us
void time_cat(TimeCategory timeCat);

void cat_timer_print();

double* get_cat_times();

#ifdef TIME_VERBOSE
#define time_cat_verbose(cat) time_cat(cat)
#else
#define time_cat_verbose(cat)
#endif


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

bool fileExists(std::string name);

void loadPoints(std::string filename, sequence_t& pts1, sequence_t& pts2);

void loadRT(std::string filename, truth_t& RT);

template <typename T>
void print_eig(const char* s, Eigen::MatrixBase<T>& p, int sfAfter = 3, int sfBefore = 2)
{
	printf("%s: (%dx%d)\n", s, (int)p.rows(), (int)p.cols());
	char format[40];
	//if (p.type() == CV_8U)
	//	sprintf(format, "%%%dd", sfBefore);
	//else
	    sprintf(format, "%%%d.%df", sfBefore + sfAfter, sfAfter);
	for (int r = 0; r < p.rows(); r++)
	{
		if (r == 0)
			printf("[");
		else
			printf("\n ");
		for (int c = 0; c < p.cols(); c++)
		{
			if (c > 0)
				printf(" ");
            printf(format, p(r, c));
		}
	}
	printf("]\n");
}

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
    sequence_t pts1;
    sequence_t pts2;
    truth_t RT;
};

#define unit(vec)  ((vec) / (vec).norm())

// template <typename T>
// Eigen::MatrixBase<T> unit(Eigen::MatrixBase<T>& val)
// {
//     return val / val.norm();
// }

Eigen::Matrix3d skew(Eigen::Vector3d v);

Eigen::Vector3d vex(Eigen::Matrix3d Tx);

double sinc(double x);

Eigen::Matrix3d vecToR(Eigen::Vector3d v);

Eigen::Vector3d RtoVec(Eigen::Matrix3d R);

void undistort_points(const scan_t& pts, scan_t& pts1_u, Eigen::Matrix3d camera_matrix);

Eigen::Vector3d err_truth(const Eigen::Matrix3d& R_est, const Eigen::Vector3d& t_est, const Eigen::Matrix4d& RT);

// Macro to call a determinism checker object with the file, line, and function information
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define write_check(checker, data, size) checker.write(data, size, __FILE__, __LINE__, __func__, "write_check(" STR(checker) ", " STR(data) ", " STR(size) ")")

#define write_check_val(checker, val_) { auto val = (val_); checker.write((char*)&val, sizeof(val), __FILE__, __LINE__, __func__, "write_check_val(" STR(checker) ", " STR(val_) ")"); }

#define DETERMINISM_CHECKER_BUFFER_SIZE 10000

class DeterminismChecker
{
public:
	DeterminismChecker(std::string name, int trial);

	void write(const char* data, std::size_t size, const char* file, int line, const char* func, std::string message = "");

private:
	std::string format_name_trial(std::string name, int trial);

	char in_buffer[DETERMINISM_CHECKER_BUFFER_SIZE];
	bool check;
	std::ofstream out_file;
	std::ifstream in_file;
};

}

#endif //COMMON_H
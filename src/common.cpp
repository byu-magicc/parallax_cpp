#include <random>
#include <chrono>
#include <cmath>
#include "common.h"
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/eigen.hpp"
#include "ptr/GN_step.h"
#include "opencv2/calib3d.hpp"

#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <time.h>
#else
#include <sys/time.h>
#endif

#include <map>
#include <iomanip>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdarg>

using namespace cv;
using namespace std;
using namespace Eigen;

// Convenient string format function.
// See https://stackoverflow.com/questions/69738/c-how-to-get-fprintf-results-as-a-stdstring-w-o-sprintf/69911#69911
std::string vformat (const char *fmt, va_list ap)
{
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.  Be prepared to allocate dynamically if it doesn't fit.
    size_t size = 1024;
    char stackbuf[1024];
    std::vector<char> dynamicbuf;
    char *buf = &stackbuf[0];
    va_list ap_copy;

    while (1) {
        // Try to vsnprintf into our buffer.
        va_copy(ap_copy, ap);
        int needed = vsnprintf (buf, size, fmt, ap);
        va_end(ap_copy);

        // NB. C99 (which modern Linux and OS X follow) says vsnprintf
        // failure returns the length it would have needed.  But older
        // glibc and current Windows return -1 for failure, i.e., not
        // telling us how much was needed.

        if (needed <= (int)size && needed >= 0) {
            // It fit fine so we're done.
            return std::string (buf, (size_t) needed);
        }

        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So try again using a dynamic buffer.  This
        // doesn't happen very often if we chose our initial size well.
        size = (needed > 0) ? (needed+1) : (size*2);
        dynamicbuf.resize (size);
        buf = &dynamicbuf[0];
    }
}

std::string str_format (const char *fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat (fmt, ap);
    va_end (ap);
    return buf;
}

vector<double> cpuTimes;
vector<double> actualTimes;
vector<int> numChildren;

std::map<std::string, AverageTime> averageTimes;

// Source: stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows
double get_wall_time()
{
#ifdef _WIN32
	return (double)clock() / CLOCKS_PER_SEC;
#else
	struct timeval time;
	gettimeofday(&time, NULL);
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
}

double get_cpu_time()
{
#ifdef _WIN32
	FILETIME a, b, c, d;
	GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d);
	return 	(double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
#else
	return (double)clock() / CLOCKS_PER_SEC;
#endif
}

void tic()
{
	cpuTimes.push_back(get_cpu_time());
	actualTimes.push_back(get_wall_time());
	numChildren.push_back(0);
}

string formatTime(double timeVal_ms, int sigFigs)
{
	string units = "ms";
	if(timeVal_ms >= 1e3)
	{
		timeVal_ms /= 1e3;
		units = "s";
	}
	else if(timeVal_ms <= 1)
	{	
		timeVal_ms *= 1e3;
		units = "us";
	}
	int precision = max(sigFigs - 1 - int(floor(log10(timeVal_ms))), 0);
	char buffer[50];
	sprintf(buffer, "%.*f%s", precision, timeVal_ms, units.c_str());
	return string(buffer);
}

string formatTime2(double cpuTime_ms, double actualTime_ms, int sigFigs)
{
	char buffer[50];
	sprintf(buffer, "%s/%s", formatTime(cpuTime_ms, sigFigs).c_str(), formatTime(actualTime_ms, sigFigs).c_str());
	return string(buffer);
}

// Single ─│┌┐└┘├┤┬┴┼
// Mixed  ╒╓╕╖╘╙╛╜╞╟╡╢╤╥╧╨╪╫
// Double ═║╚╝╔╗╠╣╦╩╬

//timeMeasurement toc(string s = "", int count = 1, int sigFigs = 2, bool print = true)
timeMeasurement toc(string s, int count, int sigFigs, bool print)
{
	if (cpuTimes.size() > 0)
	{
		double total_cpuTime_ms = (get_cpu_time() - cpuTimes.back()) * 1e3;
		double total_actualTime_ms = (get_wall_time() - actualTimes.back()) * 1e3;
		double cpuTime_ms = total_cpuTime_ms / count;
		double actualTime_ms = total_actualTime_ms / count;
		cpuTimes.pop_back();
		actualTimes.pop_back();
		numChildren.pop_back();
		averageTimes[s].totalCpuTime += total_cpuTime_ms;
		averageTimes[s].totalActualTime += total_actualTime_ms;
		averageTimes[s].count += count;
		double cpuAv = averageTimes[s].totalCpuTime / averageTimes[s].count;
		double actualAv = averageTimes[s].totalActualTime / averageTimes[s].count;
		string prefix = "";
		for(int i = 0; i < numChildren.size(); i++)
		{
			if(i < ((int)numChildren.size()) - 1)
			{
				if(numChildren[i] > 0)
					prefix += "│";
				else
					prefix += " ";
			}
			else
			{
				if(numChildren[i] > 0)
					prefix += "├";
				else
					prefix += "┌";
			}
		}
		if(numChildren.size() > 0)
			numChildren.back()++;
		if (print)
		{
			string strCurr = (count > 0) ? formatTime2(cpuTime_ms, actualTime_ms, sigFigs) : "";
			string strAv   = (averageTimes[s].count > 5) ? " av: " + formatTime2(cpuAv, cpuAv, sigFigs) : "";
			printf("%s%-*s %-12s%s\n", prefix.c_str(), 25 - (int)numChildren.size(), (s + ":").c_str(), strCurr.c_str(), strAv.c_str());
		}
		return {cpuTime_ms, actualTime_ms, cpuAv, actualAv};
	}
	else
	{
		printf("Error: Must call tic before toc.\n");
		return {0, 0, 0, 0};
	}
}

timeMeasurement toc_peek()
{
	if (cpuTimes.size() > 0)
	{
		double cpuTime_ms = (get_cpu_time() - cpuTimes.back()) * 1e3;
		double actualTime_ms = (get_wall_time() - actualTimes.back()) * 1e3;
		return {cpuTime_ms, actualTime_ms, 0, 0};
	}
	else
	{
		printf("Error: Must call tic before toc.\n");
		return {0, 0, 0, 0};
	}
}

void resetTimeAverages()
{
	averageTimes = std::map<std::string, AverageTime>();
}

Tokenizer::Tokenizer()
{

}

Tokenizer::Tokenizer(string& str) : data(&str[0]), length(str.length())
{

}

Tokenizer::Tokenizer(char* data_, int length_) : data(data_), length(length_)
{

}

// Get next token and shorten string by token
Tokenizer Tokenizer::nextToken(char delimiter)
{
	int index = 0;
	while (data[index] != delimiter && index < length)
		index++;
	Tokenizer child = Tokenizer(data, index);
	data += (index + 1);
	length -= (index + 1);
	return child;
}

int Tokenizer::countTokens(char delimiter)
{
	int count = 1;
	for (int i = 0; i < length; i++)
		if (data[i] == delimiter)
			count++;
	return count;
}

Tokenizer Tokenizer::nextLine()
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

int Tokenizer::countLines()
{
	return countTokens('\n');
}

bool Tokenizer::hasToken()
{
	return length > 0;
}

string Tokenizer::str()
{
	return string(data, length);
}

int Tokenizer::toInt()
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
			CV_Assert(false);
		}
	}
	return sign * num;
}

float Tokenizer::toFloat()
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
			CV_Assert(false);
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

Tokenizer Tokenizer::clone()
{
	return Tokenizer(data, length);
}

bool fileExists(string name)
{
	ifstream ifile(name);
	return ifile.good();
}

void loadPoints(string filename, sequence_t& data1, sequence_t& data2)
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

void loadRT(string filename, truth_t& data)
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

VideoPointData::VideoPointData(string yaml_filename)
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

Matrix3d skew(Vector3d v)
{
	Matrix3d Tx;
	Tx << 0,    -v(2),   v(1),
	      v(2),  0,     -v(0),
	     -v(1),  v(0),   0;
	return Tx;
}

Matrix3d vecToR(Vector3d v)
{
	Matrix3d R;
	Mat v_mat;
	Mat R_mat;
	eigen2cv(v, v_mat);
	Rodrigues(v_mat, R_mat);
	cv2eigen(R_mat, R);
	return R;
}

Vector3d RtoVec(Matrix3d R)
{
	Vector3d v;
	Mat R_mat;
	Mat v_mat;
	eigen2cv(R, R_mat);
	Rodrigues(R_mat, v_mat);
	cv2eigen(v_mat, v);
	return v;
}

void undistort_points(const scan_t& pts, scan_t& pts_u, Matrix3d camera_matrix)
{
	// Note: We aren't inverting the actual camera matrix. We assume camera matrix is formatted
	// as expected:
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

Vector3d err_truth(const Matrix3d& R_est, const Vector3d& t_est, const Matrix4d& RT)
{
	// Extract R and T
	Matrix3d R = RT.block<3, 3>(0, 0);
	Vector3d t = RT.block<3, 1>(0, 3);
	t = unit(t);

	// E error
	// Note: To avoid scaling, we are just re-creating E_est. However, we could always use the L2 norm or something.
	Matrix3d E = skew(t)*R;
	Matrix3d E_est = skew(t_est)*R_est;
	double err_E = min((E - E_est).norm(), (E - (-E_est)).norm());

	// R and t error
	double err_t_angle = acos(unit(t_est).dot(t));
	double err_R_angle = RtoVec(R * R_est.transpose()).norm();
	Vector3d err;
	err << err_E, err_R_angle, err_t_angle;
	return err;
}
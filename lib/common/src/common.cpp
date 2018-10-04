#include <random>
#include <chrono>
#include <cmath>
#include "common.h"
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <iostream>

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

using namespace std;
using namespace Eigen;

///////////////////
// String Format //
///////////////////

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

std::string common::str_format (const char *fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat (fmt, ap);
    va_end (ap);
    return buf;
}

/////////////////
// Tic and Toc //
/////////////////

vector<double> cpuTimes;
vector<double> actualTimes;
vector<int> numChildren;

std::map<std::string, common::AverageTime> averageTimes;

// Source: stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows
double common::get_wall_time()
{
#ifdef _WIN32
	return (double)clock() / CLOCKS_PER_SEC;
#else
	struct timeval time;
	gettimeofday(&time, NULL);
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
#endif
}

double common::get_cpu_time()
{
#ifdef _WIN32
	FILETIME a, b, c, d;
	GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d);
	return 	(double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
#else
	return (double)clock() / CLOCKS_PER_SEC;
#endif
}

void common::tic()
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
common::timeMeasurement common::toc(string s, int count, int sigFigs, bool print)
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

common::timeMeasurement common::toc_peek()
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

void common::resetTimeAverages()
{
	averageTimes = std::map<std::string, AverageTime>();
}

/////////////////
// Progres Bar //
/////////////////

double start_time;

void common::print_duration(double duration)
{
	int total_sec = floor(duration);
	double frac_sec = duration - total_sec;
	int sec = total_sec % 60;
	int total_min = total_sec / 60;
	int min = total_min % 60;
	int total_hour = total_min / 60;
	if(total_sec < 10)
		printf("00:%05.2f", sec + frac_sec);
	else if(total_min < 60)
		printf("%02d:%02d", min, sec);
	else
		printf("%02d:%02d:%02d", total_hour, min, sec);
}

string common::repeat_str(string s, int reps)
{
	string str = string(s.length() * reps, ' ');
	for(int i = 0; i < reps; i++)
		for(int j = 0; j < s.length(); j++)
			str[i * s.length() + j] = s[j];
	return str;
}

void common::progress(int iter, int max_iters)
{
	const int progress_bar_size = 100;
	double ratio = ((double)iter) / max_iters;
	int prcnt = round(ratio*100);
	int bars = round(ratio*progress_bar_size);
	string progress_bar = repeat_str("█", bars) + repeat_str(" ", progress_bar_size - bars);
	printf("\r%3d%%|%s| %d/%d | ", prcnt, progress_bar.c_str(), iter, max_iters);

	double curr_time = get_wall_time();
	if(iter == 0)
		start_time = curr_time;
	else
	{
		double approx_time_left = (max_iters - iter) * (curr_time - start_time) / iter;
		print_duration(approx_time_left);
	}
	if(iter == max_iters)
		printf("\n");
	fflush(stdout);
}

////////////////////
// Category Timer //
////////////////////

double catTimes[common::TIME_CATS_COUNT] = {}; // Init to zero
double catTimerStartTime;
common::TimeCategory currentTimeCat = common::TimeCatNone;

void common::cat_timer_reset()
{
	for(int i = 0; i < common::TIME_CATS_COUNT; i++)
		catTimes[i] = 0;
	currentTimeCat = TimeCatNone;
}

// Fast category timer, overhead of aprox 0.1us
void common::time_cat(common::TimeCategory timeCat)
{
	// Get elapsed time
	double currTime = get_wall_time();
	double elapsedTime = currTime - catTimerStartTime;
	
	// Add timer value to 
	if(currentTimeCat != TimeCatNone)
		catTimes[(int)currentTimeCat] += elapsedTime;

	// Start next timer
	catTimerStartTime = currTime;
	currentTimeCat = timeCat;
}

void common::cat_timer_print()
{
	for(int i = 0; i < TIME_CATS_COUNT; i++)
		printf("Cat %d, time %f\n", i, catTimes[i]);
}

double* common::get_cat_times()
{
	return catTimes;
}

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

Matrix3d common::skew(Vector3d v)
{
	Matrix3d Tx;
	Tx << 0,    -v(2),   v(1),
	      v(2),  0,     -v(0),
	     -v(1),  v(0),   0;
	return Tx;
}

Vector3d common::vex(Matrix3d Tx)
{
	Vector3d w;
	w << Tx(2, 1), Tx(0, 2), Tx(1, 0);
	return w;
}

double common::sinc(double x)
{
    // Taylor series expansion of sin is:           x - x^3/3! + x^5/5! - ...
    // Thus the Taylor series expansion of sinc is: 1 - x^2/3! + x^4/5! - ...
    // Double precision is approximately 16 digits. Since the largest term is x^2,
    // we will experience numerical underflow if |x| < 1e-8.
    // Of course, floating point arithmetic can handle much smaller values than this (as small as 1e-308).
    // I haven't seen any problems with small numbers so far, so we could just check for division by zero,
    // but this solution is guarenteed not to have problems.
    if (fabs(x) < 1e-8)
        return 1;
    else
        return sin(x) / x;
}

Matrix3d common::vecToR(Vector3d w)
{
	double theta = w.norm();
	Matrix3d wx = skew(w);

	// R = eye(3) + sinc(theta)*wx + 0.5*sinc(theta/2)^2*wx^2;	
	double sinc2 = sinc(theta / 2);
	Matrix3d R = Matrix3d::Identity() + sinc(theta)*wx + 0.5 * sinc2 * sinc2 * wx * wx;
	return R;
}

Vector3d common::RtoVec(Matrix3d R)
{
	// The rodrigues formula gives us
	// R = I + sin(theta)*wx_hat + (1 - cos(theta))*wx_hat^2
	// Notice that the first and third terms are symmetric,
	// while the second term is skew-symmetric and has no diagonal components.
	// The diagonal components of the matrix are are an easy way to get the "I + (1 - cos)" terms
	// We can cancel out symmetric terms using (R - R') / 2. This allows us to get the "sin" term.
	double cos_theta = (R.trace() - 1) / 2;
	Vector3d sin_theta_w_hat = vex(R - R.transpose()) / 2;
	double sin_theta = sin_theta_w_hat.norm();

	// To get theta, we could use atan2, but it is slightly more accurate to 
	// use acos or asin, depending on which area of the unit circle we are in.
	// For best accuracy, we should avoid using acos or asin if the value is near 1.
	// An easy way to prevent this is to simply use whichever value is smallest.
	// (the multiplication by 2 slightly alters the asin/acos regions and was determined hueristically)
	// We need to handle theta = [0, pi]. (quadrant 1 and 2)
	// Theta by definition is norm(w), so it can't be negative. Angles larger than pi
	// end up wrapping around the other way, whichever is shortest.
	// theta = atan2(sin_theta, cos_theta);
	double theta;
	if (fabs(cos_theta) < fabs(sin_theta)*2)
		theta = acos(cos_theta);
	else if (cos_theta > 0)
		theta = asin(sin_theta);
	else //if (cos_theta < 0)
		theta = M_PI - asin(sin_theta);
	Vector3d w = sin_theta_w_hat / sinc(theta);
	return w;
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

Vector3d common::err_truth(const Matrix3d& R_est, const Vector3d& t_est, const Matrix4d& RT)
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

common::DeterminismChecker::DeterminismChecker(string name, int trial) : check(false)
{
	string out_filename = format_name_trial(name, trial);
	out_file.open(out_filename);
	if (trial > 0)
	{
		string in_filename = format_name_trial(name, trial - 1);
		in_file.open(in_filename);
		check = true;
	}
}

void common::DeterminismChecker::write(const char* data, std::size_t size, const char* file, int line, const char* func, string message)
{
	out_file.write(data, size);		
	if(check)
	{
		if(size >= DETERMINISM_CHECKER_BUFFER_SIZE)
		{
			cout << "DeterminismChecker buffer to small (" << size << " > " << DETERMINISM_CHECKER_BUFFER_SIZE << ")" << endl;
			exit(EXIT_FAILURE);
		}
		in_file.read(in_buffer, size);
		if(memcmp(data, in_buffer, size) != 0)
		{
			cout << "Determinism check failed in " << func << ", file " << file << ", line " << line << endl;
			cout << message << endl;
			exit(EXIT_FAILURE);
		}
	}
}

string common::DeterminismChecker::format_name_trial(string name, int trial)
{
	return "../logs/" + name + str_format("%d", trial) + ".bin";
}
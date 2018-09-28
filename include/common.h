#ifndef COMMON_H
#define COMMON_H

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include <deque>

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

#endif //COMMON_H
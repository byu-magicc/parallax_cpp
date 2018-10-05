#ifndef COMMON_H
#define COMMON_H

#include <fstream>
#include <eigen3/Eigen/Eigen>
#include "comm_math.h"
#include "comm_loaders.h"

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
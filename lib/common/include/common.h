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

std::string replace(std::string str, const std::string from, const std::string to);

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
enum TimeCategoryVerbose { TimeCatVerboseNone = -1, TimeCatVerboseSetup, TimeCatVerboseSVD,
	TimeCatVerbosePolyGetCoeffs1, TimeCatVerbosePolyGetCoeffs2, TimeCatVerbosePolyGetCoeffs3,
	TimeCatVerboseSolvePoly, TimeCatVerboseConstructE, 
	TimeCatVerboseMakeJ, TimeCatVerboseSolveMatrix, TimeCatVerboseManifoldUpdate,
	TimeCatVerboseCalcResidual, TIME_CATS_VERBOSE_COUNT };

void cat_timer_reset();

// Fast category timer, overhead of aprox 0.1us
void time_cat_fcn(TimeCategory timeCat);

void time_cat_verbose_fcn(TimeCategoryVerbose timeCatVerbose);

void cat_timer_print();

double* get_cat_times();

double* get_cat_times_verbose();

#define time_cat(cat) time_cat_fcn(cat)
//#define time_cat(cat)

#ifdef TIME_VERBOSE
#define time_cat_verbose(cat) time_cat_verbose_fcn(cat)
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

// Enum string and macro
std::vector<std::string> get_enum_vector(std::string comma_separated_enums);

int get_enum_from_string(std::vector<std::string> enum_names_vector, std::string str);

#define enum_str(name, vec_name, ...) enum name {__VA_ARGS__}; const std::vector<std::string> vec_name = common::get_enum_vector(#__VA_ARGS__);

// Logging
enum_str(log_t, log_t_vec, 
	log_timing, log_timing_verbose, log_accuracy,
	log_estimate, log_truth,
	log_optimizer, log_comparison_accuracy, log_comparison_tr, log_comparison_gn, 
	log_consensus, log_chierality,
	log_unit_test_pts_world, log_unit_test_pts_camera)
	//log_test1, log_test2, log_test3, log_test4, log_test5, log_test6, log_test7, log_test8, log_test9, log_test10)

extern std::vector<bool> logs_enabled;

void init_logs(std::string yaml_filename, std::string result_directory);

void write_log(log_t log_id, const char* s, int n);

void close_logs();

// Assert
#define release_assert(expr) { if (!(expr)) common::release_error(#expr, __FILE__, __LINE__, __func__); }

class Exception : public std::exception
{
public:
	Exception();

	Exception(std::string _msg);

	const char* what() const throw();

	std::string msg;
};

void release_error(const char* expr, const char* file, int line, const char* func);

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
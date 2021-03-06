#include "common/common.h"


using namespace std;
using namespace Eigen;
namespace fs = std::experimental::filesystem;


///////////////////////
// String Formatting //
///////////////////////

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

string common::replace(string str, const string from, const string to)
{
	size_t start_pos = 0;
	while((start_pos = str.find(from, start_pos)) != string::npos)
	{
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
	}
	return str;
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

// Single ?????????????????????????????????
// Mixed  ??????????????????????????????????????????????????????
// Double ?????????????????????????????????

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
					prefix += "???";
				else
					prefix += " ";
			}
			else
			{
				if(numChildren[i] > 0)
					prefix += "???";
				else
					prefix += "???";
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
	string progress_bar = repeat_str("???", bars) + repeat_str(" ", progress_bar_size - bars);
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

double catTimes_ms[common::TIME_CATS_COUNT] = {}; // Init to zero
double catTimesVerbose_ms[common::TIME_CATS_VERBOSE_COUNT] = {}; // Init to zero
double catTimerStartTime;
double catTimerVerboseStartTime;
common::TimeCategory currentTimeCat = common::TimeCatNone;
common::TimeCategoryVerbose currentTimeCatVerbose = common::TimeCatVerboseNone;

void common::cat_timer_reset()
{
	for(int i = 0; i < common::TIME_CATS_COUNT; i++)
		catTimes_ms[i] = 0;
	for(int i = 0; i < common::TIME_CATS_VERBOSE_COUNT; i++)
		catTimesVerbose_ms[i] = 0;
	currentTimeCat = TimeCatNone;
	currentTimeCatVerbose = TimeCatVerboseNone;
}

// Fast category timer, overhead of aprox 0.1us
void common::time_cat_fcn(common::TimeCategory timeCat)
{
	// Get elapsed time
	double currTime = get_wall_time();
	double elapsedTime = currTime - catTimerStartTime;
	
	// Add timer value to 
	if(currentTimeCat != TimeCatNone)
		catTimes_ms[(int)currentTimeCat] += elapsedTime * 1000;

	// Start next timer
	catTimerStartTime = currTime;
	currentTimeCat = timeCat;
}

void common::time_cat_verbose_fcn(common::TimeCategoryVerbose timeCatVerbose)
{
	// Get elapsed time
	double currTime = get_wall_time();
	double elapsedTime = currTime - catTimerVerboseStartTime;
	
	// Add timer value to 
	if(currentTimeCatVerbose != TimeCatVerboseNone)
		catTimesVerbose_ms[(int)currentTimeCatVerbose] += elapsedTime * 1000;

	// Start next timer
	catTimerVerboseStartTime = currTime;
	currentTimeCatVerbose = timeCatVerbose;
}

void common::cat_timer_print()
{
	for(int i = 0; i < TIME_CATS_COUNT; i++)
		printf("Cat %d, time %f\n", i, catTimes_ms[i]);
}

double* common::get_cat_times()
{
	return catTimes_ms;
}

double* common::get_cat_times_verbose()
{
	return catTimesVerbose_ms;
}

/////////////////
// Enum String //
/////////////////

vector<string> common::get_enum_vector(string comma_separated_enums)
{
	// Tokenize the comma-separated enum names
	string enums_names_no_spaces = replace(comma_separated_enums, " ", "");
	vector<string> enum_names_vector;
	Tokenizer tokenizer = Tokenizer(enums_names_no_spaces);
	for(int i = 0; tokenizer.hasToken(); i++)
	{
		Tokenizer item = tokenizer.nextToken(',');

		// Remove enum prefix and underscore
		if(item.countTokens('_') > 1)
			item.nextToken('_');
		enum_names_vector.push_back(item.str());
	}
	return enum_names_vector;
}

int common::get_enum_from_string(vector<string> enum_names_vector, string str)
{
	// Find the enum name that matches the input
	for(int i = 0; i < enum_names_vector.size(); i++)
	{
		if(str == enum_names_vector[i])
		{
			cout << str << " = " << i << endl;
			return i;
		}
	}
	string options = "";
	for(int i = 0; i < enum_names_vector.size(); i++)
	{
		if (i > 0)
			options += ", ";
		options += enum_names_vector[i];
	}
	cout << str << " is not a valid option. Valid options are {" << options << "}" << endl;
	exit(EXIT_FAILURE);
}


/////////////
// Logging //
/////////////

std::vector<std::ofstream> log_files(common::log_t_vec.size());
std::vector<bool> common::logs_enabled(common::log_t_vec.size(), false);

void common::init_logs(string yaml_filename, string result_directory)
{
	YAML::Node node = YAML::LoadFile(yaml_filename);
	for(int i = 0; i < log_t_vec.size(); i++)
	{
		string log_name = log_t_vec[i];
		string log_param = "log_" + log_name;
		string log_filename = fs::path(result_directory) / (log_name + ".bin");
		if (node[log_param] && node[log_param].as<bool>())
		{
			cout << "Opening log file " << log_filename << endl;
			log_files[i].open(log_filename);
			if(log_files[i].is_open())
				logs_enabled[i] = true;
			else
				cout << "Error opening log file " << log_filename << endl;
		}
	}
}

void common::write_log(log_t log_id, const char* s, int n)
{
	int i = (int)log_id;
	assert(i >= 0 && i < log_files.size());
	if(logs_enabled[i])
		log_files[i].write(s, n);
}

void common::close_logs()
{
	for(int i = 0; i < log_t_vec.size(); i++)
		if (log_files[i].is_open())
			log_files[i].close();
}

////////////////////////////////////////////
// Release Assert and Determinism Checker //
////////////////////////////////////////////

common::Exception::Exception()
{
}

common::Exception::Exception(string _msg) : msg(_msg)
{
}

const char* common::Exception::what() const throw()
{
	return msg.c_str();
}

void common::release_error(const char* expr, const char* file, int line, const char* func)
{
	string msg = str_format("Assertion failed: %s, file %s, line %d, in %s", expr, file, line, func);
	throw Exception(msg);
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
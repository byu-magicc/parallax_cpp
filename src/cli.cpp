#include <eigen3/Eigen/Dense>
#include <iostream>
#include "gnsac_ptr_eig.h"
#include "gnsac_ptr_ocv.h"
#include "gnsac_eigen.h"
#include "common.h"
#include <vector>
#include <fstream>
#include <random>
#include <chrono>
#include <memory>
#include <experimental/filesystem>

#include "opencv2/core/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/core/eigen.hpp"

using namespace std;
using namespace Eigen;
using namespace common;
namespace fs = std::experimental::filesystem;

string readFile(string filename)
{
	ifstream file(filename);
	if(!file.is_open())
		throw common::Exception("Error opening file \"" + filename + "\"");
	stringstream ss;
	ss << file.rdbuf();
	return ss.str();
	file.close();
}

void yamlToKeyVals(string yaml, vector<string>& keys, vector<string>& vals)
{
	Tokenizer tokenizer = Tokenizer(yaml);
	while(tokenizer.hasToken())
	{
		Tokenizer line = tokenizer.nextToken('\n');
		if(line.countTokens(':') == 2)
		{
			keys.push_back(line.nextToken(':').str());
			vals.push_back(line.nextToken(':').str());
		}
		else
		{
			keys.push_back(line.str());
			vals.push_back("");
		}
	}
}

string keyValToStr(string key, string val)
{
	if(val != "")
		return key + ":" + val + "\n";
	else
		return key + "\n";
}

string concatYamlStr(string yaml1, string yaml2, bool separate = false)
{
	vector<string> keys1, keys2;
	vector<string> vals1, vals2;
	yamlToKeyVals(yaml1, keys1, vals1);
	yamlToKeyVals(yaml2, keys2, vals2);
	
	// Add keys from yaml1 if they aren't overwritten by yaml2.
	string yamlOut = "";
	for(int i = 0; i < keys1.size(); i++)
	{
		bool found = false;
		for(int j = 0; j < keys2.size(); j++)
			if(keys1[i] == keys2[j])
				found = true;
		if(!found)
			yamlOut += keyValToStr(keys1[i], vals1[i]);
	}
	if(separate)
		yamlOut += "\n";

	// Add everything from yaml2
	for(int i = 0; i < keys2.size(); i++)
		yamlOut += keyValToStr(keys2[i], vals2[i]);
	return yamlOut;
}

string readYamlIncludeOpt(string filename, string directory)
{
	string yaml = readFile(filename);
	vector<string> keys;
	vector<string> vals;
	yamlToKeyVals(yaml, keys, vals);
	string yamlOut = "";
	for(int i = 0; i < keys.size(); i++)
	{
		if(keys[i] == "!include")
			yamlOut = concatYamlStr(yamlOut, readYamlIncludeOpt(fs::path(directory) / vals[i], directory)) + "\n";
		else
			yamlOut = concatYamlStr(yamlOut, keyValToStr(keys[i], vals[i]));
	}
	return yamlOut;
}

void concatFiles(string name1, string name2, string name_out, string directory)
{
	string yaml1 = readYamlIncludeOpt(name1, directory);
	string yaml2 = readYamlIncludeOpt(name2, directory);
	string yaml_out = concatYamlStr(yaml1, yaml2, true);
	ofstream file_out;
	file_out.open(name_out);
	file_out << yaml_out;
	file_out.close();
}

void run_test(string video_str, string test_str, string solver_str, int frames = -1)
{
	// Create folder for results
	string dir_part1 = fs::path("../logs");
	string dir_part2 = fs::path("../logs") / video_str;
	string dir_part3 = fs::path("../logs") / video_str / test_str;
	string result_directory = fs::path("../logs") / video_str / test_str / solver_str;
	cout << result_directory << endl;
	if(!fs::exists(dir_part1))
		fs::create_directory(dir_part1);
	if(!fs::exists(dir_part2))
		fs::create_directory(dir_part2);
	if(!fs::exists(dir_part3))
		fs::create_directory(dir_part3);
	if(!fs::exists(result_directory))
		fs::create_directory(result_directory);

	// Load point data
	string video_yaml = fs::path("../param/videos") / (video_str + ".yaml");
	cout << video_yaml << endl;
	VideoPointData video_data = VideoPointData(video_yaml);
	if (video_data.RT.size() < video_data.pts1.size())
	{
		printf("Error: Missing truth data (%d < %d)\n", (int)video_data.RT.size(), (int)video_data.pts1.size());
		assert(0);
	}
	else
		printf("Warning: Truth data size does not match point data size (%d != %d)\n",
			(int)video_data.RT.size(), (int)video_data.pts1.size());

	// Concatenate solver and test to create a run yaml
	string solver_yaml = fs::path("../param/tests") / test_str / (solver_str + ".yaml");
	string test_yaml = fs::path("../param/tests") / test_str / (test_str + ".yaml");
	string run_yaml = fs::path(result_directory) / "solver.yaml";
	cout << "Solver: " << solver_yaml << endl;
	cout << "Tests: " << test_yaml << endl;
	concatFiles(solver_yaml, test_yaml, run_yaml, "../param/solvers");
	shared_ptr<ESolver> solver = ESolver::from_yaml(run_yaml);

	// Random number generation
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine rng(seed);
	std::normal_distribution<double> dist(0.0, 1.0);

	// Open log files
	common::init_logs(run_yaml, result_directory);

	// Loop for all points
	if (frames == -1)
		frames = video_data.pts1.size();
	common::progress(0, frames);
	for (int frame = 0; frame < frames; frame++)
	{
		// Undistort points
		scan_t pts1, pts2;
		undistort_points(video_data.pts1[frame], pts1, video_data.camera_matrix);
		undistort_points(video_data.pts2[frame], pts2, video_data.camera_matrix);

		// Calculate essential matrix
		Matrix3d R0 = Matrix3d::Identity();
		Vector3d t0;
		t0 << dist(rng), dist(rng), dist(rng);
		tic();
		cat_timer_reset();
		EHypothesis result;
		solver->find_best_hypothesis(pts1, pts2, video_data.RT[frame], result);
		timeMeasurement time_E = toc("FindE", 1, 2, false);
		common::write_log(common::log_timing, (char*)get_cat_times(), sizeof(double) * TIME_CATS_COUNT);
		common::write_log(common::log_timing, (char*)&time_E.actualTime, sizeof(double));
		common::write_log(common::log_timing_verbose, (char*)get_cat_times_verbose(), sizeof(double) * TIME_CATS_VERBOSE_COUNT);
		common::write_log(common::log_timing_verbose, (char*)&time_E.actualTime, sizeof(double));

		// Calculate error to truth essential matrix.
		Vector4d err;
		if(result.has_RT)
			err = common::err_truth(result.R, result.t, video_data.RT[frame]);
		else
			err = common::err_truth(result.E, video_data.RT[frame]);
		Matrix4d RT_est = common::RT_combine(result.R, result.t);
		common::write_log(common::log_estimate, (char*)RT_est.data(), sizeof(double)*4*4);
		common::write_log(common::log_truth, (char*)video_data.RT[frame].data(), sizeof(double)*4*4);
		common::write_log(common::log_accuracy, (char*)err.data(), sizeof(double) * 4);
		common::progress(frame + 1, frames);
	}
	common::close_logs();
}

void run_tests(string video_str, string test_str)
{
	// First make sure that the test dir and yaml exists
	string solver_dir = fs::path("../param/solvers");
	string test_dir = fs::path("../param/tests") / test_str;
	string test_yaml = fs::path("../param/tests") / test_str / (test_str + ".yaml");
	if(!fs::exists(test_yaml))
		throw common::Exception("Test \"" + test_yaml + "\" does not exist.");

	// Search test folder to obtain all yaml files. Make sure they can all be loaded succesfully
    for (auto& solver_yaml : fs::directory_iterator(test_dir))
	{
		if(solver_yaml.path() != test_yaml)
		{
			try
			{
				string unused = readYamlIncludeOpt(solver_yaml.path(), solver_dir);
			}
			catch(common::Exception e)
			{
				cout << RED_TEXT << "Error loading " << solver_yaml << BLACK_TEXT << endl;
				cout << e.what() << endl;
			}
		}
	}

	// Now run the tests!
	cout << GREEN_TEXT << "Running test " << test_yaml << " with all solvers in directory" BLACK_TEXT << endl;
    for (auto& solver_yaml : fs::directory_iterator(test_dir))
	{
		cout << GREEN_TEXT << "Solver " << solver_yaml.path() << BLACK_TEXT << endl;
		string solver_str = solver_yaml.path().stem();
		run_test(video_str, test_str, solver_str);
	}
}

int main(int argc, char *argv[])
{
	// Get rid of first arg (executable name)
	argc--; argv++;

	// Run unit tests
	if (argc == 1 && string(argv[0]) == "unit")
	{
		gnsac_eigen::run_tests();
		return 0;
	}

	// Run test with all solvers in directory
	if(argc == 2)
	{
		string video = string(argv[0]);
		string test = string(argv[1]);
		run_tests(video, test);
		return 0;
	}

	// Run test with a specific solver
	if(argc >= 2)
	{
		string video = string(argv[0]);
		string test = string(argv[1]);
		string solver = string(argv[2]);
		int frames = (argc >= 4) ? atoi(argv[3]) : -1;
		run_test(video, test, solver, frames);
		return 0;
	}

	// If there weren't sufficient arguments, exit.
	cout << "Usage: Run from the parallax_cpp/build folder." << endl;
	cout << "Enter yaml filenames without folder directory or extension (ie. enter \"gn_eigen\" for \"../param/solvers/gn_eigen.yaml\"" << endl;
	cout << "./cli unit" << endl;
	cout << "./cli video_yaml test_yaml [solver_yaml [frames]]" << endl;

}
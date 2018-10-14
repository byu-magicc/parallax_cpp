#include "solvers.h"
#include "common.h"
#include "comm_loaders.h"
#include "gnsac_ptr_eig.h"
#include "gnsac_ptr_ocv.h"
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include "calib3d.hpp"

using namespace std;
using namespace Eigen;

common::EHypothesis::EHypothesis() : cost(0), E(Matrix3d::Zero()), R(Matrix3d::Zero()), t(Vector3d::Zero())
{
	
}

common::EHypothesis::EHypothesis(Matrix3d E_) : cost(0), E(E_), R(Matrix3d::Zero()), t(Vector3d::Zero())
{
	
}

common::EHypothesis::EHypothesis(Matrix3d E_, Matrix3d R_, Vector3d t_) : cost(0), E(E_), R(R_), t(t_)
{
	
}


common::ESolver::ESolver(string yaml_filename, YAML::Node node)
{
	
}

void common::ESolver::generate_hypotheses(const scan_t& subset1, const scan_t& subset2, const EHypothesis& initial_guess, vector<EHypothesis>& hypotheses)
{
	cout <<"Error: Not Implemented!" << endl;
	exit(EXIT_FAILURE);
}

double common::ESolver::score_hypothesis(const scan_t& pts1, const scan_t& pts2, const EHypothesis& hypothesis)
{
	cout <<"Error: Not Implemented!" << endl;
	exit(EXIT_FAILURE);
}

void common::ESolver::refine_hypothesis(const scan_t& pts1, const scan_t& pts2, const EHypothesis& best_hypothesis, EHypothesis& result)
{
	cout <<"Error: Not Implemented!" << endl;
	exit(EXIT_FAILURE);
}

void common::ESolver::find_best_hypothesis(const scan_t& pts1, const scan_t& pts2, const Matrix4d& RT_truth, EHypothesis& result)
{
	cout <<"Error: Not Implemented!" << endl;
	exit(EXIT_FAILURE);
}

shared_ptr<common::ESolver> common::ESolver::from_yaml(string yaml_filename)
{
	YAML::Node node = YAML::LoadFile(yaml_filename);
	string library_name;
	get_yaml_node("library", yaml_filename, node, library_name);
	if (library_name == "gnsac_ptr_eigen")
	{
		gnsac_ptr_eigen::GNSAC_Solver* ptr1 = new gnsac_ptr_eigen::GNSAC_Solver(yaml_filename, node);
		ESolver* ptr2 = (ESolver*)ptr1;
		return shared_ptr<ESolver>(ptr2);
	}
	else if (library_name == "five_point_opencv")
	{
		five_point_opencv::FivePointSolver* ptr1 = new five_point_opencv::FivePointSolver(yaml_filename, node);
		ESolver* ptr2 = (ESolver*)ptr1;
		return shared_ptr<ESolver>(ptr2);
	}
	//else if (library_name == "gnsac_ptr_opencv")
	//   return gnsac_ptr_opencv::GNSAC_Solver(yaml_filename, node);
	else
	{
		cout << "Library name \"" << library_name << "\" not recognized" << endl;
		exit(EXIT_FAILURE);
	}
}
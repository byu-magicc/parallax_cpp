#include "common/solvers.h"

#include "five_point_opencv/calib3d.hpp"
#include "solvers/gnsac_ptr_eig.h"
#include "solvers/gnsac_ptr_ocv.h"
#include "solvers/gnsac_eigen.h"


using namespace std;
using namespace Eigen;

common::EHypothesis::EHypothesis() : cost(0), E(Matrix3d::Zero()), R(Matrix3d::Zero()), t(Vector3d::Zero()), has_RT(false)
{
	
}

common::EHypothesis::EHypothesis(Matrix3d& E_) : cost(0), E(E_), R(Matrix3d::Zero()), t(Vector3d::Zero()), has_RT(false)
{
	
}

common::EHypothesis::EHypothesis(Matrix3d& E_, Matrix3d& R_, Vector3d& t_) : cost(0), E(E_), R(R_), t(t_), has_RT(true)
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
	YAML::Node node;

	// If the file exists, load the parameters form the file; otherwise
	// create default values
	
	try {
		node = YAML::LoadFile(yaml_filename);

	} catch (...) {
		std::cout << "Unable to load GNSAC param file: " << yaml_filename << "Using default values." << std::endl;

		node["library"] = "gnsac_eigen";
		node["optimizer"] = "LM";
		node["max_iterations"] = 10;
		node["exit_tolerance"] = 0;
		node["optimizer_cost"] = "sampson";
		node["LM_lambda"] = 1e-4;
		node["optimizer_seed"] = "best";
		node["optimizer_seed_noise"] = 0;
		node["n_subsets"] = 100;
		node["n_subsets_ignore_after"] = 100000;
		node["RANSAC_threshold"] = 1e-2;
		node["scoring_cost"] = "sampson";
		node["consensus_alg"] = "LMEDS";
		node["consensus_seed"] = "previous";
		node["pose_disambig"] = "trace";
		node["renormalize"] = false;
		node["refine"] = true;
	}


	string library_name;
	get_yaml_node("library", yaml_filename, node, library_name);
	if (library_name == "gnsac_ptr_eigen")
	{
		shared_ptr<gnsac_ptr_eigen::GNSAC_Solver> ptr1 = make_shared<gnsac_ptr_eigen::GNSAC_Solver>(yaml_filename, node);
		return dynamic_pointer_cast<ESolver>(ptr1);
	}
	if (library_name == "gnsac_eigen")
	{
		shared_ptr<gnsac_eigen::GNSAC_Solver> ptr1 = make_shared<gnsac_eigen::GNSAC_Solver>(yaml_filename, node);
		return dynamic_pointer_cast<ESolver>(ptr1);
	}
	else if (library_name == "five_point_opencv")
	{
		shared_ptr<five_point_opencv::FivePointSolver> ptr1 = make_shared<five_point_opencv::FivePointSolver>(yaml_filename, node);
		return dynamic_pointer_cast<ESolver>(ptr1);
	}
	//else if (library_name == "gnsac_ptr_opencv")
	//   return gnsac_ptr_opencv::GNSAC_Solver(yaml_filename, node);
	else
	{
		cout << "Library name \"" << library_name << "\" not recognized" << endl;
		exit(EXIT_FAILURE);
	}
}
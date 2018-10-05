#include "solvers.h"
#include "common.h"
#include "comm_loaders.h"
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <string>
#include <iostream>

using namespace std;
using namespace Eigen;

common::ESolver::ESolver(string yaml_filename, YAML::Node node)
{
    
}

void common::ESolver::generate_hypotheses(const scan_t& min_subset1, const scan_t& min_subset2, std::vector<Eigen::Matrix3d>& hypotheses)
{
    
}

double common::ESolver::score_hypothesis(const scan_t& pts1, const scan_t& pts2, const EHypothesis& hypothesis)
{

}

void common::ESolver::find_best_hypothesis(const scan_t& pts1, const scan_t& pts2, EHypothesis& result, const EHypothesis hypothesis0)
{

}

common::ESolver common::ESolver::from_yaml(string yaml_filename)
{
	YAML::Node node = YAML::LoadFile(yaml_filename);
    string class_name;
    get_yaml_node("class", yaml_filename, node, class_name);
    // if (class_name == "gnsac_ptr_eigen")
    //     return gnsac_ptr_eigen::GNSAC_Solver(yaml_filename, node);
    // else if (class_name == "gnsac_ptr_opencv")
    //     return gnsac_ptr_opencv::GNSAC_Solver(yaml_filename, node);
    // else
    {
        cout << "Class name \"" << class_name << "\" not recognized" << endl;
        exit(EXIT_FAILURE);
    }
}
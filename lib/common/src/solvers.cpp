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
    //else if (library_name == "gnsac_ptr_opencv")
    //   return gnsac_ptr_opencv::GNSAC_Solver(yaml_filename, node);
    else
    {
        cout << "Library name \"" << library_name << "\" not recognized" << endl;
        exit(EXIT_FAILURE);
    }
}
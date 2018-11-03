#ifndef SOLVERS_H
#define SOLVERS_H

#include "comm_loaders.h"
#include <string>
#include <vector>
#include <memory>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>

namespace common
{

class EHypothesis
{
public:
	EHypothesis();
	EHypothesis(Eigen::Matrix3d& E);
	EHypothesis(Eigen::Matrix3d& E, Eigen::Matrix3d& R, Eigen::Vector3d& t);
	Eigen::Matrix3d E;
	Eigen::Matrix3d R;
	Eigen::Vector3d t;
	double cost;
	bool has_RT;
};

// Note that the "vitual" keyword is important, otherwise the parent class method 
// gets called if calling it from a parent class pointer!
// See https://stackoverflow.com/questions/2391679/why-do-we-need-virtual-functions-in-c
class ESolver
{
public:
	ESolver(std::string yaml_filename, YAML::Node node);

public:
	virtual void generate_hypotheses(const scan_t& subset1, const scan_t& subset2, const EHypothesis& initial_guess, std::vector<EHypothesis>& hypotheses);

	virtual void refine_hypothesis(const scan_t& pts1, const scan_t& pts2, const EHypothesis& best_hypothesis, EHypothesis& result);

	virtual void find_best_hypothesis(const scan_t& pts1, const scan_t& pts2, const Eigen::Matrix4d& RT_truth, EHypothesis& result);

	virtual double score_hypothesis(const scan_t& pts1, const scan_t& pts2, const EHypothesis& hypothesis);

	static std::shared_ptr<ESolver> from_yaml(std::string yaml_filename);
};

}

#endif //SOLVERS_H
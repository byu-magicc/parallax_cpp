#ifndef SOLVERS_H
#define SOLVERS_H

#include "comm_loaders.h"
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <eigen3/Eigen/Eigen>

namespace common
{

class EHypothesis
{
public:
	EHypothesis() : cost(0)
	{
		
	}
	Eigen::Matrix3d E;
	double cost;
};

class ESolver
{
private:
	ESolver(std::string yaml_filename, YAML::Node node);

public:
	void generate_hypotheses(const scan_t& min_subset1, const scan_t& min_subset2, std::vector<Eigen::Matrix3d>& hypotheses);

	double score_hypothesis(const scan_t& pts1, const scan_t& pts2, const EHypothesis& hypothesis);

	void find_best_hypothesis(const scan_t& pts1, const scan_t& pts2, EHypothesis& result, const EHypothesis hypothesis0 = EHypothesis());

	static ESolver from_yaml(std::string yaml_filename);
};

}

#endif //SOLVERS_H
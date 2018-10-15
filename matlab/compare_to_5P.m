function [result_GN, result_5P, frames] = compare_to_5P(video, method)
	%% Load data
	A  = read_binary(['../logs/' video '/' method '/5-point_accuracy.bin'],  11);
	C_TR = read_binary(['../logs/' video '/' method '/5-point_comparison_tr.bin'], [2, 11]);
	C_GN = read_binary(['../logs/' video '/' method '/5-point_comparison_gn.bin'], [2, 10]);
	C_TR(C_TR == -1) = nan;
	C_GN(C_GN == -1) = nan;
	A(A == -1) = nan;
	TR = reshape(sum(C_TR .^ 2, 1), 11, []);
	GN = reshape(sum(C_GN .^ 2, 1), 10, []);
	frames = size(A, 2) / 100;

	%% Categorize all hypotheses
	% How many total GN/LM and 5P hypotheses?
	total_GN = size(A, 2);
	num_5P = sum(~isnan(A(1:10, :)), 1);
	total_5P = sum(num_5P);

	% How many GN/LM and 5P hypotheses converged to a valid solution?
	% (note that the sampson err is the residual squared, so 1e-20 makes sense)
	convergence_threshold = 1e-20;
	valid_GN = A(11, :) < convergence_threshold;
	valid_5P = A(1:10, :) < convergence_threshold; %if nan gives 0.
	num_valid_GN = sum(valid_GN);
	num_valid_5P = sum(sum(valid_5P));

	% Did GN/LM converge to the best 5-point soln'?
	close_threshold = 1e-6;
	[dist_GN, closest_GN] = min(GN, [], 1);
	[~, closest_TR] = min(TR(1:10, :), [], 1);
	converged_to_best_5P_soln = valid_GN & (dist_GN < close_threshold) & (closest_TR == closest_GN);

	% Did GN/LM converge to a better (closer to truth) solution than the 5-point algorithm?
	[~, best_TR] = min(TR, [], 1);
	converged_to_different_soln = valid_GN & (best_TR == 11);
	mean(converged_to_different_soln);

	% How many GN/LM hypotheses acheived a best soln?
	num_best_GN = sum(converged_to_best_5P_soln | converged_to_different_soln);
	num_best_5P = sum(num_5P > 0);

	% final categories
	result_GN = [num_best_GN, num_valid_GN - num_best_GN, total_GN - num_valid_GN];
	result_5P = [num_best_5P, num_valid_5P - num_best_5P, total_5P - num_valid_5P];
end
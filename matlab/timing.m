title_str = 'Timing';
video = 'holodeck';
test = 'standard';
methods = {'poly_opencv', 'gn_eigen', 'lm_eigen'};
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
%methods = {'five-point', 'gn', 'lm', 'gn_eigen'};
ylabels = {'Hypothesis Generation', 'Hypothesis Scoring', 'Total'};
filename = 'timing.bin';
plot_comparison(title_str, video, test, methods, ylabels, filename)
return

%% Bar graph
% Read hypothesis categories
video = 'holodeck';
test = 'standard';
methods = {'poly_opencv', 'gn_eigen', 'lm_eigen'};
bar_lgnd = cell(1, length(methods));

% Read timing
total_lgnd = {'HypoGen', 'HypoScore', 'Refine', 'Disambiguate', 'Total'};
t_gen = zeros(length(methods), 1);
t_score = zeros(length(methods), 1);
for i = 1:length(methods)
	method = methods{i};
	t = read_binary(['../logs/' video '/' test '/' method '/timing.bin'], length(total_lgnd));
	t_gen(i) = mean(t(1, :));
	t_score(i) = mean(t(2, :));
	bar_lgnd{i} = replace(methods{i}, '_', ' ');
end

% read verbose timing
gen_lgnd = {'Setup', 'SVD', 'Coeffs1', 'Coeffs2', 'Coeffs3', 'SolvePoly', 'ConstructE', ...
	'MakeJ', 'SolveMatrix', 'ManifoldUpdate', 'CalcResidual', 'Total'};
gen_time = zeros(length(methods), length(gen_lgnd));
for i = 1:length(methods)
	method = methods{i};
	t = read_binary(['../logs/' video '/' test '/' method '/timing_verbose.bin'], length(gen_lgnd));
	gen_time(i, :) = mean(t, 2).';
end

% Hypothesis generation timing (verbose)
figure
b = bar(gen_time(:, 1:end-1), 'stacked', 'FaceColor', 'flat');
n_groups = length(methods);
legend(gen_lgnd{1:end-1})
ylabel('Hypothesis generation time (ms)')
xticklabels(bar_lgnd);
title('Hypothesis Generation Time')
grid on
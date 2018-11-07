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
methods = {'lm_eigen', 'gn_eigen', 'poly_opencv', 'lm_refine'};
n_subsets = 100;

% n_pts
filename = '../logs/holodeck/truth_mag/gn_eigen/truth_magnitude.bin';
A = read_binary(filename, 3);
n_pts = mean(A(3, :));

% Read and print timing
total_lgnd = {'HypoGen', 'HypoScore', 'Refine', 'Disambiguate', 'Total'};
t_gen = zeros(length(methods), 1);
t_score = zeros(length(methods), 1);
fprintf('%-25s %-15s %-15s %-15s %-15s %-15s\n', 'Method', 'HypoGen', 'HypoScore', 'Refine', 'Disambiguate', 'Total')
for i = 1:length(methods)
	method = methods{i};
	t = read_binary(['../logs/' video '/' test '/' method '/timing.bin'], length(total_lgnd));
	t = mean(t, 2);
	fprintf('%-25s %-15e %-15e %-15e %-15e %-15e\n', methods{i}, t)
	fprintf('%-25s %-15e %-15e %-15e %-15e %-15e\n', ' - per subset', t / n_subsets)
	fprintf('%-25s %-15e %-15e %-15e %-15e %-15e\n', ' - per point', t / n_pts)
end

%% read verbose timing
gen_lgnd = {'Setup', 'SVD', 'Coeffs1', 'Coeffs2', 'Coeffs3', 'SolvePoly', 'ConstructE', ...
	'MakeJ', 'SolveMatrix', 'ManifoldUpdate', 'CalcResidual', 'Total'};
gen_time = zeros(length(methods), length(gen_lgnd));
bar_lgnd = cell(1, length(methods));
for i = 1:length(methods)
	method = methods{i};
	t = read_binary(['../logs/' video '/' test '/' method '/timing_verbose.bin'], length(gen_lgnd));
	gen_time(i, :) = mean(t, 2).' / n_subsets;
	bar_lgnd{i} = replace(methods{i}, '_', ' ');
end

% OpenCV
figure(1)
method_idxs = [3 3];
lgnd_idxs = 1:7;
b = barh([gen_time(method_idxs, lgnd_idxs)]*1000, 'stacked', 'FaceColor', 'flat');
n_groups = length(methods);
legend(gen_lgnd{lgnd_idxs})
xlabel('Hypothesis generation time per subset (us)')
yticklabels(bar_lgnd(method_idxs));
grid on
xlim([0, 500])
delta_pos = [0.05, 0, -0.05, 0];
set(gca, 'Position', get(gca, 'Position') + delta_pos)

% LM/GN
figure(2)
method_idxs = [1 2];
lgnd_idxs = 8:11;
b = barh([gen_time(method_idxs, lgnd_idxs)]*1000, 'stacked', 'FaceColor', 'flat');
n_groups = length(methods);
legend(gen_lgnd{lgnd_idxs})
xlabel('Hypothesis generation time per subset (us)')
yticklabels(bar_lgnd(method_idxs));
grid on
delta_pos = -[0.05, 0, -0.05, 0];
set(gca, 'Position', get(gca, 'Position') + delta_pos)
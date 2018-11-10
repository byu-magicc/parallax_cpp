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
	n_hypotheses = read_binary(['../logs/' video '/' test '/' method '/num_hypotheses.bin'], 1);
	n_hypotheses = mean(n_hypotheses);
	fprintf('%-25s %-15e %-15e %-15e %-15e %-15e\n', methods{i}, t)
	fprintf('%-25s %-15e %-15e %-15e %-15e %-15e\n', ' - per subset', t / n_subsets)
	fprintf('%-25s %-15e %-15e %-15e %-15e %-15e\n', sprintf(' - per hypothesis (%.2f)', n_hypotheses), t / n_subsets / n_hypotheses)
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

%% OpenCV
gen_lgnd = {'Setup', 'SVD', 'Coeffs1', 'Coeffs2', 'Coeffs3', 'Solve poly', 'Construct E', ...
	'Make Jacobian', 'Solve matrix', 'Manifold update', 'Calc residual', 'Total'};
bar_lgnd = {'LM + LMedS', 'GN + LMedS', 'OpenCV Poly'}
x_limits = [0 500];
ax1 = axes();
method_idxs = 1:3;
lgnd_idxs = 1:7;
b = barh(gen_time(method_idxs, lgnd_idxs)*1000, 'stacked', 'FaceColor', 'flat', 'Parent', ax1);
set(ax1, 'Box', 'off') % turn off the tick marks around the legend
n_groups = length(methods);
legend(gen_lgnd{lgnd_idxs})
xlabel('Hypothesis generation time per subset (microseconds)')
yticklabels(bar_lgnd(method_idxs));
grid on
xlim(x_limits)
delta_pos = [0.02, 0.04, -0.05, -0.03];
set(gca, 'Position', get(gca, 'Position') + delta_pos)
set(gcf, 'Position', [560   674   560   258]);

% LM/GN
ax2 = copyobj(ax1, gcf); % copy axis
co = [211, 0, 0
      64/2, 147, 0
      0, 119/2, 255
      94, 25, 255
      1, 41, 95] / 255;
set(ax2, 'ColorOrder', co, 'NextPlot', 'replacechildren')
delete(get(ax2, 'Children'));
lgnd_idxs = 8:11;
b = barh(gen_time(method_idxs, lgnd_idxs)*1000, 'stacked', 'FaceColor', 'flat', 'Parent', ax2);
n_groups = length(methods);
set(ax2, 'Color', 'none', 'YTick', [], 'XTick', [], 'Box', 'off', 'XLim', x_limits)
%set(ax2, 'Position', get(ax1, 'Position'))
legend(ax2, gen_lgnd{lgnd_idxs}, 'Location', 'SouthEast')
ax2.XLabel.String = ''

%% Swap order if we need to drag the other legend around.
figChildren = get(gcf, 'Children')
uistack(figChildren([3, 4]));

%% Change xlabel
figChildren = get(gcf, 'Children')
figChildren(4).XLabel.String = 'Hypothesis generation time per subset (microseconds)';
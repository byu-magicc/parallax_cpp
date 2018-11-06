%% GN vs. LM: What percent are converged after a certian number of iterations?
% Read in data (these are big files)
video = 'holodeck';
test = 'optimizer';
methods = {'gn_eigen', 'lm_eigen'};
n_iters = 50;
n_subsets = 100;
O = zeros(5, n_iters, 3599 * n_subsets, length(methods));
for i = 1:length(methods)
	method = methods{i};
	filename = ['../logs/' video '/' test '/' methods{i} '/optimizer.bin'];
	O(:, :, :, i) = read_binary(filename, [5, n_iters]);
end

%% Plot results
convergence_threshold = 1e-8;
lgnd = cell(1, length(methods));
hold on;
for i = 1:length(methods)
	% r, delta, lambda, attempts, delta_gn
	pcnt_conv_J = mean(O(5, :, :, i) < convergence_threshold, 3);
	pcnt_conv_r = mean(O(1, :, :, i) < convergence_threshold, 3);
	av_step_size = mean(O(2, :, :, i), 3);
	%plot(1:n_iters, pcnt_conv_r);
	plot(1:n_iters, av_step_size);
	lgnd{i} = replace(methods{i}, '_', ' ');
end
legend(lgnd)
xlabel('Iterations')
ylabel('Percent converged')
grid on;
xlim([0 50])
return

%% Does GN or LM perform better overal?
clc
video = 'holodeck';
test = 'standard';
methods = {'poly_opencv', 'gn_eigen', 'lm_eigen'};
lgnd = cell(1, length(methods));
fprintf('%-15s %-15s %-15s %-15s\n', 'Method', 'Rotation', 'Translation', 'LMedS error')
for i = 1:length(methods)
	filename = ['../logs/' video '/' test '/' methods{i} '/accuracy.bin'];
	A = read_binary(filename, 5);
	fprintf('%-15s %-15e %-15e %-15e\n', methods{i}, mean(A([1 2 5], :), 2).')
	lgnd{i} = replace(methods{i}, '_', ' ');
end

%% Investigate non-convergent plots
% 5 x n_iters x (3599*n_subsets) x length(methods);
O_mask = O(5, end, :, 2) > 1;
O_lm_nc = O(:, :, O_mask, 2);

%% Plot results

% r, delta, lambda, attempts, delta_gn
%histogram(O_lm_nc(5, end, :), 100, 'BinLimits', [0, 1e4])
%histogram(log(O_lm_nc(5, end, :)), 100)
%histogram(O_lm_nc(4, end, :), 6, 'BinLimits', [-0.5, 5.5])
%histogram(log(O_lm_nc(3, end, :)), 100)
%histogram(log(O_lm_nc(2, end, :)), 100)
%histogram(log(O_lm_nc(1, end, :)), 100)

histogram(log(O(2, end, :, 2)), 100)



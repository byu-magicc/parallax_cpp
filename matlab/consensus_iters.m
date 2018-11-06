% Plots the consensus error (Sampson) vs. the number of iterations of the algorithm.
video = 'holodeck';
%methods = {'poly_opencv', 'gn_eigen', 'lm_eigen'};

% Best way to seed algorithm?
%methods = {'gn_eigen_prior', 'gn_eigen_prior_recursive', 'gn_eigen_random', 'gn_eigen_random_recursive'};
methods = {'poly_opencv', 'lm_eigen_random', 'lm_eigen_random_recursive', 'lm_eigen_prior', 'lm_eigen_prior_recursive'};

% Best cost function?
% For GN, the algebraic cost seems to work better, but for LM the sampson cost seems to be slightly better.
%methods = {'poly_opencv', 'lm_eigen', 'lm_eigen_single', 'lm_eigen_sampson'};
%methods = {'poly_opencv', 'gn_eigen', 'gn_eigen_single', 'gn_eigen_sampson'};

% Does it matter if we exit early?
% It doesn't matter with 10 iterations anyway.
%methods = {'gn_eigen', 'gn_eigen_no_exit'};
%methods = {'lm_eigen', 'lm_eigen_no_exit'};
use_mean = 1;
lgnd = cell(size(methods, 1), 1);
test = 'consensus_iters'; n_subsets = 100;
%test = 'consensus_iters200'; n_subsets = 200;
%test = 'consensus_iters1000'; n_subsets = 1000;
hold on;
%clrs = {'b', 'r', 'g', 'm'};
for i=1:length(methods)
	err = read_binary(['../logs/' video '/' test '/' methods{i} '/consensus.bin'], n_subsets);
	if use_mean
		err_mean = mean(err, 2);
		err_std = std(err, 0, 2);
		plot(err_mean);
		%plot(err_mean, clrs{i});
		%plot(err_mean + err_std, [clrs{i} '--']);
		%plot(err_mean - err_std, [clrs{i} '--']);
	else
		%q1 = quantile(err, 0.25, 2);
		q2 = quantile(err, 0.5, 2);
		q3 = quantile(err, 0.75, 2);
		q4 = quantile(err, 0.9, 2);
		plot(q2, clrs{i});
		%h1 = plot(q1, [clrs{i} '--']);
		%h3 = plot(q3, [clrs{i} '--']);
		%h4 = plot(q4, [clrs{i} '--']);
		%set(h1, 'HandleVisibility', 'Off');
		%set(h3, 'HandleVisibility', 'Off');
		%set(h4, 'HandleVisibility', 'Off');
	end
	lgnd{i} = replace(methods{i}, '_', ' ');
end
xlim([0 min(n_subsets, 200)])
ylim([3e-8 10e-8])
legend(lgnd)
xlabel('LMEDS iterations')
ylabel('LMEDS error')
grid on;
% Plots the consensus error (Sampson) vs. the number of iterations of the algorithm.
video = 'holodeck';
%methods = {'poly_opencv', 'gn_eigen', 'lm_eigen'};

% Best way to seed algorithm?
%methods = {'gn_eigen_prior', 'gn_eigen_prior_recursive', 'gn_eigen_random', 'gn_eigen_random_recursive'};
%methods = {'poly_opencv', 'lm_eigen_random', 'lm_eigen_random_recursive', 'lm_eigen_prior', 'lm_eigen_prior_recursive'};
methods = {'poly_opencv', 'lm_eigen_prior_recursive'};

% Best cost function?
% For GN, the algebraic cost seems to work better, but for LM the sampson cost seems to be slightly better.
%methods = {'poly_opencv', 'lm_eigen', 'lm_eigen_single', 'lm_eigen_sampson'};
%methods = {'poly_opencv', 'gn_eigen', 'gn_eigen_single', 'gn_eigen_sampson'};

% Does it matter if we exit early?
% It doesn't matter with 10 iterations anyway.
%methods = {'gn_eigen', 'gn_eigen_no_exit'};
%methods = {'lm_eigen', 'lm_eigen_no_exit'};
use_mean = 1;
scale_t = 1;
lgnd = cell(size(methods, 1), 1);
timing_test = 'standard';
%test = 'consensus_iters'; n_subsets = 100;
%test = 'consensus_iters200'; n_subsets = 200;
test = 'consensus_iters1000'; n_subsets = 1000;
hold on;
%clrs = {'b', 'r', 'g', 'm'};
for i=1:length(methods)
	if strcmp(methods{i}, 'poly_opencv')
		test_i = 'consensus_iters';
		n_subsets_i = 100;
	else
		test_i = test;
		n_subsets_i = n_subsets;
	end
	err = read_binary(['../logs/' video '/' test_i '/' methods{i} '/consensus.bin'], n_subsets_i);
	if scale_t
		if contains(methods{i}, 'lm_eigen')
			timing_method = 'lm_eigen';
		elseif contains(methods{i}, 'gn_eigen')
			timing_method = 'gn_eigen';
		elseif contains(methods{i}, 'poly_opencv')
			timing_method = 'poly_opencv';
		else
			error('Unknown match for method');
		end
		t_av = read_binary(['../logs/' video '/' timing_test '/' timing_method '/timing.bin'], 3);
		t_av = mean(t_av(3, :));
		x = (1:n_subsets_i) * t_av / 100;
	else
		x = 1:n_subsets_i;
	end
	if use_mean
		err_mean = mean(err, 2);
		err_std = std(err, 0, 2);
		plot(x, err_mean);
		%plot(x, err_mean, clrs{i});
		%plot(x, err_mean + err_std, [clrs{i} '--']);
		%plot(x, err_mean - err_std, [clrs{i} '--']);
	else
		%q1 = quantile(err, 0.25, 2);
		q2 = quantile(err, 0.5, 2);
		q3 = quantile(err, 0.75, 2);
		q4 = quantile(err, 0.9, 2);
		plot(x, q2, clrs{i});
		%h1 = plot(x, q1, [clrs{i} '--']);
		%h3 = plot(x, q3, [clrs{i} '--']);
		%h4 = plot(x, q4, [clrs{i} '--']);
		%set(h1, 'HandleVisibility', 'Off');
		%set(h3, 'HandleVisibility', 'Off');
		%set(h4, 'HandleVisibility', 'Off');
	end
	lgnd{i} = replace(methods{i}, '_', ' ');
end
if scale_t
	xlim([0 30])
	xlabel('time (ms)')
	ylim([2.5e-8 10e-8])
else
	xlim([0 min(n_subsets, 200)])
	xlabel('LMEDS iterations')
	ylim([3e-8 10e-8])
end
legend(lgnd)
ylabel('LMEDS error')
grid on;
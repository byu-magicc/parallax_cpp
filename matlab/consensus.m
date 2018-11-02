% Plots the consensus error (Sampson) vs. the number of iterations of the algorithm.
video = 'holodeck';
%methods = {'poly_opencv', 'gn_eigen', 'lm_eigen'};
%methods = {'poly_opencv', 'gn_eigen', 'gn_eigen_no_seed', 'gn_eigen_no_prior'};
%methods = {'poly_opencv', 'lm_eigen', 'lm_eigen_no_seed', 'lm_eigen_no_prior'};
methods = {'poly_opencv', 'gn_eigen_no_seed', 'lm_eigen_no_seed'};
%methods = {'gn_eigen', 'gn_eigen_no_exit'};
%methods = {'lm_eigen', 'lm_eigen_no_exit'};
use_mean = 0;
lgnd = cell(size(methods, 1), 1);
test = 'consensus';
hold on;
clrs = {'b', 'r', 'g', 'm'};
for i=1:length(methods)
	err = read_binary(['../logs/' video '/' methods{i} '/' test '/consensus.bin'], 1000);
	if use_mean
		err_mean = mean(err, 2);
		err_std = std(err, 0, 2);
		plot(err_mean, clrs{i});
		plot(err_mean + err_std, [clrs{i} '--']);
		plot(err_mean - err_std, [clrs{i} '--']);
	else
		q1 = quantile(err, 0.25, 2);
		q2 = quantile(err, 0.5, 2);
		q3 = quantile(err, 0.75, 2);
		%q4 = quantile(err, 0.9, 2);
		plot(q2, clrs{i});
		h1 = plot(q1, [clrs{i} '--']);
		h3 = plot(q3, [clrs{i} '--']);
		%h4 = plot(q4, [clrs{i} '--']);
		set(h1, 'HandleVisibility', 'Off');
		set(h3, 'HandleVisibility', 'Off');
		%set(h4, 'HandleVisibility', 'Off');
	end
	lgnd{i} = replace(methods{i}, '_', ' ');
end
xlim([0 200])
ylim([0 4e-8])
legend(lgnd)
xlabel('LMEDS iterations')
ylabel('LMEDS error')
grid on;
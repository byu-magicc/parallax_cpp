%% GN vs. LM: What percent are converged after a certian number of iterations?
%video = 'holodeck';
video = 'old';
methods = {'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
lgnd = cell(1, length(methods));
hold on;
convergence_threshold = 1e-10;
for i = 1:length(methods)
	method = methods{i};
	O = read_binary(['../logs/' video '/' method '/optimizer.bin'], [4, 1000]);
	
	% r, delta, lambda, attempts
	pcnt_converged = mean(O(1, :, :) < convergence_threshold, 3);
	%pcnt_converged = mean(O(2, :, :), 3);
	%pcnt_converged = mean(O(3, :, :) < 1e10, 3);
	plot(1:1000, pcnt_converged);
	lgnd{i} = replace(methods{i}, '_', ' ');
end
legend(lgnd)
xlabel('Iterations')
ylabel('Percent converged')
grid on;
xlim([0 200])
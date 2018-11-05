%% Lambda0: What percent are converged after a certian number of iterations?
video = 'holodeck';
test = 'lambda_sweep';
hold on;
convergence_threshold = 1e-8;
powers = [2 3 4 5 6];
power_idxs = 11 - powers;
lgnd = cell(1, length(powers));
for i = 1:length(powers)
	O = read_binary(['../logs/' video '/' test '/lm_eigen' sprintf('%d', power_idxs(i)) '/optimizer.bin'], [4, 20]);
	
	% r, delta, lambda, attempts
	pcnt_converged = mean(O(1, :, :) < convergence_threshold, 3);
	plot(1:20, pcnt_converged);
	lgnd{i} = sprintf('\\lambda_0 = 10^{-%d}', powers(i));
end
legend(lgnd)
xlabel('Levengerg-Marquardt Iterations')
ylabel('Percent converged')
grid on;

%% Plot attempts
hold on;
for i = 1:length(powers)
	O = read_binary(['../logs/' video '/' test '/lm_eigen' sprintf('%d', power_idxs(i)) '/optimizer.bin'], [4, 20]);
	
	% r, delta, lambda, attempts
	attempts = mean(O(2, :, :) + 1, 3);
	plot(1:20, attempts);
	lgnd{i} = sprintf('\\lambda_0 = 10^{-%d}', powers(i));
end
legend(lgnd)
xlabel('Levengerg-Marquardt Iterations')
ylabel('Average attempts per iteration')
grid on;
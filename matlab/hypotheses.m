% Load data
A  = read_binary(['../logs/' method '/5-point_accuracy.bin'],  11);
T  = read_binary(['../logs/' method '/5-point_timing.bin'], [3, 2]);
C_TR = read_binary(['../logs/' method '/5-point_comparison_tr.bin'], [2, 11]);
C_GN = read_binary(['../logs/' method '/5-point_comparison_gn.bin'], [2, 10]);
O = read_binary(['../logs/' method '/optimizer.bin'], [4, 1000]);
C_TR(C_TR == -1) = nan;
C_GN(C_GN == -1) = nan;
A(A == -1) = nan;
TR = reshape(sum(C_TR .^ 2, 1), 11, []);
GN = reshape(sum(C_GN .^ 2, 1), 10, []);

%% did GN/LM converge to a valid solution?
% (note that the sampson err is the residual squared, so 1e-20 makes sense)
convergence_threshold = 1e-20;
valid_GN = A(11, :) < convergence_threshold;
valid_5P = min(A(1:10, :), [], 1) < convergence_threshold;
ratio_valid_GN = mean(valid_GN)
ratio_valid_5P = mean(valid_5P)
%valid = min(GN, [], 1) < 1e-8;

%% Did GN/LM converge to the best 5-point soln'?
% 21%
%[~, idx_GN] = min(GN, [], 1)
%[~, idx_TR] = min(TR, [], 1)
close_threshold = 1e-6;
[dist_GN, closest_GN] = min(GN, [], 1);
[~, closest_TR] = min(TR(1:10, :), [], 1);
converged_to_best_5P_soln = valid_GN & (dist_GN < close_threshold) & (closest_TR == closest_GN);
mean(converged_to_best_5P_soln)

%% Did GN/LM converge to a better (closer to truth) solution than the 5-point algorithm?
% 23%
[~, best_TR] = min(TR, [], 1);
converged_to_different_soln = valid_GN & (best_TR == 11);
mean(converged_to_different_soln)

%% What percentage of cases did LM acheive a best soln?
% 34% without having a prior instead of 25% for five-point.
num_5P = sum(~isnan(TR(1:10, :)), 1);
total_5P = sum(num_5P);

ratio_best_GN = mean(converged_to_best_5P_soln | converged_to_different_soln)
ratio_best_5P = 1 / mean(num_5P(num_5P > 0))

%% What percent found a local min?



%% graph!!!
y = [2 2 3; 2 5 6; 2 8 9; 2 11 12];
bar(y, 'stacked')
legend('a', 'b', 'c')





%% what percent of GN are closer than any E?

%% what are the relative errors?
% Looks like the median LM actually converged to a lower error (2e-33 instead of 2e-32)
% after 1000 iterations.
nanmedian(min(reshape(A(1:10, :), 1, []), [], 1))
nanmedian(A(11, :))




















%%

for i = 1:9
	subplot(3, 3, i)
	plot(squeeze(O(1, :, 2410 + i)))
end

%% Plot a single case
i = 3174;
subplot(411)
plot(squeeze(O(1, :, i)))
xlabel('iter')
ylabel('residual')
subplot(412)
plot(squeeze(O(2, :, i)))
xlabel('iter')
ylabel('norm(\delta)')
subplot(413)
plot(squeeze(O(3, :, i)))
xlabel('iter')
ylabel('\lambda')
subplot(414)
plot(squeeze(O(4, :, i)))
xlabel('iter')
ylabel('attempts')

%%

%converged = A(11, :) < 3e-7;
%mean(converged)
median(A(11, :))

%%
histogram(A(11, :), 100, 'BinLimits', [0, 1e-32])
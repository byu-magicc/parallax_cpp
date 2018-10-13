method = 'test2';
A(A == -1) = nan;
A  = read_binary(['../logs/' method '/5-point_accuracy.bin'],  11);
T  = read_binary(['../logs/' method '/5-point_timing.bin'], [3, 2]);
C_TR = read_binary(['../logs/' method '/5-point_comparison_tr.bin'], [2, 10]);
C_GN = read_binary(['../logs/' method '/5-point_comparison_gn.bin'], [2, 10]);
O = read_binary(['../logs/' method '/optimizer.bin'], [4, 1000]);

%% What percent are converged after a certian number of iterations?


%%


% What do we use as a convergence value?
% Levenburg Marquardt
% The median error of starting guesses is 2.9e-5.
% After 10 iters,   the median error is   4.7e-8.
% After 100 iters,  the median error is   2.9e-8.
% After 1000 iters, the median error is   2.6e-8.
% This suggests a local min.

%%
% r, delta, lambda, attempts
for i = 1:9
	subplot(3, 3, i)
	plot(squeeze(O(1, :, 2410 + i)))
end

%%
i = 3185;
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


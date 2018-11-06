%% Plot against time (not authoritative, but useful)
title_str = 'Essential Matrix Err';
methods = {'gn_eigen_algebraic', 'gn_eigen_single', 'gn_eigen_sampson', ...
           'lm_eigen_algebraic', 'lm_eigen_single', 'lm_eigen_sampson'};
video = 'holodeck';
test = 'scoring_cost';
ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t', 'LMEDS err'};
name = 'accuracy.bin';
plot_comparison(title_str, video, test, methods, ylabels, name);

%% Calculate median error
video = 'holodeck';
test = 'scoring_cost';
lgnd = cell(1, length(methods));
fprintf('%-25s %-15s %-15s %-15s %-15s \n', 'Method', 'Rotation', 'Translation', 'Correct R', 'Correct t')
for i = 1:length(methods)
	filename = ['../logs/' video '/' test '/' methods{i} '/' name];
	A = read_binary(filename, 5);
	fprintf('%-25s %-15d %-15d %-15.3f %-15.3f\n', methods{i}, mean(A(1:4, :), 2).')
end


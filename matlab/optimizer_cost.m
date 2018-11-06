%% Plot against time (not authoritative, but useful)
title_str = 'Essential Matrix Err';
methods = {'lm_eigen_algebraic', 'lm_eigen_single', 'lm_eigen_sampson'};
video = 'holodeck';
test = 'optimizer_cost';
ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t', 'LMedS err'};
name = 'accuracy.bin';
plot_comparison(title_str, video, test, methods, ylabels, name);

%% Calculate median error
video = 'holodeck';
test = 'optimizer_cost';
lgnd = cell(1, length(methods));
fprintf('%-25s %-15s %-15s %-15s \n', 'Method', 'Rotation err', 'Translation err', 'LMedS err')
for i = 1:length(methods)
	filename = ['../logs/' video '/' test '/' methods{i} '/' name];
	A = read_binary(filename, 5);
	fprintf('%-25s %-15e %-15e %-15e \n', methods{i}, mean(A([1 2 5], :), 2).')
end


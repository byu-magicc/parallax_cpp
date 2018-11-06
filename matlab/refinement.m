%% Plot against time (not authoritative, but useful)
title_str = 'Essential Matrix Err';
methods = {'poly_opencv', 'lm_eigen', 'lm_refine', 'lm_refine2'};
video = 'holodeck';
test = 'refinement';
ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t', 'LMedS err'};
name = 'accuracy.bin';
name_r = 'refinement.bin';
plot_comparison(title_str, video, test, methods, ylabels, name);

%% Calculate median error
clc
video = 'holodeck';
test = 'refinement';
lgnd = cell(1, length(methods));
fprintf('%-15s %-15s %-15s %-15s %-15s %-15s \n', 'Method', 'Refine sucess', 'Refine amount', 'Rotation err', 'Translation err', 'LMedS err')
for i = 1:length(methods)
	filename = ['../logs/' video '/' test '/' methods{i} '/' name];
	A = read_binary(filename, 5);
	if contains(methods{i}, 'refine')
		filename = ['../logs/' video '/' test '/' methods{i} '/' name_r];
		B = read_binary(filename, 2);
		sucess = B(2, :) < B(1, :);
		sucess_ratio = mean(sucess);
		sucess_mean = mean(B(1, sucess) - B(2, sucess));
		%sucess_mean = (sum(B(2, sucess)) + sum(B(1, ~sucess))) / length(sucess);
		%sucess_mean = (sum(B(1, :))) / length(sucess);
	else
		sucess_ratio = 0;
		sucess_mean = 0;
	end
	fprintf('%-15s %-15.3f %-15e %-15e %-15e %-15e \n', methods{i}, sucess_ratio, sucess_mean, mean(A([1 2 5], :), 2).')
end


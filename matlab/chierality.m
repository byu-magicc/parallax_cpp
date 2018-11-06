clc
video = 'holodeck';
test = 'chierality';
%methods = {'poly_opencv', 'gn_eigen'};
%methods = {'gn_eigen'};
%methods = {'poly_opencv_no_chierality', 'poly_opencv_chierality', 'poly_opencv_trace', ...
%           'lm_eigen_no_chierality', 'lm_eigen_chierality', 'lm_eigen_trace'};
methods = {'lm_eigen_no_chierality', 'lm_eigen_chierality', 'lm_eigen_trace', 'lm_eigen_normalize'};
lgnd = cell(1, length(methods));
fprintf('%-25s %-15s %-15s %-15s %-15s %-15s %-15s\n', 'Method', 'Rotation', 'Translation', 'Correct R', 'Correct t', 'LMedS error', 'Both correct')
for i = 1:length(methods)
	filename = ['../logs/' video '/' test '/' methods{i} '/accuracy.bin'];
	A = read_binary(filename, 5);
	fprintf('%-25s %-15d %-15d %-15.3f %-15.3f %-15d %-15.3f\n', methods{i}, [mean(A(1:5, :), 2).', mean(A(3, :) & A(4, :))])
	lgnd{i} = replace(methods{i}, '_', ' ');
end


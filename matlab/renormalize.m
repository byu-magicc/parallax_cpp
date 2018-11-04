clc
video = 'holodeck';
test = 'renormalize';
%methods = {'poly_opencv', 'gn_eigen'};
%methods = {'gn_eigen'};
methods = {'lm_eigen_prior1', 'lm_eigen_prior2', ...
           'lm_eigen_prior_recursive1', 'lm_eigen_prior_recursive2', ...
           'lm_eigen_random1', 'lm_eigen_random2', ...
           'lm_eigen_random_recursive1', 'lm_eigen_random_recursive2'};
lgnd = cell(1, length(methods));
fprintf('%-28s %-15s %-15s %-15s %-15s \n', 'Method', 'Rotation', 'Translation', 'Correct R', 'Correct t')
for i = 1:length(methods)
	filename = ['../logs/' video '/' test '/' methods{i} '/accuracy.bin'];
	A = read_binary(filename, 4);
	fprintf('%-28s %-15d %-15d %-15.3f %-15.3f\n', methods{i}, mean(A, 2).')
	lgnd{i} = replace(methods{i}, '_', ' ');
end
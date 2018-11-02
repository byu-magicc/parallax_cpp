video = 'holodeck';
%methods = {'poly_opencv', 'gn_eigen_trace'};
methods = {'poly_opencv_trace', 'poly_opencv_chierality', 'poly_opencv_no_chierality'};
test = 'chierality';
lgnd = cell(1, length(methods));
fprintf('%-20s %-15s %-15s %-15s %-15s \n', 'Method', 'Rotation', 'Translation', 'Rot ratio', 'Tr ratio')
for i = 1:length(methods)
	filename = ['../logs/' video '/' methods{i} '/' test '/accuracy.bin'];
	A = read_binary(filename, 4);
	fprintf('%-20s %-15d %-15d %-15.3f %-15.3f\n', methods{i}, sum(A(3:4, :), 2).', mean(A(3:4, :), 2).')
	lgnd{i} = replace(methods{i}, '_', ' ');
end


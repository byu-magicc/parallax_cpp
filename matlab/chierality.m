video = 'holodeck';
%methods = {'five-point', 'gn', 'lm'};
methods = {'gn', 'chierality'};
lgnd = cell(1, length(methods));
fprintf('%-15s %-15s %-15s %-15s %-15s \n', 'Method', 'Rotation', 'Translation', 'Rot ratio', 'Tr ratio')
for i = 1:length(methods)
	filename = ['../logs/' video '/' methods{i} '/accuracy.bin'];
	A = read_binary(filename, 4);
	fprintf('%-15s %-15d %-15d %-15.3f %-15.3f\n', methods{i}, sum(A(3:4, :), 2).', mean(A(3:4, :), 2).')
	lgnd{i} = replace(methods{i}, '_', ' ');
end


title_str = 'Essential Matrix Err';
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
%methods = {'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
%methods = {'five-point', 'gn', 'lm', 'chierality', 'gn_eigen'};
methods = {'poly_opencv', 'gn_eigen'};
video = 'holodeck';
test = 'standard';

% Rotation and Translation error
ylabels = {{'R err', '(radians)'}, {'t err', '(radians)'}};
offset = -1;
hold on;
lgnd = cell(size(methods, 1), 1);
for i = 1:length(methods)
	filename = ['../logs/' video '/' test '/' methods{i} '/accuracy.bin'];
	A = read_binary(filename, 5);
	for j = 1:2
		subplot(4, 1, j*2 + offset)
		plot(A(j, :));
		hold on;
	end
	lgnd{i} = replace(methods{i}, '_', ' ');
end

% Legend
for j = 1:length(ylabels)
	subplot(4, 1, j*2 + offset)
	grid on;
	ylabel(ylabels{j})
	xlim([0 3600])
	legend(lgnd)
end

% Rotation and Translation magnitude
ylabels = {{'R magnitude', '(radians)'}, {'t magnitude', '(radians)'}};
hold on;
lgnd = cell(size(methods, 1), 1);
filename = '../logs/holodeck/truth_mag/gn_eigen/truth_magnitude.bin';
A = read_binary(filename, 2);
offset = 0;
for j = 1:2
	subplot(4, 1, j*2 + offset)
	plot(A(j, :));
	ylabel(ylabels{j})
	grid on;
	xlim([0 3600])
	legend('truth')
end
subplot(4, 1, 4)
xlabel('Frame')
grid on;





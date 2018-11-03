%% Calculate median error
powers = 1:10;
video = 'holodeck';
test = 'ransac';
metrics = zeros(4, length(powers), 4);
fprintf('%-25s %-20s %-15s %-15s %-15s \n', 'RANSAC threshold', 'Rotation', 'Translation', 'Correct R', 'Correct t')

% GN_RANSAC
for i = 1:length(powers)
	filename = ['../logs/' video '/' test '/gn_ransac' sprintf('%d', powers(i)) '/' name];
	A = read_binary(filename, 4);
	metrics(:, i, 1) = mean(A, 2);
	fprintf('%-25s %-20d %-15d %-15.3f %-15.3f\n', sprintf('GN, threshold=10^-%d', powers(i)), mean(A, 2).')
end

% LM_RANSAC
for i = 1:length(powers)
	filename = ['../logs/' video '/' test '/lm_ransac' sprintf('%d', powers(i)) '/' name];
	A = read_binary(filename, 4);
	metrics(:, i, 2) = mean(A, 2);
	fprintf('%-25s %-20d %-15d %-15.3f %-15.3f\n', sprintf('LM, threshold=10^-%d', powers(i)), mean(A, 2).')
end

% GN_LMEDS
method = 'gn_lmeds';
filename = ['../logs/' video '/' test '/' method '/' name];
A = read_binary(filename, 4);
fprintf('%-25s %-20d %-15d %-15.3f %-15.3f\n', method, mean(A, 2).')
for i = 1:length(powers)
	metrics(:, i, 3) = mean(A, 2);
end

% LM_LMEDS
method = 'lm_lmeds';
filename = ['../logs/' video '/' test '/' method '/' name];
A = read_binary(filename, 4);
fprintf('%-25s %-20d %-15d %-15.3f %-15.3f\n', method, mean(A, 2).')
for i = 1:length(powers)
	metrics(:, i, 4) = mean(A, 2);
end

% Plot results
title_str = 'Average Rotation and Translation Error';
ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t'};
methods = {'gn_ransac', 'lm_ransac', 'gn_lmeds', 'lm_lmeds'};
lgnd = cell(1, length(methods));
for i = 1:length(ylabels)
	subplot(length(ylabels), 1, i)
	for j = 1:length(methods)
		plot(-powers, metrics(i, :, j))
		hold on;
		lgnd{j} = replace(methods{j}, '_', ' ');
	end
	ylabel(ylabels{i})
	grid on;
	if i == 1
		title(title_str);
	end	
end
legend(lgnd)
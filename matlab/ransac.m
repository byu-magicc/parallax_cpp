%% Calculate median error
powers = 1:10;
video = 'holodeck';
test = 'ransac';
metrics = zeros(4, length(powers), 4);
fprintf('%-25s %-20s %-15s %-15s %-15s \n', 'RANSAC threshold', 'Rotation', 'Translation', 'Correct R', 'Correct t')
%methods = {'gn_ransac', 'lm_ransac', 'gn_lmeds', 'lm_lmeds'};
methods = {'lm_ransac', 'lm_lmeds'};
%methods = {'gn_ransac', 'gn_lmeds'};

for ii = 1:length(methods)
	% RANSAC
	method = methods{ii};
	if contains(method, 'ransac')
		for i = 1:length(powers)
			filename = ['../logs/' video '/' test '/' method sprintf('%d', powers(i)) '/' name];
			A = read_binary(filename, 4);
			metrics(:, i, ii) = mean(A, 2);
			fprintf('%-25s %-20d %-15d %-15.3f %-15.3f\n', sprintf('GN, threshold=10^-%d', powers(i)), mean(A, 2).')
		end
	% LMEDS
	elseif contains(method, 'lmeds')
		filename = ['../logs/' video '/' test '/' method '/' name];
		A = read_binary(filename, 4);
		fprintf('%-25s %-20d %-15d %-15.3f %-15.3f\n', method, mean(A, 2).')
		for i = 1:length(powers)
			metrics(:, i, ii) = mean(A, 2);
		end
	end
end

%% Plot results
title_str = 'Average Rotation and Translation Error';
ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t'};
lgnd = cell(1, length(methods));
for i = 1:length(ylabels)
	subplot(length(ylabels), 1, i)
	for j = 1:length(methods)
		semilogx(10.^-powers, metrics(i, :, j))
		hold on;
		lgnd{j} = replace(methods{j}, '_', ' ');
	end
	xlim(10.^(-[powers(end), powers(1)]))
	ylabel(ylabels{i})
	grid on;
	if i == 1
		title(title_str);
	end
	set(gca, 'XminorTick', 'off')
	set(gca, 'XminorGrid', 'off')
end
xlabel('RANSAC threshold')
subplot(length(ylabels), 1, 1)
legend(lgnd)
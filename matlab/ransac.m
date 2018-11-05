%% Calculate median error
%thresholds = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
thresholds = [1.00e-10, 3.16e-10, 1.00e-09, 3.16e-09, 1.00e-08, 3.16e-08, 1.00e-07, 3.16e-07, 1.00e-06, 3.16e-06, 1.00e-05, 3.16e-05, 1.00e-04, 3.16e-04, 1.00e-03, 3.16e-03, 1.00e-02, 3.16e-02, 1.00e-01];
video = 'holodeck';
test_sweep = 'ransac_sweep';
test = 'ransac';
metrics = zeros(5, length(thresholds), 4);
%methods = {'gn_ransac', 'lm_ransac', 'gn_lmeds', 'lm_lmeds'};
methods = {'lm_ransac', 'lm_lmeds'};
%methods = {'gn_ransac', 'gn_lmeds'};

for ii = 1:length(methods)
	% RANSAC
	method = methods{ii};
	if contains(method, 'ransac')
		for i = 1:length(thresholds)
			filename = ['../logs/' video '/' test_sweep '/' method sprintf('%d', i) '/' name];
			A = read_binary(filename, 5);
			metrics(:, i, ii) = mean(A, 2);
		end
	% LMEDS
	elseif contains(method, 'lmeds')
		filename = ['../logs/' video '/' test '/' method '/' name];
		A = read_binary(filename, 5);
		for i = 1:length(thresholds)
			metrics(:, i, ii) = mean(A, 2);
		end
	end
end

%% Plot results
title_str = 'Average Rotation and Translation Error';
%ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t', 'LMeds err'};
ylabels = {'R (radians)', 't (radians)'};
lgnd = cell(1, length(methods));
for i = 1:length(ylabels)
	subplot(length(ylabels), 1, i)
	for j = 1:length(methods)
		semilogx(thresholds, metrics(i, :, j))
		hold on;
		lgnd{j} = replace(methods{j}, '_', ' ');
	end
	xlim([thresholds(1), thresholds(end)])
	ylabel(ylabels{i})
	grid on;
	if i == 1
		title(title_str);
	end
	set(gca, 'XminorTick', 'off')
	set(gca, 'XminorGrid', 'off')
	xlim(10.^[-7 -2])
end
xlabel('RANSAC threshold')
subplot(length(ylabels), 1, 1)
legend(lgnd)
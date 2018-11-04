%n_subsets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300];
n_subsets = [50, 100, 150, 200, 250, 300];
%n_subsets = n_subsets(1:20);

%% Calculate median error
video = 'holodeck';
test_sweep = 'subsets_sweep';
test = 'subsets';
metrics = zeros(2, length(n_subsets), 4);
%methods = {'lm_eigen', 'lm_eigen_prior', 'lm_eigen_prior_max', 'poly_opencv'};
%methods = {'lm_eigen', 'lm_eigen_prior', 'poly_opencv'};
%methods = {'lm_eigen_random_recursive', 'poly_opencv'};
methods = {'lm_eigen_prior_recursive', 'lm_eigen_random_recursive', 'poly_opencv'};

for ii = 1:length(methods)
	% RANSAC
	method = methods{ii};
	if contains(method, 'lm_eigen')
		for i = 1:length(n_subsets)
			filename = ['../logs/' video '/' test_sweep '/' method sprintf('%d', i) '/' name];
			A = read_binary(filename, 4);
			metrics(:, i, ii) = mean(A(1:2, :), 2);
			%metrics(:, i, ii) = quantile(A(1:2, :), 0.75, 2);
		end
	% LMEDS
	elseif contains(method, 'poly_opencv')
		filename = ['../logs/' video '/' test '/' method '/' name];
		A = read_binary(filename, 4);
		for i = 1:length(n_subsets)
			metrics(:, i, ii) = mean(A(1:2, :), 2);
			%metrics(:, i, ii) = quantile(A(1:2, :), 0.75, 2);
		end
	end
end

%% Plot results
title_str = 'Average Rotation and Translation Error';
%ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t'};
ylabels = {'R (radians)', 't (radians)'};
lgnd = cell(1, length(methods));
for i = 1:length(ylabels)
	subplot(length(ylabels), 1, i)
	for j = 1:length(methods)
		plot(n_subsets, metrics(i, :, j))
		hold on;
		lgnd{j} = replace(methods{j}, '_', ' ');
	end
	ylabel(ylabels{i})
	grid on;
	if i == 1
		title(title_str);
	end
	set(gca, 'XminorTick', 'off')
	set(gca, 'XminorGrid', 'off')
end
xlabel('Number of subsets')
subplot(length(ylabels), 1, 1)
legend(lgnd)
%% Calculate median error
optimizer_seed_noise = [0, -1e-10, -0.001, -0.01, -0.1];
video = 'holodeck';
test_sweep = 'optimizer_seed_noise_sweep';
test = 'subsets';
%methods = {'lm_eigen_prior', 'poly_opencv'};
%methods = {'lm_eigen_prior', 'lm_eigen_prior_recursive', 'lm_eigen_random', 'lm_eigen_random_recursive', 'poly_opencv'};
methods = {'lm_eigen_prior_recursive', 'lm_eigen_random_recursive', 'poly_opencv'};

metrics = zeros(3, length(optimizer_seed_noise), length(methods));
t = 500;
err = zeros(length(methods), 3599, length(optimizer_seed_noise));
for ii = 1:length(methods)
	% RANSAC
	method = methods{ii};
	if contains(method, 'lm_eigen') || contains(method, 'gn_eigen')
		for i = 1:length(optimizer_seed_noise)
			filename = ['../logs/' video '/' test_sweep '/' method sprintf('%d', i) '/' name];
			A = read_binary(filename, 5);
			metrics(:, i, ii) = mean(A([1 2 5], 1:t), 2);
			%metrics(:, i, ii) = quantile(A(1:2, :), 0.75, 2);
			err(ii, :, i) = A(5, :);
		end
	% LMEDS
	elseif contains(method, 'poly_opencv')
		filename = ['../logs/' video '/' test '/' method '/' name];
		A = read_binary(filename, 5);
		for i = 1:length(optimizer_seed_noise)
			metrics(:, i, ii) = mean(A([1 2 5], 1:t), 2);
			%metrics(:, i, ii) = quantile(A(1:2, :), 0.75, 2);
			err(ii, :, i) = A(5, :);
		end
	end
end

%% Plot results
title_str = 'Average Rotation and Translation Error';
%ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t'};
ylabels = {'R (radians)', 't (radians)', 'Sampson err'};
lgnd = cell(1, length(methods));
for i = 1:length(ylabels)
	subplot(length(ylabels), 1, i)
	for j = 1:length(methods)
		xvals = 1:length(optimizer_seed_noise);
		%xvals = optimizer_seed_noise;
		plot(xvals, metrics(i, :, j))
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

%% histogram / kernel density
subset_idx = 1;
filename = ['../logs/' video '/' test '/' method '/' name];
for i = 1:length(methods)
	max_x = 1e-7;
	bandwidth = max_x * 3e5;
	x = linspace(0, max_x, 1000);
	[Y, x] = ksdensity(err(i, :, subset_idx), x, 'Support', 'positive', 'Bandwidth', bandwidth);
	plot(x, Y);
	%semilogy(x, Y + 1e-100);
	hold on;
	%if length(methods) == 1
	%	histogram(err(i, :), 100, 'Normalization', 'pdf');
	%end
end
legend(lgnd);
xlabel('Err')
ylabel('PDF')

%% Plot against time
filename = '../logs/holodeck/truth_mag/gn_eigen/truth_magnitude.bin';
A = read_binary(filename, 2);
subplot(511)
plot(A(1, :))
grid on;
ylabel('norm(R)')
subplot(512)
plot(A(2, :))
grid on;
ylabel('norm(t)')
subplot(513)
plot(err(5, :, 1));
grid on;
ylabel('sampson err (OpenCV)')
xlabel('Frame')
subplot(514)
plot(err(2, :, 1));
grid on;
ylabel('sampson err (PriorRecursive)')
xlabel('Frame')
subplot(515)
plot(err(4, :, 1));
grid on;
ylabel('sampson err (RandomRecursive)')
xlabel('Frame')
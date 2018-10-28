% Some of the optimizations are converging.
figure
n_pts = 50;
pts_world = read_binary('../logs/test/unit_test_pts_world.bin', [3, n_pts]);
pts_camera_nip = read_binary('../logs/test/unit_test_pts_camera.bin', [2, n_pts, 2]);
optim = read_binary(['../logs/test/optimizer.bin'], [4, 20]);
pts1 = pts_camera_nip(:, :, 1);
pts2 = pts_camera_nip(:, :, 2);
subplot(221);
plot3(pts_world(1, :), pts_world(2, :), pts_world(3, :), '*')
grid on;
axis equal;
subplot(223);
plot(pts1(1, :), pts1(2, :), 'k.')
grid on
subplot(224);
plot(pts2(1, :), pts2(2, :), 'k.')
grid on

subplot(222);
hold on;
for i = 1:6
	% r, delta, lambda, attempts
	%plot(1:20, optim(1, :, i));
	plot(1:20, optim(2, :, i));
end

%%
figure
optimizers = {'GaussNewton', 'LevenbergMarquardt'};
cost_functions = {'Algebraic', 'Single', 'Sampson'};
k = 0;
for i = 1:2
	for j = 1:3
		k = k + 1;
		subplot(2, 3, k)
		
		% r, delta, lambda, attempts
		plot(1:20, optim(1, :, k));
		%plot(1:20, optim(2, :, i));
		title([optimizers{i} ', ' cost_functions{j}]);
	end
end

%%
figure
details = {'residual', 'delta', 'lambda', 'attempts'};
kk = [1 2 3; 4 5 6];
i = 2;
j = 3;
k = kk(i, j);
for d = 1:4
	subplot(4, 1, d)
	plot(1:20, optim(d, :, k));
	if d == 1
		title([optimizers{i} ', ' cost_functions{j}]);
	end
	ylabel(details{d});	
end

%% But what's the same and what's changed?
figure
log_sizes = {[3, 3, 2], [3, 3, 2], [3, 1, 2], [3, 3, 2], [3, 3, 2], [3, 3, 2], [3, 1, 2]};
log_names = {'E', 'R', 't', 'TR', 'rot.R', 'vec.R', 'vec.v'};
n_logs = length(log_sizes);
assert(length(log_names) == n_logs);
for i = 1:n_logs
	log_filename = sprintf('../logs/test/test%d.bin', i);
	log = read_binary(log_filename, log_sizes{i});
	diff = log(:, :, 1, :) - log(:, :, 2, :);
	diff_norm = sum(sum(diff.^2, 1), 2);
	subplot(n_logs, 1, i);
	plot(squeeze(diff_norm(1:20)));
	ylabel(log_names{i});
	if i == 1
		title('Difference before and after boxplus')
	end
end





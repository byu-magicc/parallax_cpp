%%
file = fopen('../logs/log_test.bin', 'r');
number = fread(file, 'double');
%number = reshape(number, 2, []);
%disp(number)
plot(number)

%% Accuracy
figure(1)
clf
file = fopen('../logs/log_test.bin', 'r');
number = fread(file, 'double');
fclose(file);

ylabels = {'E', 'R (radians)', 't (radians)'};
number = reshape(number, length(ylabels), []);
for i = 1:length(ylabels)
	subplot(length(ylabels), 1, i)
	plot(number(i, :))
	ylabel(ylabels{i})
	grid on;
	if i == 1
		title('Essential Matrix Err');
	end
end

%% Timing
file = fopen('../logs/time_E.bin', 'r');
number = fread(file, 'double');
fclose(file);

timeCats = {'Hypothesis Generation', 'Hypothesis Scoring', 'Total'};
number = reshape(number, length(timeCats), []);
for i = 1:length(timeCats)
	subplot(length(timeCats), 1, i)
	plot(number(i, :))
	ylabel(timeCats{i})
	grid on;
end

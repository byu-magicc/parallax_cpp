% Read hypothesis categories
video = 'holodeck';
methods = {'five-point', 'gn', 'lm'};
bar_lgnd = cell(1, length(methods));
cat_count = zeros(length(methods), 3);
for i = 1:length(methods)
	if ~contains(methods{i}, 'five')
		[cat_count(i, :), cat_count_5P] = compare_to_5P(video, methods{i});
	end
end
for i = 1:length(methods)
	if contains(methods{i}, 'five')
		cat_count(i, :) = cat_count_5P;
	end
	bar_lgnd{i} = replace(methods{i}, '_', ' ');
end

% Read timing
t_gen = zeros(length(methods), 1);
t_score = zeros(length(methods), 1);
for i = 1:length(methods)
	method = methods{i};
	t = read_binary(['../logs/' video '/' method '/timing.bin'], 3);
	t_gen(i) = mean(t(1, :));
	t_score(i) = mean(t(2, :));
end

%% verbose timing
gen_lgnd = {'Setup', 'SVD', 'Coeffs1', 'Coeffs2', 'Coeffs3', 'SolvePoly', 'ConstructE', ...
	'MakeJ', 'SolveMatrix', 'ManifoldUpdate', 'CalcResidual', 'Total'};
gen_time = zeros(length(methods), length(gen_lgnd));
for i = 1:length(methods)
	method = methods{i};
	t = read_binary(['../logs/' video '/' method '/timing_verbose.bin'], length(gen_lgnd));
	gen_time(i, :) = mean(t, 2).';
end

%% Hypothesis category bar graph
figure
red = [1, 0, 0];
green = [0, 1, 0];
blue = [0, 0, 1];
cat_clrs = [green; blue; red];
cat_lgnd = {'closest to truth', 'valid', 'invalid'};
b = bar(cat_count, 'stacked', 'FaceColor', 'flat');
for i = 1:size(cat_clrs, 1)
	b(i).CData = repmat(cat_clrs(i, :), length(methods), 1);
end
legend(cat_lgnd)
ylabel('number of hypotheses');
xticklabels(bar_lgnd);
title('Hypothesis Count')

%% Hypo_gen timing bar graph
figure
b = bar(gen_time(:, 1:end-1) / 100, 'stacked', 'FaceColor', 'flat');
n_groups = length(methods);
legend(gen_lgnd{1:end-1})
ylabel('time per subset (ms)');
xticklabels(bar_lgnd);
title('Hypothesis Generation Time')

%% Hypo_score timing bar graph
figure
cat_score_time = cat_count ./ sum(cat_count, 2) .* t_score;
b = bar(cat_score_time, 'stacked', 'FaceColor', 'flat');
for i = 1:size(cat_clrs, 1)
	b(i).CData = repmat(cat_clrs(i, :), length(methods), 1);
end
legend(cat_lgnd)
ylabel('Hypothesis scoring time (ms)');
xticklabels(bar_lgnd);
title('Hypothesis Scoring Time')
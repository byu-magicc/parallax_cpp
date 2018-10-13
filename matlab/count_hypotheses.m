% Read hypothesis categories
methods = {'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
bar_lgnd = cell(1, length(methods) + 1);
cat_GN = zeros(length(methods), 3);
for i = 1:length(methods)
	[cat_GN(i, :), cat_5P] = compare_to_5P(methods{i});
	bar_lgnd{i + 1} = replace(methods{i}, '_', ' ');
end
cat_count = [cat_5P; cat_GN];
bar_lgnd{1} = 'five-point';

% Read timing
t_gen = zeros(length(methods) + 1, 1);
t_score = zeros(length(methods) + 1, 1);
for i = 1:length(methods)
	method = methods{i};
	t = read_binary(['../logs/' method '/timing.bin'], 3);
	t_gen(i + 1) = mean(t(1, :));
	t_score(i + 1) = mean(t(2, :));
end

%% Hypothesis category bar graph
red = [1, 0, 0];
green = [0, 1, 0];
blue = [0, 0, 1];
cat_clrs = [green; blue; red];
cat_lgnd = {'closest to truth', 'valid', 'invalid'};
b = bar(cat_count, 'stacked', 'FaceColor', 'flat');
n_groups = length(methods) + 1;
for i = 1:size(cat_clrs, 1)
	b(i).CData = repmat(cat_clrs(i, :), n_groups, 1);
end
legend(cat_lgnd)
ylabel('number of hypotheses');
xticklabels(bar_lgnd);

%% Hypo_gen timing bar graph

%% Hypo_score timing bar graph
cat_score_time = cat_count ./ sum(cat_count, 2) .* t_score;
b = bar(cat_score_time, 'stacked', 'FaceColor', 'flat');
n_groups = length(methods) + 1;
for i = 1:size(cat_clrs, 1)
	b(i).CData = repmat(cat_clrs(i, :), n_groups, 1);
end
legend(cat_lgnd)
ylabel('Hypothesis scoring time (ms)');
xticklabels(bar_lgnd);
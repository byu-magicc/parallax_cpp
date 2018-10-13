% Analyze data
methods = {'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
bar_lgnd = cell(1, length(methods) + 1);
cat_GN = zeros(length(methods), 3);
for i = 1:length(methods)
	[cat_GN(i, :), cat_5P] = compare_to_5P(methods{i});
	bar_lgnd{i + 1} = replace(methods{i}, '_', ' ');
end
bar_lgnd{1} = 'five-point';

% Show results
y = [cat_5P; cat_GN];
red = [1, 0, 0];
green = [0, 1, 0];
blue = [0, 0, 1];
b = bar(y, 'stacked', 'FaceColor', 'flat');
n_groups = length(methods) + 1;
b(1).CData = repmat(green, n_groups, 1);
b(2).CData = repmat(blue, n_groups, 1);
b(3).CData = repmat(red, n_groups, 1);
legend('closest to truth', 'valid', 'invalid')
ylabel('number of hypotheses');
xticklabels(bar_lgnd);
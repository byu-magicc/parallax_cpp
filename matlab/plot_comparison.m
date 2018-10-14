function plot_comparison(title_str, dataset, methods, ylabels, name)
	lgnd = cell(1, length(methods));
	figure(1)
	clf
	for i = 1:length(methods)
		filename = ['../logs/' dataset '/' methods{i} '/' name];
		file = fopen(filename, 'r');
		val = fread(file, 'double');
		fclose(file);
		
		% Hack to allow compatibility with three error rows
		if length(ylabels) == 2 && mod(length(val), 2) ~= 0
			val = reshape(val, 3, []);
			val = val(2:3, :);
		end
		
		% reshape
		val = reshape(val, length(ylabels), []);
		for j = 1:length(ylabels)
			subplot(length(ylabels), 1, j)
			plot(val(j, :))
			ylabel(ylabels{j})
			grid on;
			if i == 1
				hold on;
				if j == 1
					title(title_str);
				end
			end
		end
		lgnd{i} = replace(methods{i}, '_', ' ');
	end
	legend(lgnd)
end
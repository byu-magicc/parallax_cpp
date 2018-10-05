function plot_comparison(title_str, methods, ylabels, filename)
	lgnd = cell(1, length(methods));
	figure(1)
	clf
	for i = 1:length(methods)
		file = fopen(['../logs/' methods{i} '/' filename], 'r');
		number = fread(file, 'double');
		fclose(file);
		number = reshape(number, length(ylabels), []);
		for j = 1:length(ylabels)
			subplot(length(ylabels), 1, j)
			plot(number(j, :))
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
function plot_comparison(title_str, methods, ylabels, name)
	lgnd = cell(1, length(methods));
	figure(1)
	clf
	for i = 1:length(methods)
		filename = ['../logs/' methods{i} '/' name];
		file = fopen(filename, 'r');
		number = fread(file, 'double');
		fclose(file);
		
		% Hack to allow compatibility with three error rows
		if length(ylabels) == 2 && mod(length(number), 2) ~= 0
			number = reshape(number, 3, []);
			number = number(2:3, :);
		end
		
		% reshape
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
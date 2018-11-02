% Plots the consensus error (Sampson) vs. the number of iterations of the algorithm.
video = 'holodeck';
methods = {'gn_eigen', 'poly_opencv'};
lgnd = cell(size(methods, 1), 1);
test = 'consensus';
hold on;
for i=1:length(methods) 
	err = read_binary(['../logs/' video '/' methods{i} '/' test '/consensus.bin'], 1000);
	plot(median(err, 2))
	lgnd{i} = replace(methods{i}, '_', ' ');
end
legend(lgnd)
xlabel('LMEDS iterations')
ylabel('LMEDS error')
grid on;
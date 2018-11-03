% Look at estimate
video = 'holodeck';
method = 'gn_eigen';
test = 'standard';
lgnd = cell(1, length(methods));
filename_est = ['../logs/' video '/' method '/' test '/estimate.bin'];
filename_truth = ['../logs/' video '/' method '/' test '/truth.bin'];
RT_est = read_binary(filename_est, [4, 4]);
RT_truth = read_binary(filename_truth, [4, 4]);
title_str = 'Timing';
video = 'holodeck';
test = 'standard';
methods = {'poly_opencv', 'gn_eigen', 'lm_eigen'};
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
%methods = {'five-point', 'gn', 'lm', 'gn_eigen'};
ylabels = {'Hypothesis Generation', 'Hypothesis Scoring', 'Total'};
filename = 'timing.bin';
plot_comparison(title_str, video, test, methods, ylabels, filename)
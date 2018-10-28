title_str = 'Timing';
video = 'holodeck';
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
methods = {'five-point', 'gn', 'lm', 'gn_eigen'};	
ylabels = {'Hypothesis Generation', 'Hypothesis Scoring', 'Total'};
filename = 'timing.bin';
plot_comparison(title_str, video, methods, ylabels, filename)
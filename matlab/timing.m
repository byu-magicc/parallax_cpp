title_str = 'Timing';
methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
ylabels = {'Hypothesis Generation', 'Hypothesis Scoring', 'Total'};
filename = 'timing.bin';
plot_comparison(title_str, methods, ylabels, filename)
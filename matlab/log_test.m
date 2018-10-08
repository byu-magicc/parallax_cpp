%% Accuracy
title_str = 'Essential Matrix Err';
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'test'};
methods = {'gnsac_ptr_opencv', 'test'};
ylabels = {'R (radians)', 't (radians)'};
filename = 'accuracy.bin';
plot_comparison(title_str, methods, ylabels, filename);

%% Timing
title_str = 'Timing';
methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'test'};
ylabels = {'Hypothesis Generation', 'Hypothesis Scoring', 'Total'};
filename = 'timing.bin';
plot_comparison(title_str, methods, ylabels, filename)
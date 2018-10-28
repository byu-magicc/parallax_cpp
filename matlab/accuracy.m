title_str = 'Essential Matrix Err';
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
%methods = {'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
video = 'holodeck';
methods = {'five-point', 'gn', 'lm', 'chierality', 'gn_eigen'};
ylabels = {'R (radians)', 't (radians)', 'Correct R', 'Correct t'};
filename = 'accuracy.bin';
plot_comparison(title_str, video, methods, ylabels, filename);
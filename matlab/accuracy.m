title_str = 'Essential Matrix Err';
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
%methods = {'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
video = 'holodeck';
methods = {'five-point', 'gn', 'lm'};
ylabels = {'R (radians)', 't (radians)'};
filename = 'accuracy.bin';
plot_comparison(title_str, video, methods, ylabels, filename);
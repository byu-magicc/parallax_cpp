title_str = 'Essential Matrix Err';
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
%methods = {'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
dataset = 'holodeck';
methods = {'five-point'};
ylabels = {'R (radians)', 't (radians)'};
filename = 'accuracy.bin';
plot_comparison(title_str, dataset, methods, ylabels, filename);
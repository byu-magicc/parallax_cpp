title_str = 'Essential Matrix Err';
methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen'};
ylabels = {'E', 'R (radians)', 't (radians)'};
filename = 'accuracy.bin';
plot_comparison(title_str, methods, ylabels, filename);
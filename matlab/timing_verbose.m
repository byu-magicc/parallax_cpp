title_str = 'Timing';
dataset = 'holodeck';
%methods = {'gnsac_ptr_opencv', 'gnsac_ptr_eigen', 'gnsac_ptr_eigen_gn', 'gnsac_ptr_eigen_lm'};
methods = {'five-point2'}; %, 'gn', 'lm'};
ylabels = {'Setup', 'SVD', 'Coeffs1', 'Coeffs2', 'Coeffs3', 'SolvePoly', 'ConstructE', 'Total'};
filename = 'timing_verbose.bin';
plot_comparison(title_str, dataset, methods, ylabels, filename)
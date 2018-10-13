function val = read_binary(filename, shape_minor)
	file = fopen(filename, 'r');
	val = fread(file, 'double');
	n = prod(shape_minor);
	if mod(length(val), n) ~= 0
		error('Error reshaping data from %s: %d is not divisible by %d\n', filename, length(val), n);
	end
	m = length(val) / n;
	val = reshape(val, [shape_minor m]);
	fclose(file);
end
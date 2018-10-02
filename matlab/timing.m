file = fopen('../logs/timing.bin', 'r');
number = fread(file, 'double');
number = reshape(number, 2, []);
disp(number)
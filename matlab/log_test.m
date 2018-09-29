file = fopen('../logs/log_test.bin', 'r');
number = fread(file, 'double');
%number = reshape(number, 2, []);
%disp(number)
plot(number)
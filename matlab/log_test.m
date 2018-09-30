%%
file = fopen('../logs/log_test.bin', 'r');
number = fread(file, 'double');
%number = reshape(number, 2, []);
%disp(number)
plot(number)

%% Accuracy
file = fopen('../logs/log_test.bin', 'r');
number = fread(file, 'double');
fclose(file);

number = reshape(number, 3, []);
subplot(311)
plot(number(1, :))
subplot(312)
plot(number(2, :))
subplot(313)
plot(number(3, :))

%% Timing
file = fopen('../logs/time_E.bin', 'r');
number = fread(file, 'double');
fclose(file);
plot(number);
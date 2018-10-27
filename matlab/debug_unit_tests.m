n_pts = 50;
pts_world = read_binary('../logs/pts_world.bin', [3, n_pts]);
pts_camera_nip = read_binary('../logs/pts_camera.bin', [2, n_pts, 2]);
pts1 = pts_camera_nip(:, :, 1);
pts2 = pts_camera_nip(:, :, 2);
subplot(221);
plot3(pts_world(1, :), pts_world(2, :), pts_world(3, :), '*')
grid on;
axis equal;
subplot(223);
plot(pts1(1, :), pts1(2, :), 'k.')
grid on
subplot(224);
plot(pts2(1, :), pts2(2, :), 'k.')
grid on


t1 = [-6.69451e-05, 3.32804e-05, -1];
t2 = [0.894057, 0.393283, 0.214452];
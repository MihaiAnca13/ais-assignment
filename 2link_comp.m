% extract validation data as set of XY
XY = test_data1(:,1:2);
THETA1P = evalfis(XY,anfis1); % theta1 predicted by anfis1
THETA2P = evalfis(XY,anfis2); % theta2 predicted by anfis2

NN_THETA = MLP_2joints({XY(:,1)';XY(:,2)'});
NN_THETA = NN_THETA{1}';
NN_THETA1P = NN_THETA(:,1);
NN_THETA2P = NN_THETA(:,2);

%% calculate overall RMSE, MEAN, MEADIAN and MAXIMUM on angles
angle_errors1 = THETA1P-test_data1(:,3);
angle_errors2 = THETA2P-test_data2(:,3);

NN_angle_errors1 = NN_THETA1P-test_data1(:,3);
NN_angle_errors2 = NN_THETA2P-test_data2(:,3);

%RMSE
RMSE1 = sqrt(mean((angle_errors1).^2));
RMSE2 = sqrt(mean((angle_errors2).^2));

NN_RMSE1 = sqrt(mean(NN_angle_errors1.^2));
NN_RMSE2 = sqrt(mean(NN_angle_errors2.^2));

%MEAN
MEAN1 = mean(abs(angle_errors1-mean(angle_errors1)));
MEAN2 = mean(abs(angle_errors2-mean(angle_errors2)));

NN_MEAN1 = mean(abs(NN_angle_errors1-mean(NN_angle_errors1)));
NN_MEAN2 = mean(abs(NN_angle_errors2-mean(NN_angle_errors2)));

%MEDIAN
MEDIAN1 = median(abs(angle_errors1-median(angle_errors1)));
MEDIAN2 = median(abs(angle_errors2-median(angle_errors2)));

NN_MEDIAN1 = median(abs(NN_angle_errors1-median(NN_angle_errors1)));
NN_MEDIAN2 = median(abs(NN_angle_errors2-median(NN_angle_errors2)));

%MAXIMUM
MAXIMUM1 = max(abs(angle_errors1));
MAXIMUM2 = max(abs(angle_errors2));

NN_MAXIMUM1 = max(abs(NN_angle_errors1));
NN_MAXIMUM2 = max(abs(NN_angle_errors2));

["Calculation","Anfis","MLP","Delta";
 "RMSE1",RMSE1,NN_RMSE1,RMSE1-NN_RMSE1;
 "RMSE2",RMSE2,NN_RMSE2,RMSE2-NN_RMSE2;
 "MEAN1",MEAN1,NN_MEAN1,MEAN1-NN_MEAN1;
 "MEAN2",MEAN2,NN_MEAN2,MEAN2-NN_MEAN2;
 "MEDIAN1",MEDIAN1,NN_MEDIAN1,MEDIAN1-NN_MEDIAN1;
 "MEDIAN2",MEDIAN2,NN_MEDIAN2,MEDIAN2-NN_MEDIAN2;
 "MAXIMUM1",MAXIMUM1,NN_MAXIMUM1,MAXIMUM1-NN_MAXIMUM1;
 "MAXIMUM2",MAXIMUM2,NN_MAXIMUM2,MAXIMUM2-NN_MAXIMUM2;
 ]
%% calculate overall RMSE, MEAN, MEADIAN and MAXIMUM on position
Xp = l1 * cos(THETA1P) + l2 * cos(THETA1P + THETA2P);
Yp = l1 * sin(THETA1P) + l2 * sin(THETA1P + THETA2P);

NN_Xp = l1 * cos(NN_THETA1P) + l2 * cos(NN_THETA1P + NN_THETA2P);
NN_Yp = l1 * sin(NN_THETA1P) + l2 * sin(NN_THETA1P + NN_THETA2P);

% euclidian distance
ed = sqrt((Xp-XY(:,1)).^2 + (Yp-XY(:,2)).^2);
NN_ed = sqrt((NN_Xp-XY(:,1)).^2 + (NN_Yp-XY(:,2)).^2);

% RMSE based on euclidian distance
RMSE = sqrt(mean(ed.^2));
NN_RMSE = sqrt(mean(NN_ed.^2));

%MEAN
MEAN = mean(abs(ed-mean(ed)));
NN_MEAN = mean(abs(NN_ed-mean(NN_ed)));

%MEDIAN
MEDIAN = median(abs(ed-median(ed)));
NN_MEDIAN = median(abs(NN_ed-median(NN_ed)));

%MAXIMUM
MAXIMUM = max(abs(ed));
NN_MAXIMUM = max(abs(NN_ed));

["Calculation","Anfis","MLP","Delta";
 "RMSE",RMSE,NN_RMSE,RMSE-NN_RMSE;
 "MEAN",MEAN,NN_MEAN,MEAN-NN_MEAN;
 "MEDIAN",MEDIAN,NN_MEDIAN,MEDIAN-NN_MEDIAN;
 "MAXIMUM",MAXIMUM,NN_MAXIMUM,MAXIMUM-NN_MAXIMUM
 ]

%% display quivers
err_X = Xp-XY(:,1);
err_Y = Yp-XY(:,2);

NN_err_X = NN_Xp-XY(:,1);
NN_err_Y = NN_Yp-XY(:,2);

% all_v = [err_X(:);err_Y(:);NN_err_X;NN_err_Y];
% a = min(all_v);
% b = max(all_v);
% 
% err_X = map(err_X(:), a, b);
% err_Y = map(err_Y(:), a, b);
% NN_err_X = map(NN_err_X(:), a, b);
% NN_err_Y = map(NN_err_Y(:), a, b);

figure
subplot(3,1,1)
quiver(XY(:,1),XY(:,2),err_X(:),err_Y(:))
title("Anfis");
axis([2 18 0 16]); 

subplot(3,1,2)
quiver(XY(:,1),XY(:,2),NN_err_X(:),NN_err_Y(:))
title("MLP");
axis([2 18 0 16]); 

subplot(3,1,3);
plot(XY(:,1),XY(:,2),'.'); 
title("Target");
axis([2 18 0 16]); 

%% 3d plot of errors
anfis_3 = sqrt(err_X.^2+err_Y.^2);
nn_3 = sqrt(NN_err_X.^2+NN_err_Y.^2);

CircleSize=100;
figure;
h2=scatter3(XY(:,1),XY(:,2),anfis_3,CircleSize,anfis_3,'s','filled');
xlabel('X axis');
ylabel('Y axis');
zlabel('Error');
title('Anfis');
colorbar;
figure;
h2=scatter3(XY(:,1),XY(:,2),nn_3,CircleSize,nn_3,'s','filled');
xlabel('X axis');
ylabel('Y axis');
zlabel('Error');
colorbar;
title('MLP');
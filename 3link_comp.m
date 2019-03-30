% extract validation data as set of XY
XY = val_data1(:,1:2);
THETA1P = evalfis(XY,anfis1); % theta1 predicted by anfis1
THETA2P = evalfis(XY,anfis2); % theta2 predicted by anfis2
THETA3P = evalfis(XY,anfis3); % theta3 predicted by anfis3


SUM_THETA = val_data1(:,3)+val_data2(:,3)+val_data3(:,3);
NN_THETA = MLP_3joints({XY(:,1)';XY(:,2)';SUM_THETA(:)'});
NN_THETA = NN_THETA{1}';
NN_THETA1P = NN_THETA(:,1);
NN_THETA2P = NN_THETA(:,2);
NN_THETA3P = NN_THETA(:,3);

%% calculate overall RMSE, MEAN, MEDIAN and MAXIMUM on angles
angle_errors1 = THETA1P-val_data1(:,3);
angle_errors2 = THETA2P-val_data2(:,3);
angle_errors3 = THETA3P-val_data3(:,3);

NN_angle_errors1 = NN_THETA1P-val_data1(:,3);
NN_angle_errors2 = NN_THETA2P-val_data2(:,3);
NN_angle_errors3 = NN_THETA3P-val_data3(:,3);

%RMSE
RMSE1 = sqrt(mean((angle_errors1).^2));
RMSE2 = sqrt(mean((angle_errors2).^2));
RMSE3 = sqrt(mean((angle_errors3).^2));

NN_RMSE1 = sqrt(mean(NN_angle_errors1.^2));
NN_RMSE2 = sqrt(mean(NN_angle_errors2.^2));
NN_RMSE3 = sqrt(mean(NN_angle_errors3.^2));

%MEAN
MEAN1 = mean(abs(angle_errors1-mean(angle_errors1)));
MEAN2 = mean(abs(angle_errors2-mean(angle_errors2)));
MEAN3 = mean(abs(angle_errors3-mean(angle_errors3)));

NN_MEAN1 = mean(abs(NN_angle_errors1-mean(NN_angle_errors1)));
NN_MEAN2 = mean(abs(NN_angle_errors2-mean(NN_angle_errors2)));
NN_MEAN3 = mean(abs(NN_angle_errors3-mean(NN_angle_errors3)));

%MEDIAN
MEDIAN1 = median(abs(angle_errors1-median(angle_errors1)));
MEDIAN2 = median(abs(angle_errors2-median(angle_errors2)));
MEDIAN3 = median(abs(angle_errors3-median(angle_errors3)));

NN_MEDIAN1 = median(abs(NN_angle_errors1-median(NN_angle_errors1)));
NN_MEDIAN2 = median(abs(NN_angle_errors2-median(NN_angle_errors2)));
NN_MEDIAN3 = median(abs(NN_angle_errors3-median(NN_angle_errors3)));

%MAXIMUM
MAXIMUM1 = max(abs(angle_errors1));
MAXIMUM2 = max(abs(angle_errors2));
MAXIMUM3 = max(abs(angle_errors3));

NN_MAXIMUM1 = max(abs(NN_angle_errors1));
NN_MAXIMUM2 = max(abs(NN_angle_errors2));
NN_MAXIMUM3 = max(abs(NN_angle_errors3));

["Calculation","Anfis","MLP","Delta";
 "RMSE1",RMSE1,NN_RMSE1,RMSE1-NN_RMSE1;
 "RMSE2",RMSE2,NN_RMSE2,RMSE2-NN_RMSE2;
 "RMSE3",RMSE3,NN_RMSE3,RMSE3-NN_RMSE3;
 "MEAN1",MEAN1,NN_MEAN1,MEAN1-NN_MEAN1;
 "MEAN2",MEAN2,NN_MEAN2,MEAN2-NN_MEAN2;
 "MEAN3",MEAN3,NN_MEAN3,MEAN3-NN_MEAN3;
 "MEDIAN1",MEDIAN1,NN_MEDIAN1,MEDIAN1-NN_MEDIAN1;
 "MEDIAN2",MEDIAN2,NN_MEDIAN2,MEDIAN2-NN_MEDIAN2;
 "MEDIAN3",MEDIAN3,NN_MEDIAN3,MEDIAN3-NN_MEDIAN3;
 "MAXIMUM1",MAXIMUM1,NN_MAXIMUM1,MAXIMUM1-NN_MAXIMUM1;
 "MAXIMUM2",MAXIMUM2,NN_MAXIMUM2,MAXIMUM2-NN_MAXIMUM2;
 "MAXIMUM3",MAXIMUM3,NN_MAXIMUM3,MAXIMUM3-NN_MAXIMUM3;
 ]
%% calculate overall RMSE, MEAN, MEADIAN and MAXIMUM on position
Xp = l1 * cos(THETA1P) + l2 * cos(THETA1P + THETA2P) + l3 * cos(THETA1P + THETA2P + THETA3P); % compute x coordinates
Yp = l1 * sin(THETA1P) + l2 * sin(THETA1P + THETA2P) + l3 * sin(THETA1P + THETA2P + THETA3P); % compute y coordinates

NN_Xp = l1 * cos(NN_THETA1P) + l2 * cos(NN_THETA1P + NN_THETA2P) + l3 * cos(NN_THETA1P + NN_THETA2P + NN_THETA3P); % compute x coordinates
NN_Yp = l1 * sin(NN_THETA1P) + l2 * sin(NN_THETA1P + NN_THETA2P) + l3 * sin(NN_THETA1P + NN_THETA2P + NN_THETA3P); % compute y coordinates

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

figure(1)
subplot(3,1,1)
quiver(XY(:,1),XY(:,2),err_X(:),err_Y(:))
title("Anfis");
axis([-2 22 0 19]); 

subplot(3,1,2)
quiver(XY(:,1),XY(:,2),NN_err_X(:),NN_err_Y(:))
title("MLP");
axis([-2 22 0 19]); 

subplot(3,1,3);
plot(XY(:,1),XY(:,2),'.'); 
title("Target");
axis([-2 22 0 19]); 

%% 3d plot of errors
anfis_3 = sqrt(err_X.^2+err_Y.^2);
nn_3 = sqrt(NN_err_X.^2+NN_err_Y.^2);

CircleSize=100;
figure(2);
h2=scatter3(XY(:,1),XY(:,2),anfis_3,CircleSize,anfis_3,'s','filled');
xlabel('X axis');
ylabel('Y axis');
zlabel('Error');
title('Anfis');
colorbar;
caxis([0 1])
figure(3);
h2=scatter3(XY(:,1),XY(:,2),nn_3,CircleSize,nn_3,'s','filled');
xlabel('X axis');
ylabel('Y axis');
zlabel('Error');
colorbar;
caxis([0 1])
title('MLP');

%% ANOVA test
close all;
anfis_anova = [angle_errors1(:) angle_errors2(:) angle_errors3(:)];
nn_anova = [NN_angle_errors1(:) NN_angle_errors2(:) NN_angle_errors3(:)];

anfis_p = anova1(anfis_anova);
nn_p = anova1(nn_anova);
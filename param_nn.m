l1 = 10; % length of first arm
l2 = 7; % length of second arm
% l3 = 4;

nr_p = 50;

theta1 = linspace(0, pi/4, nr_p); % all possible theta1 values
theta2 = linspace(0, pi/2, nr_p); % all possible theta2 values
% theta3 = linspace(0, pi/4, nr_p); % all possible theta3 values

[THETA1,THETA2] = meshgrid(theta1,theta2);
% [THETA1,THETA2,THETA3] = meshgrid(theta1,theta2,theta3);

% X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2) + l3 * cos(THETA1 + THETA2 + THETA3); % compute x coordinates
% Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2) + l3 * sin(THETA1 + THETA2 + THETA3); % compute y coordinates

X = l1 * cos(THETA1) + l2 * cos(THETA1 + THETA2);
Y = l1 * sin(THETA1) + l2 * sin(THETA1 + THETA2);

data1 = [X(:) Y(:) THETA1(:)]; % create x-y-theta1 dataset
data2 = [X(:) Y(:) THETA2(:)]; % create x-y-theta2 dataset
% data3 = [X(:) Y(:) THETA3(:)]; % create x-y-theta3 dataset
%%
train_data1 = data1(1:2:end,:);
train_data2 = data2(1:2:end,:);
% train_data3 = data3(1:2:end,:);
val_data1 = data1(2:2:end,:);
val_data2 = data2(2:2:end,:);
% val_data3 = data3(2:2:end,:);
%%
TRAIN_P = 80;
PARTS = 25;
GROUP_SIZE = length(data1)/PARTS;

all_points = 1:length(data1);

parts = reshape(all_points, [PARTS, GROUP_SIZE]);

train_points = [];
val_points = [];
test_points = [];
for i = 1:PARTS
    t = datasample(parts(i,:),floor(TRAIN_P/100*GROUP_SIZE), 'Replace', false);
    rest = setdiff(parts(i,:), t);
    
    v = datasample(rest,floor(size(rest,2)/2), 'Replace', false);
    tp = setdiff(rest, v);
    
    train_points = [train_points t];
    val_points = [val_points v];
    test_points = [test_points tp];
end

% load the points into data sets
train_data1 = data1(train_points, :);
train_data2 = data2(train_points, :);
% train_data3 = data3(train_points, :);

val_data1 = data1(val_points, :);
val_data2 = data2(val_points, :);
% val_data3 = data3(val_points, :);

test_data1 = data1(test_points, :);
test_data2 = data2(test_points, :);
% test_data3 = data3(test_points, :);
%%
% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation

x = {X(:)'; Y(:)';};
t = [data1(:,3)'; data2(:,3)';];

err = [];
time = [];
start = 5;
fin = 25;
for i = start:fin
    fprintf('Training with %d neurons.\n', i);
    tic;
    
    % Create a Fitting Network
    hiddenLayerSize = i;
    net = fitnet(hiddenLayerSize,trainFcn);

    % net.numInputs = 2;
    net.numInputs = 2;
    net.InputConnect(1,1) = 1;
    net.InputConnect(1,2) = 1;

    % manually set the train/val/test points
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = train_points;
    net.divideParam.valInd = val_points;
    net.divideParam.testInd = test_points;

    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'mse';  % Mean Squared Error

    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotregression', 'plotfit'};

    %configure
    net = configure(net,x,t);

    % Train the Network
    [net,tr] = train(net,x,t);

    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y);

    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
    valTargets = t .* tr.valMask{1};
    testTargets = t .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,y);
    valPerformance = perform(net,valTargets,y);
    testPerformance = perform(net,testTargets,y);
    
    train_t = toc;
%     figure;
%     plot(1:epochs,chkErr,'r');
%     hold on;
%     plot(1:epochs,trnErr,'b');
%     hold off;
%     pause(0.1);

    err = [err valPerformance];
    time = [time train_t];
end
%%
figure
subplot(4,1,1);
x = start:fin;
plot(x,err);
ylabel('Error'); 
xlabel('Neurons');
subplot(4,1,2);
plot(x,time);
ylabel('Time');
xlabel('Neurons');
subplot(4,1,3);
p_err = rescale(err,0,1);
p_time = rescale(time,0,1);
plot(x,p_err,'r');
hold on;
plot(x,p_time,'b');
hold off;
ylabel('Normalized err and time');
xlabel('Neurons');

subplot(4,1,4);
score = [];
for i = 1:length(p_err)
    score = [score 1/(p_time(i)+p_err(i))*10];
end
plot(x,score);
axis([start fin 0 max(score)+1]);
ylabel('Fitness');
xlabel('Neurons');

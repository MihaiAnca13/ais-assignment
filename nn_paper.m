%% prepare data sets
X = data1(:,1);
Y = data1(:,2);

% 2 link train data
% x = {X(:)';Y(:)'};
% t = [data1(:,3)'; data2(:,3)'];

% 3 link train data
x = {X(:)'; Y(:)';};
t = [data1(:,3)'; data2(:,3)'; data3(:,3)'];

%%
% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation

% Create a Fitting Network
hiddenLayerSize = [6 3];
net = fitnet(hiddenLayerSize,trainFcn);

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
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'MLP_2layers');
%     y = MLP_3joints(x);
end

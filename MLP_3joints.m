function [Y,Xf,Af] = MLP_3joints(X,~,~)
%MLP_3JOINTS neural network simulation function.
%
% Generated by Neural Network Toolbox function genFunction, 04-Mar-2019 11:59:50.
% 
% [Y] = MLP_3joints(X,~,~) takes these arguments:
% 
%   X = 3xTS cell, 3 inputs over TS timesteps
%   Each X{1,ts} = 1xQ matrix, input #1 at timestep ts.
%   Each X{2,ts} = 1xQ matrix, input #2 at timestep ts.
%   Each X{3,ts} = 1xQ matrix, input #3 at timestep ts.
% 
% and returns:
%   Y = 1xTS cell of 3 outputs over TS timesteps.
%   Each Y{1,ts} = 3xQ matrix, output #1 at timestep ts.
% 
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = -1.87867965644036;
x1_step1.gain = 0.087417632050152;
x1_step1.ymin = -1;

% Layer 1
b1 = [1.2027314367056314293;-18.855596735024612087;16.451338163872442522;-0.96602782277435372826;-1.6514762709147499109;-12.245863766046969801;-1.1221220972695362672;1.8287510555359354214;0.38232451561762914594;3.8126440277222481967];
IW1_1 = [4.7681234932256817416;8.9175122055945568178;-12.068926734141770751;0.53740495516319508607;1.2993455739595343168;10.136110662230150226;0.83805923834864737287;-1.882362149366818338;0.29559147403750002381;-0.1649623520240297847];
IW1_2 = [2.0311000543925668005;0.72241155000715706613;-0.50819685302047945541;-0.0074546719049656141182;0.0067963597167941874541;0.019010783966353899527;0.012811475704623884533;-0.00042891949698119257242;-0.0097529865897765990329;-0.15266490746916069887];
IW1_3 = [0.99766656430807554301;2.2014304316685557872;-1.6446504522878913068;0.72385404517654072798;0.59131289875820847168;1.8355911385124987767;0.21799830081111293278;-1.4792918772102578995;-0.085844477526220680508;-0.38843734945797409042];

% Layer 2
b2 = [14.694090332165091084;-15.761106928984242259;16.705244330957039978];
LW2_1 = [13.919615246709543754 6.7131642313994595028 -1.4951138783948560995 9.560209168653234002 -18.116358812089703889 10.430432650152507534 32.046972726075239279 4.4176440158813505832 -5.0730445444267449062 -1.4047829161804801235;-14.842945211623108648 -9.1739951623961406568 1.9497992830089119209 -11.821568578860885523 16.344569395466038486 -12.017244946004986161 -26.491600752318980483 -4.7534485049193984096 -4.3129290961341979482 2.5269378764056997788;16.096408174889013765 11.506953246290393622 -2.3705664311464600402 18.351500991140635932 -15.974479605266829196 13.745099936501576465 22.538654478499640987 5.4655074681934037173 9.1121095359053292384 -3.3289571314659012202];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = [2.54647908947033;1.27323954473516;2.54647908947033];
y1_step1.xoffset = [0;0;0];

% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if ~isCellX
  X = {X};
end

% Dimensions
TS = size(X,2); % timesteps
if ~isempty(X)
  Q = size(X{1},2); % samples/series
else
  Q = 0;
end

% Allocate Outputs
Y = cell(1,TS);

% Time loop
for ts=1:TS

    % Input 1
    Xp1 = mapminmax_apply(X{1,ts},x1_step1);
    
    % Input 2
    % no processing
    
    % Input 3
    % no processing
    
    % Layer 1
    a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*Xp1 + IW1_2*X{2,ts} + IW1_3*X{3,ts});
    
    % Layer 2
    a2 = repmat(b2,1,Q) + LW2_1*a1;
    
    % Output 1
    Y{1,ts} = mapminmax_reverse(a2,y1_step1);
end

% Final Delay States
Xf = cell(3,0);
Af = cell(2,0);

% Format Output Arguments
if ~isCellX
  Y = cell2mat(Y);
end
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
  y = bsxfun(@minus,x,settings.xoffset);
  y = bsxfun(@times,y,settings.gain);
  y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
  a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
  x = bsxfun(@minus,y,settings.ymin);
  x = bsxfun(@rdivide,x,settings.gain);
  x = bsxfun(@plus,x,settings.xoffset);
end

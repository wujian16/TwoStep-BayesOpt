%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run the simulator, parse its output, and print it to stdout
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% the input parameters
%b = [10 10 10 10 10 10 10 20];
%length = 10;
%seed = randi(100000000);

% invocation of the simulator
[fnAvg] = ATOHongNelson(b,length,seed);
% function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov]=ATOHongNelson(BaseStockLevel,TotalTime,seed,other)
% AssembleToOrder2() returns more, ignore the others for now

% output the mean and variance of the profit after 20 days
formatSpec = 'fn=%4.8f\nFnVar=NaN\n';
fprintf(1,formatSpec,fnAvg) % 1 is for stdout

%exit;
clear all
close all
clc

%% Loading data
load('Report5_2.mat');
rng(0)

%% Autoregressive model
lags = 1:24;
[X, Y] = autoregressive(y, lags);
beta = X \ Y;
y_t = sum(beta .* y(end - 24 + 1:end));

%% Estimate lasso for logarithmic grid of λ
lambda = logspace(-4, 0, 100); % 100 values between 10^(-4) and 10^0
[beta, FitInfo] = lasso(X, Y, 'Lambda', lambda);

lassoPlot(beta, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');
lambdaIndex = 10; % Index of the desired λ value
%beta = B(:, lambdaIndex);

y_hat = X*beta(:,1)+FitInfo.Intercept(1);
MSE = mean((Y-y_hat).^2);
MSE2 = FitInfo.MSE(1);

disp([MSE MSE2]); 

%% Select the optimal λ with BIC

N = length(y);

for i = 1:size(beta,2)
    y_hat = X*beta(:,i)+FitInfo.Intercept(i);
    L = sum((Y-y_hat).^2);
    ic(i) = bic(L,N,FitInfo.DF(i));
end

[~, idx] = min(ic);
BIC_optimal = FitInfo.Lambda(idx);

disp(['BIC optimal: ', num2str(BIC_optimal)]);

%% Select the optimal λ with CV with 10 folds

[B, FitInfo] = lasso(X,Y,'CV',10, 'Lambda', lambda); % 10 folds

CV_optimal = FitInfo.LambdaMinMSE;
CV_optimal_idx = FitInfo.IndexMinMSE;

% Corrected
CV2_optimal = FitInfo.Lambda1SE;
CV2_optimal_idx = FitInfo.Index1SE;

lassoPlot(B, FitInfo)

disp(['Optimal Lambda (MinMSE): ', num2str(CV_optimal)]);
disp(['Optimal Lambda (1SE): ', num2str(CV2_optimal)]);

%% For each of these models recognize the variables that stays in the final model (important variables)

important_variables_BIC = find(B(:, idx) ~= 0);
important_variables_CV_MinMSE = find(B(:, CV_optimal_idx) ~= 0);
important_variables_CV_1SE = find(B(:, CV2_optimal_idx) ~= 0);

disp('Important variables for BIC:');
disp(important_variables_BIC);

disp('Important variables for CV (MinMSE):');
disp(important_variables_CV_MinMSE);

disp('Important variables for CV (1SE):');
disp(important_variables_CV_1SE);

%% BIC function

function ic = bic(L,n,k)
ic = n*log(L/n)+log(n)*k;
end

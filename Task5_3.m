clear all
close all
clc

%first letter of my surname is S

t = readtable('POLEX.csv');
idx = strcmp(t.FirstLetterOfYourName, 'S');
data = t(idx, :);

%% Correlation

y = data.Price;
x1 = data.Load;
x2 = data.RES;

% Price vs. Load
[r1] = corr(y, x1) %transpose ' , Pearson
[r2] = corr(y, x1, 'type', 'Spearman')

% Price vs. RES
[r3] = corr(y, x2) %transpose ' , Pearson
[r4] = corr(y, x2, 'type', 'Spearman')

%r - correlation coef (-1, 1)
%positive r > 0.5
%negative r < -0.5
%anything between - no correlation

disp('Correlation between Price and Load is positive (r > 0.5).')
disp('Correlation between Price and RED is positive (r > 0.5).')

%% Liliefors test

[h1, p1] = lillietest(y) % p1=0.001???

%output: [h1, p1] -> h is the decision variable, if h = 0 then we do not
%reject H0, but if h = 1 we reject H0; p1 < alpha -> reject, p1 >= alpha ->
%dont reject H0

disp("For significance level alpha = 0.05, we reject hypothesis H0. When a hypothesis H0 is rejected, if a statistic falls in the critical region.")

%% Autoregressive model
lags = 1:168;
max_log = 168;
y = data.Price;
[X, Y] = autoregressive(y, lags);
beta = X \ Y;
y_t = sum(beta .* y(end - max_log + 1:end));

Lt = data(max_log+1:end, 5);
Rt = data(max_log+1:end, 6);
Et = data(max_log+1:end, 7);
Nt = data(max_log+1:end, 8);

X_tabl = array2table(X);
X = horzcat(X_tabl, Lt, Rt, Et, Nt);

%% Run linear regression
X = table2array(X);
[betas L se] = OLS2(X, Y);

%% BIC
N = length(Y);
K1 = size(X,2);

BIC = bic2(L, N, K1)

%% t-test

p = myttest(betas, se, N, K1);
disp('Irrelevant columns are when their p-values are larger than 0.05.')

rows = find(p < 0.05);
disp('Rows smaller than 0.05:');
disp(rows);

X_tabl = array2table(X);
XF = X_tabl(:, rows);

XF = table2array(XF);
[betas2 L2 se2] = OLS2(XF, Y);
K2 = size(XF,2);

BIC2 = bic2(L2, N, K2)

%% Estimate 20 factors

N = size(X,1);
X = zscore(X);
Kmax = 20;
[F_t, D] = eigs(X*X', Kmax);
F = F_t * sqrt(N);
plot(F(:,1:3))

%% IPC criterion to assess optimal number of factors

K = size(X);
Kmax = 20;
lambda = F'*X/N;

V = nan(1,Kmax);
for kk=1:Kmax
    e = X - F(:, 1:kk)*lambda(1:kk,:);
    V(kk) = mean(mean(e.^2));
end

for kk=1:Kmax
    IPC(kk) = log(V(kk))+kk*(N+K)/(N*K)*log(N*K/(N+K));
end

figure(2)
plot(IPC)
title('Plot of IPC')
xlabel('Factors')
ylabel('Variance')

%% With OLS estimate the model consisting of optimal number of factors and intercept

rng(0)
beta_OLS = OLS([F ones(N,1)],Y);

y_hat = [F ones(N,1)]* beta_OLS;

L3 = sum((Y-y_hat).^2);

K3 = 21;
BIC3 = bic2(L3, N, K3)

%% Estimate lasso for default grid of λ and 7 folds of cross validation

[beta, fitInfo] = lasso(X,Y);

for i = 1:size(beta,2)
    y_hat = X*beta(:,i)+fitInfo.Intercept(i);
    L = sum((Y-y_hat).^2);
    ic(i) = bic2(L,N,fitInfo.DF(i));
end

[~, idx] = min(ic);
BIC_optimal = fitInfo.Lambda(idx);

[beta, fitInfo] = lasso(X,Y,'CV',7);

CV_optimal = fitInfo.LambdaMinMSE;
CV_optimal_idx = fitInfo.IndexMinMSE;

% Corrected
CV2_optimal = fitInfo.Lambda1SE;
CV2_optimal_idx = fitInfo.Index1SE;

lassoPlot(beta, fitInfo)

disp(['Optimal Lambda (MinMSE): ', num2str(CV_optimal)]);
disp(['Optimal Lambda (1SE): ', num2str(CV2_optimal)]);

%% Calculate BIC for model with optimal λ

y_hat_optimal = X * beta(:, CV2_optimal_idx) + fitInfo.Intercept(CV2_optimal_idx);
L_optimal = sum((Y - y_hat_optimal).^2);
p_optimal = fitInfo.DF(CV2_optimal_idx);
N = size(Y, 1);
BIC4 = bic2(L_optimal, N, p_optimal)

%% Table

resultTable = table();

resultTable.Method{1} = 'Linear Regression';
resultTable.BIC(1) = BIC;
resultTable.NumVariables(1) = K1;

resultTable.Method{2} = 'Linear Regression after t-test';
resultTable.BIC(2) = BIC2;
resultTable.NumVariables(2) = K2;

resultTable.Method{3} = 'PCA';
resultTable.BIC(3) = BIC3;
resultTable.NumVariables(3) = K3;

resultTable.Method{4} = 'LASSO';
resultTable.BIC(4) = BIC4;
resultTable.NumVariables(4) = p_optimal;

disp(resultTable);

%% Check the autocorrelation of the residuals for 20 first lags with Q-test for the model with lowest BIC

%The lowest BIC is from method - Linear Regression after t-test

e = Y - XF*betas2;

p = 20; %lags

rho = autocorr(e, p);
rho(1) = [];

Q = N*(N+2)*sum(rho.^2./(N-(1:20)'));
p_Q = 1-chi2cdf(Q, p) %0.35 > 0.05 - residuals are not autocorrelated, we cant rejest H0 hypothesis
disp('The p-value from Q test (p_Q) is 0.8947 and it is bigger that 0.05 (significance level), so residuals are not autocorrelated, so we should NOT reject hypothesis H0.')

%% Check the homoscedasticity with Breusch-Pagan LM test for the model with lowest BIC

g=e.^2/(sum(e.^2)/N)-1;
z = XF;
BP = 0.5*g'*z*(z'*z)^(-1)*z'*g;
p_BP=1-chi2cdf(BP,K2)
disp('The p-value from BP test (p_BP) is 0 and it is smaller that 0.05 (significance level), so residuals are heteroscedastic, so we should reject hypothesis H0.')

%4.988
%4.245
%4.793
%4.368

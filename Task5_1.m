clear all
close all
clc

load('Report5_1.mat');
N = length(Y);
y = zscore(Y);
Kmax = 20;
[F_t, D] = eigs(y*y', Kmax);
F = F_t * sqrt(N);

explained_var = diag(D)./trace(y'*y);

figure(1)
bar(explained_var)
title('Bar plot of variances')
xlabel('Factors')
ylabel('Variance')

lambda = F'*y/N;

K = size(Y);
V = nan(1,Kmax);
for kk=1:Kmax
    e = y - F(:, 1:kk)*lambda(1:kk,:);
    V(kk) = mean(mean(e.^2));
end

for kk=1:Kmax
    PC(kk) = V(kk)+kk*V(Kmax)*(N+K)/(N*K)*log(N*K/(N+K));
    IPC(kk) = log(V(kk))+kk*(N+K)/(N*K)*log(N*K/(N+K));
end

figure(2)
plot(PC)
title('Plot of PC')
xlabel('Factors')
ylabel('Variance')

figure(3)
plot(IPC)
title('Plot of IPC')
xlabel('Factors')
ylabel('Variance')

disp(PC(1, 16))
disp(IPC(1, 5))

disp('When looking at bar plot we can see that the optimal number of factors should be 5. Because after 5 factors, the variance significantly decreases.')
disp('When taking into consideration PC value, we see that the lowest value is 13 factor. But what is important is that after 5 factors, the variance does not decreases much.')
disp('Plotting IPC is confirming the fact that the optimal number of factors should be 5.')
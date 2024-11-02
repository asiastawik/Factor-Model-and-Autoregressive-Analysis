function [X, Y] = autoregressive(y, lags)
n = length(y);
p = length(lags);
l = max(lags);
X = zeros(n-l,p); 
Y = y(l+1:end);

for i = 1:p
    X(:,i) = y(l+1-lags(i):end-lags(i));
end
end
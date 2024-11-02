function [betas L se] = OLS2(X,Y)
betas = (X'*X)^(-1)*X'*Y;
e = Y-X*betas;
L = sum(e.^2);
normX = diag((X'*X)^-1);
se = sqrt(var(e).*normX);
end
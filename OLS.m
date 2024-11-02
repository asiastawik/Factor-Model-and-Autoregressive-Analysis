function [betas L] = OLS(X,Y)
betas = (X'*X)^(-1)*X'*Y;
e = Y-X*betas;
L = sum(e.^2);
end
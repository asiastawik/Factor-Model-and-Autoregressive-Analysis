function p = myttest(betas, se, n, k)
stat = betas./se;
p = 2*(1-tcdf(abs(stat), n-k));
end
function [c,g] = g_logistic(x,A,b,g_star,lambda_g)
% lower-level for fmincon

[m,n]=size(A);
g = [];
c1 = (1/m)*ones(1,m)*log((1+exp(-(A*x).*b))) - g_star - 1e-10;
c2 = sum(abs(x))-lambda_g;
c = [c1;c2];
end
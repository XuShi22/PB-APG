function [c,g] = g_linear(x,A,b,g_star)
% lower-level for fmincon

[m,n]=size(A);
g = [];
c = (1/m)*(A*x-b)'*(A*x-b)/2 - g_star - 1e-10;
end
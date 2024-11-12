function [f,grad_f] = f_linear(x,alpha)
% upper-level for fmincon

f = norm(x,1) + alpha*(x'*x)/2;
grad_f = [];
end
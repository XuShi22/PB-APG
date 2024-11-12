function [f,grad_f] = f_logistic(x)
% upper-level for fmincon

f = x'*x/2;
grad_f = x;
end
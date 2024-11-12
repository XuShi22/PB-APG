function [f_vec,g_vec,time_vec,x] = BigSAM(fun_f,grad_f,grad_g,fun_g,param,x0)
% BiG-SAM in "A first order method for solving convex bilevel optimization problems",
% S. Sabach and S. Shtern, SIOPT 2017

eta_f = param.eta_f;
eta_g= param.eta_g;
lambda = param.lam;
gamma = param.gamma;
maxiter = param.maxiter;
maxtime = param.maxtime;
x = x0;
tic;
iter = 0;
f_vec = [];
g_vec = [];
time_vec = [];
while iter <= maxiter
    iter = iter+1;
    x_lo = x-eta_g*grad_g(x);
    x_lo = ProjectOntoL1Ball(x_lo,lambda);
    x_up = x-eta_f*grad_f(x);
    alpha = min([2*gamma/iter,1]);
    x = alpha*x_up + (1-alpha)*x_lo;
    cpu_t = toc;
    f_vec = [f_vec;fun_f(x)];
    g_vec = [g_vec;fun_g(x)];
    time_vec = [time_vec;cpu_t];
    if mod(iter,1000) == 1
        fprintf('Iteration: %d\n',iter)
    end
    if cpu_t>maxtime
        break
    end
end
end
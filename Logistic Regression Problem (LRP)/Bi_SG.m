function [f_vec,g_vec,time_vec,x] = Bi_SG(fun_f,grad_f,grad_g,fun_g,param,x0)
% Bi-SG in "Convex Bi-Level Optimization Problems with Non-smooth Outer Objective Function",
% R. Merchav and S. Shtern, SIOPT 2023

eta_g= param.eta_g;
lambda = param.lam;
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
    a = 0.95;
    eta_k = (iter + 1)^(-a);
    x = x_lo-eta_k*grad_f(x_lo);
    cpu_t1 = toc;
    f_vec = [f_vec;fun_f(x)];
    g_vec = [g_vec;fun_g(x)];
    time_vec = [time_vec;cpu_t1];
    if mod(iter,1000) == 1
        fprintf('Iteration: %d\n',iter)
    end
    if cpu_t1>maxtime
        break
    end
end
end
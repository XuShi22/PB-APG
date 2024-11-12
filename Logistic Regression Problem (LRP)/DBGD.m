function [f_vec,g_vec,time_vec,x] = DBGD(fun_f,grad_f,grad_g,fun_g,param,x0)
% DBGD in "Bi-objective trade-off with dynamic barrier gradient descent",
% C. Gong, X. Liu, and Q. Liu, Neurips 2021

stepsize = param.stepsize;
alpha = param.alpha;
beta = param.beta;
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
    grad_f_x = grad_f(x);
    grad_g_x = grad_g(x);
    phi = min(alpha*fun_g(x),beta*(grad_g_x'*grad_g_x));
    weight = max((phi-grad_f_x'*grad_g_x)/(grad_g_x'*grad_g_x),0);
    v = grad_f_x+weight*grad_g_x;
    x = x-stepsize*v;
    % Projection to simplex
    x = ProjectOntoL1Ball(x,lambda);
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
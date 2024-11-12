function [f_vec,g_vec,time_vec,x] = CG_BiO(fun_f,grad_f,grad_g,fun_g,param,x0)
% CG-BiO in "A Conditional Gradient-based Method for Simple Bilevel Optimization with Convex Lower-level Problem",
% R. Jiang, N. Abolfazli, A. Mokhtari, E. Yazdandoost Hamedani, AISTATS2023

epsilon_f= param.epsilonf;
epsilon_g= param.epsilong;
lambda = param.lam;
maxiter = param.maxiter;
maxtime = param.maxtime;
n = length(x0);
x = x0;
fun_g_x0 = param.fun_g_x0;
tic;
iter = 0;
f_vec = [];
g_vec = [];
time_vec = [];
while iter <= maxiter
    iter = iter+1;
    gamma = 2/(iter+2+10);
    b=[grad_g(x)'*x+fun_g_x0-fun_g(x); lambda];
    A = [grad_g(x)' -grad_g(x)'; ones(1,2*n)];
    lb=[sparse(2*n,1)];
    f=[grad_f(x)' -grad_f(x)'];
    options = optimoptions('linprog','Algorithm','dual-simplex','Display','off');
    vec = linprog(f,A,b,[],[],lb,[],options);
    d=vec(1:n)-vec(n+1:end);
    if grad_f(x)'*(x-d)<=epsilon_f && (fun_g(x)-fun_g_x0)<=epsilon_g/2
        break;
    end
    x = (1-gamma)*x + gamma*d;
    cpu_t = toc;
    cpu_t = cpu_t + param.inittime;
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
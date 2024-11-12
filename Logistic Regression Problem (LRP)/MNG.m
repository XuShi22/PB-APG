function [f_vec,g_vec,time_vec,x] = MNG(fun_f,grad_f,fun_g,grad_g,param,x0)
% MNG in "A first order method for finding minimal norm-like solutions of convex optimization problems",
% A. Beck and S. Sabach, Math. Program. 2014

M = param.M;
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
    y = ProjectOntoL1Ball(x-1/M*grad_g(x),lambda);
    G_M = M*(x-y);
    A_ineq = [G_M';-grad_f(x)'];
    b_ineq = [G_M'*x-3/4/M*(G_M'*G_M);-grad_f(x)'*x];
    H = eye(length(x));
    f = zeros(length(x),1);
    options = optimoptions('quadprog','Display','None');
    x = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);
    cpu_t = toc;
    f_vec = [f_vec;fun_f(x)];
    g_vec = [g_vec;fun_g(y)];
    time_vec = [time_vec;cpu_t];
    if mod(iter,1000) == 1
        fprintf('Iteration: %d\n',iter)
    end
    if cpu_t>maxtime
        break
    end
end
end
function[f_vec,g_vec,time_vec,x] = Alg_Projection(fun_f,grad_f,grad_g,fun_g,param,x0)
% a-IRG in "A method with convergence rates for optimization
% problems with variational inequality constraints", H. D. Kaushik and F. Yousefian, SIOPT 2021

eta_0 = 1e-3;
gamma_0 = 1/param.L_g;
lambda=param.lam;
f_vec = [];
g_vec = [];
time_vec = [];
x = x0;
maxiter = param.maxiter;
maxtime = param.maxtime;
tic;
for k = 1 : maxiter
    eta_k = (eta_0)/(k+1)^0.25;
    gamma_k = gamma_0/sqrt(k+1);
    x = x - gamma_k*(grad_g(x)+eta_k*(grad_f(x)));
    x = ProjectOntoL1Ball(x,lambda);
    cpu_t = toc;
    f_vec = [f_vec;fun_f(x)];
    g_vec = [g_vec;fun_g(x)];
    time_vec = [time_vec;cpu_t];
    if mod(k,1000) == 1
        fprintf('Iteration: %d\n',k)
    end
    if cpu_t>maxtime
        break
    end
end
end
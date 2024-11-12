function [x, fk, f_vec, g_vec, time_vec,its] = R_APM(fun_f,fun_g, grad_f,proxg, x0, opts, fun_f_p, fun_g_p)
% Regularized Accelerated Proximal Method (R-APM) (Algorithm 1) in "Achieving optimal complexity guarantees for a class of bilevel convex optimization problems",
% S. Samadi ,D. Burbano, and F. Yousefian, arXiv: 2310.12247

gamma = 1/opts.L0;
maxits = opts.maxiter;
x = x0;
y = x0;
its = 1;

f_vec = [];
g_vec = [];
time_vec = [];
tic;
tk0 = 1;
while(its < maxits)
    x_old = x;
    x = proxg(y-gamma*grad_f(y));
    tk1 = (0.5 + sqrt(0.25 + tk0^2));
    ak = (tk0 - 1) / tk1;
    y = x + ak*(x-x_old);
    tk0 = tk1;
    cpu_t = toc;
    f_vec = [f_vec;fun_f_p(x)];
    g_vec = [g_vec;fun_g_p(x)];
    time_vec = [time_vec;cpu_t];

    res = norm(x_old-x, 'fro');
    if mod(its, 1e2)==0
        itsprint(sprintf('step %08d: residual = %.3e\n', its,res), its);
    end

    if (res<opts.tol)||(res>1e10)
        break;
    end

    if cpu_t > opts.maxtime
        break;
    end

    its = its + 1;

end
fprintf('\n');

fk = fun_f(x)+fun_g(x);
end
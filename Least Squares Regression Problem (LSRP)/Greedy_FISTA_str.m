function [x, fk, f_vec, g_vec, time_vec, its] = Greedy_FISTA_str(fun_f,fun_g, grad_f,proxg, x0, opts, fun_f_p, fun_g_p)
% Greddy FISTA (Algorithm 4.3) in "Improving "fast iterative shrinkage-thresholding algorithm": faster, smarter, and greedier",
% J. Liang, T. Luo, and CB. Schonlieb, SIAM J Sci Comput 2022
% Here we modify 'opts.a' satisfies strongly convex setting

if ~isfield(opts, 'maxit')
    opts.maxit = 1e+6;
end
if ~isfield(opts, 'c_gamma')
    opts.c_gamma = 1.3;
end
if ~isfield(opts, 'a')
    opts.a = @(k) 1; %max(2/(1+k/5), 1.0);
end
gamma0 = 1/opts.L0;
gamma = opts.c_gamma * gamma0;
maxits = opts.maxit+1;
x = x0;
y = x0;
ek = zeros(maxits, 1);
a = opts.a;
% tor = 0;
S = 1;
xi = 0.96;
its = 1;

f_vec = [];
g_vec = [];
time_vec = [];
tic;
while(its<maxits)
    x_old = x;
    y_old = y;
    % opts0 = opts;
    % opts0.lambda = gamma;
    x = proxg(y-gamma*grad_f(y),gamma);
    y = x + a(its)*(x-x_old);
    
    cpu_t = toc;
    f_vec = [f_vec;fun_f_p(x)];
    g_vec = [g_vec;fun_g_p(x)];
    % acc_vec = [acc_vec;TSA(x)];
    time_vec = [time_vec;cpu_t];
    % gradient criteria
    vk = (y_old(:)-x(:))'*(x(:)-x_old(:));
    if vk >= 0
        y = x;
    end

    res = norm(x_old-x, 2);
    if mod(its, 1e2)==0
        itsprint(sprintf('step %08d: residual = %.3e\n', its,res), its);
    end

    ek(its) = res;
    if (res<opts.tol)||(res>1e10)
        break;
    end

    % safeguard
    if res>S*ek(1)
        gamma = max(gamma0, gamma*xi);
    end

    its = its + 1;

end
fprintf('\n');

fk = fun_f(x)+fun_g(x);

end
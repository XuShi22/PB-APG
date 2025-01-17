function [last_iter , f_hist] = CG_lowerlevel(fun_g,grad_g,x0,param)
% Standard CG algorithm for solving the lower-level problem
% CG-BiO in "A Conditional Gradient-based Method for Simple Bilevel Optimization with Convex Lower-level Problem",
% R. Jiang, N. Abolfazli, A. Mokhtari, E. Yazdandoost Hamedani, AISTATS2023
disp('CG for the lower level starts');

epsilon_g= param.epsilong;
lambda1 = param.lam1;
x = x0;
f_hist = fun_g(x0);
iteration = 0;
maxiteration = param.maxiter;
while iteration <= maxiteration
    iteration = iteration+1;
    gam = 2/(iteration+2);
    dir = linear_l1(grad_g(x),lambda1);
    if grad_g(x)'*(x-dir)<=epsilon_g
        break;
    end
    x = (1-gam)*x + gam*dir;
    f_hist = [f_hist;fun_g(x)];
end
disp('CG for the lower level is solved!');
last_iter = x;
end

function x = linear_l1(c,lambda)
x = sparse(length(c),1);
[~,ind] = max(abs(c));
x(ind) = -sign(c(ind))*lambda;
end
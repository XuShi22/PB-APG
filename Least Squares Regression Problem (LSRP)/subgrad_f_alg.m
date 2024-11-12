function subgrad = subgrad_f_alg(x,alpha)
% subgradient of upper-level

n = length(x);
subgrad = zeros(n,1);
for i = 1:n
    if x(i)>=0
        subgrad(i) = 1;
    else
        subgrad(i) = -1;
    end
end
subgrad = subgrad + alpha*x;
end
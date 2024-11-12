function w = ProximalL1norm(v, t, lambda)
% proximal operator of 1-norm
% t is step-size, lambda is regular parameter
lt = lambda*t;
n = length(v);
w = zeros(n,1);
if (lt < 0)
    error('Radius of L1 ball is negative: %2.3f\n', lt);
end
for i = 1:n
    if v(i)>lt
        w(i) = v(i)-lt;
    elseif v(i)<-lt
        w(i) = v(i)+lt;
    else
        w(i) = 0;
    end
end
end
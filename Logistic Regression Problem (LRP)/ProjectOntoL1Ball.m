function w = ProjectOntoL1Ball(v, b)
% PROJECTONTOL1BALL Projects v onto L1 ball of specified radius b 
% in "Efficient Projections onto the ℓ1-Ball for Learning in High Dimensions".
% J. Duchi, S. Shalev-Shwartz and Y. Singer, ICML 2008

if (b < 0)
    error('Radius of L1 ball is negative: %2.3f\n', b);
end
if (norm(v, 1) < b)
    w = v;
    return;
end
u = sort(abs(v),'descend');
sv = cumsum(u);
rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');
theta = max(0, (sv(rho) - b) / rho);
w = sign(v) .* max(abs(v) - theta, 0);
end
%%
% min ||x||_1 + alpha*||x||^2_2/2
%s.t. argmin ||Ax-b||_2^2
clear;clc;close all;
format long


%% load data
seed = 123456;
rng(seed);
X = load('YearPredictionMSD.txt');
[mm,~] = size(X);
num = 1000; % train number
co = randperm(mm,num); XX = X(co',:); % randomly select 1000 rows
bb = XX(:,1);AA = XX(:,2:end); %bb: label，AA: feature matrix
AA = normalize(AA,'range',[-1,1]); %scaling feature matrix to [-1,1]
AA = [AA,ones(num,1)]; % add an interceptor term
A = full(AA);b = full(bb);
b = normalize(b,'range');
[m,n]= size(A);

% add co-linear attributes
A_tilde = zeros(m,n-1);
for i = 1:n-1
    idx = randperm(n-1,10);
    B = A(:,idx); % generate a matrix contain 10 column of A
    v = unifrnd(-1,1,[10,1]);
    A_tilde(:,i) = B*v;
end
A = [A,A_tilde];
A2=A;
b2=b;
[m,n]= size(A2);

x0 = rand(n,1);% initial point
maxiter=1e6; % max iterations

%% function definition
% upper level function
alpha = 0.02;
fun_f= @(x) norm(x,1) + alpha*(x'*x)/2;
% lower level function
fun_g = @(x) (1/m)*(A2*x-b2)'*(A2*x-b2)/2;
fun_h = @(x) 0;

% upper gradient
grad_f = @(x) 0; % donot need
%lower gradient
grad_g = @(x) (1/m)*A2'*(A2*x-b2);

% upper Lipschitz
L_f = sqrt(n) + alpha;
% lower Lipschitz
L_g = (1/m)*eigs(A2'*A2,1);
A2 = full(A2);b2=full(b2);

% Find optimal
% lower optimal: lsqminnorm
xx = lsqminnorm(A2'*A2,A2'*b2);
gstar = fun_g(xx)
norm(grad_g(xx))

% total optimal: fmincon
f = @(x) f_linear(x,alpha); g = @(x) g_linear(x,A2,b2,gstar);
A=[];b=[];Aeq=[];beq=[];lb=[];ub=[];
options = optimoptions('fmincon','Display','iter','Algorithm','active-set','HessianApproximation','bfgs','MaxFunctionEvaluations',10e6,'MaxIterations',5e4,'ConstraintTolerance',1e-13,'OptimalityTolerance',1e-13,'StepTolerance',1e-13,'FunctionTolerance',1e-13);
[x_fmincon,fval,exitflag,output,lambda_fmin,grad] = fmincon(f,x0,A,b,Aeq,beq,lb,ub,g,options);
fstar_fmincon = fun_f(x_fmincon)
fun_g(x_fmincon)-gstar
norm(grad_g(x_fmincon))


%% PB-APG
epsilon = 1e-10; %termination criterion
gamma_tot = 100000; %penalty parameter
gamma1 = gamma_tot; %penalty parameter
lambda = 1;
L = alpha + gamma1*L_g;
opts.L_f = L_f;
opts.L_g = L_g;
opts.L0 = L;
opts.lambda = lambda;
opts.tol = epsilon; % tolerance
fun = @(x) alpha*(x'*x)/2 + gamma1*(1/m)*(A2*x-b2)'*(A2*x-b2)/2;
grad = @(x) alpha*x + gamma1*(1/m)*A2'*(A2*x-b2);
fun_h = @(x) lambda*norm(x,1);
proxh = @(x,t) ProximalL1norm(x, t, lambda);
opts.a = @(x) 1;
[x_fista, gstar_fista, f_vec, g_vec, time_vec, iter] = Greedy_FISTA_init(fun, fun_h, grad, proxh, x0, opts, fun_f, fun_g);
f_vec = [fun_f(x0);f_vec]; g_vec = [fun_g(x0);g_vec]; time_vec = [0;time_vec];

fun_f(x_fista) - fstar_fmincon
fun_g(x_fista) - gstar
norm(grad_g(x_fista))
fprintf('total iterations are %d\n',iter);


%% aPB-APG
gamma = gamma_tot/(20^5); %initial gamma
f_vecg1 = [fun_f(x0)]; g_vecg1 = [fun_g(x0)]; time_vecg1 = [0];
fold = 5;
x00 = x0;
tol_vec = [1e-6;1e-7;1e-8;1e-9;1e-10]; %termination criterion vector for 1e-10
% tol_vec = [1e-0;1e-1;1e-2;1e-3;1e-4]; %termination criterion vector for 1e-4
% tol_vec = [1e-3;1e-4;1e-5;1e-6;1e-7]; %termination criterion vector for 1e-7
iterg1 = 0;
opts.a = @(x) 1;
for i = 1:fold
    gamma1 = gamma*(20^i);
    L = L_f + gamma1*L_g;
    opts.L_f = L_f;
    opts.L_g = L_g;
    opts.L0 = L;
    opts.tol = tol_vec(i); % tolerance
    fun = @(x) alpha*(x'*x)/2 + gamma1*(1/m)*(A2*x-b2)'*(A2*x-b2)/2;
    grad = @(x) alpha*x + gamma1*(1/m)*A2'*(A2*x-b2);
    fun_h = @(x) lambda*norm(x,1);
    proxh = @(x,t) ProximalL1norm(x, t, lambda);

    [x_fista1g11, gstar_fista1g11, f_vecg11, g_vecg11, time_vecg11, iterg11] = Greedy_FISTA_init(fun, fun_h, grad, proxh, x00, opts, fun_f, fun_g);

    x00 = x_fista1g11;
    f_vecg1 = [f_vecg1;f_vecg11];
    g_vecg1 = [g_vecg1;g_vecg11];
    time_vecg11 = time_vecg11 + time_vecg1(end);
    time_vecg1 = [time_vecg1;time_vecg11];
    iterg1 = iterg1 + iterg11;
end

fun_f(x_fista1g11) - fun_f(x_fmincon)
fun_g(x_fista1g11) - fun_g(x_fmincon)
norm(grad_g(x_fista1g11))
fprintf('total iterations are %d\n',iterg1);


%% PB-APG-sc
% epsilon = 1e-10; %termination criterion
gamma1 = gamma_tot; %penalty parameter
lambda = 1;
L = alpha + gamma1*L_g;
opts.L_f = L_f;
opts.L_g = L_g;
opts.L0 = L;
opts.lambda = lambda;
opts.tol = epsilon; % tolerance
fun = @(x) alpha*(x'*x)/2 + gamma1*(1/m)*(A2*x-b2)'*(A2*x-b2)/2;
grad = @(x) alpha*x + gamma1*(1/m)*A2'*(A2*x-b2);
fun_h = @(x) lambda*norm(x,1);
proxh = @(x,t) ProximalL1norm(x, t, lambda);
opts.a = @(x) (sqrt(L) - sqrt(1))/(sqrt(L) + sqrt(1));

[x_fistastr, gstar_fistastr, f_vecstr, g_vecstr, time_vecstr, iterstr] = Greedy_FISTA_str(fun, fun_h, grad, proxh, x0, opts, fun_f, fun_g);
f_vecstr = [fun_f(x0);f_vecstr]; g_vecstr = [fun_g(x0);g_vecstr]; time_vecstr = [0;time_vecstr];
fun_f(x_fistastr) - fstar_fmincon
fun_g(x_fistastr) - gstar
norm(grad_g(x_fistastr))
fprintf('total iterations are %d\n',iterstr);


%% aPB-APG-sc
gamma = gamma_tot/(20^5); %initial gamma
f_vecg1str = [fun_f(x0)]; g_vecg1str = [fun_g(x0)]; time_vecg1str = [0];
fold = 5;
x00 = x0;
iterg1str = 0;
for i = 1:fold
    gamma1 = gamma*(20^i);
    L = L_f + gamma1*L_g;
    opts.L_f = L_f;
    opts.L_g = L_g;
    opts.L0 = L;
    opts.tol = tol_vec(i); % tolerance
    fun = @(x) alpha*(x'*x)/2 + gamma1*(1/m)*(A2*x-b2)'*(A2*x-b2)/2;
    grad = @(x) alpha*x + gamma1*(1/m)*A2'*(A2*x-b2);
    fun_h = @(x) lambda*norm(x,1);
    proxh = @(x,t) ProximalL1norm(x, t, lambda);
    opts.a = @(x) (sqrt(L) - sqrt(1))/(sqrt(L) + sqrt(1));

    [x_fista1g11str, gstar_fista1g11str, f_vecg11str, g_vecg11str, time_vecg11str, iterg11str] = Greedy_FISTA_init(fun, fun_h, grad, proxh, x00, opts, fun_f, fun_g);

    x00 = x_fista1g11str;
    f_vecg1str = [f_vecg1str;f_vecg11str];
    g_vecg1str = [g_vecg1str;g_vecg11str];
    time_vecg11str = time_vecg11str + time_vecg1str(end);
    time_vecg1str = [time_vecg1str;time_vecg11str];
    iterg1str = iterg1str + iterg11str;
end

fun_f(x_fista1g11str) - fun_f(x_fmincon)
fun_g(x_fista1g11str) - fun_g(x_fmincon)
norm(grad_g(x_fista1g11str))
fprintf('total iterations are %d\n',iterg1str);

% find max time
tt = [time_vec(end),time_vecg1(end),time_vecstr(end),time_vecg1str(end)];
max_time = max(tt);

plus_time = 0.1;


%% a-IRG Algorithm
param.maxtime = max_time + plus_time;
param.maxiter=1e6;
param.L_g = L_g;
param.L_f = L_f;
subgrad_f = @(x) subgrad_f_alg(x,alpha);
disp('a-IRG Algorithm starts')
[f_vec2,g_vec2,time_vec2,xlast] = Alg_Projection(fun_f,subgrad_f,grad_g,fun_g,param,x0);
f_vec2 = [fun_f(x0);f_vec2];g_vec2 = [fun_g(x0);g_vec2];time_vec2 = [0;time_vec2];
disp('a-IRG Solution Achieved!');


%% BiG-SAM Algorithm delta = 1
param.delta = 1;
param.eta_g = 1/L_g;
param.eta_f = 1/L_f;
param.gamma = 10;
param.maxtime = max_time + plus_time;
grad_f1 = @(x) alpha*x;
disp('BiG-SAM Algorithm starts');
[f_vec3,g_vec3,time_vec3,xlast3] = BigSAM(fun_f,grad_f1,grad_g,fun_g,param,x0);
f_vec3 = [fun_f(x0);f_vec3];g_vec3 = [fun_g(x0);g_vec3];time_vec3 = [0;time_vec3];
disp('BiG-SAM Solution Achieved!');


%% BiG-SAM Algorithm delta = 1e-2
param.delta = 1e-2;
param.eta_g = 1/L_g;
param.eta_f = 1/L_f;
param.gamma = 10;
param.maxtime = max_time + plus_time;
grad_f1 = @(x) alpha*x;
disp('BiG-SAM Algorithm starts');
[f_vec3_1,g_vec3_1,time_vec3_1,xlast3_1] = BigSAM(fun_f,grad_f1,grad_g,fun_g,param,x0);
f_vec3_1 = [fun_f(x0);f_vec3_1];g_vec3_1 = [fun_g(x0);g_vec3_1];time_vec3_1 = [0;time_vec3_1];
disp('BiG-SAM Solution Achieved!');


%% Bi-SG Algorithm
param.eta_g = 1/L_g;
param.maxtime = max_time + plus_time;
disp('Bi-SG Algorithm starts');
grad_f1 = @(x) alpha*x;
[f_vec6,g_vec6,time_vec6,xlast6] = Bi_SG(fun_f,grad_f1,grad_g,fun_g,param,x0);
f_vec6 = [fun_f(x0);f_vec6];g_vec6 = [fun_g(x0);g_vec6];time_vec6 = [0;time_vec6];
disp('Bi-SG Solution Achieved!');


%% Figures
%% lower-level gap
figure (1);
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',3)
set(gcf,'DefaultLineMarkerSize',4);
set(gcf,'Position',[331,167,591,586])

semilogy(time_vec,(g_vec-gstar),'-','DisplayName','PB-APG','LineWidth',5);
hold on;
semilogy(time_vecg1,(g_vecg1-gstar),'-','DisplayName','aPB-APG','LineWidth',6);
semilogy(time_vecstr,(g_vecstr-gstar),'-','DisplayName','PB-APG-sc','LineWidth',4);
semilogy(time_vecg1str,(g_vecg1str-gstar),'-','DisplayName','aPB-APG-sc','LineWidth',3);
semilogy(time_vec3,(g_vec3-gstar),'--','DisplayName','BiG-SAM($\delta = 1$)');
semilogy(time_vec3_1,(g_vec3_1-gstar),'--','DisplayName','BiG-SAM($\delta = 0.01$)');
semilogy(time_vec2,(g_vec2-gstar),'--','DisplayName','a-IRG');
semilogy(time_vec6,(g_vec6-gstar),'--','DisplayName','Bi-SG');

ylabel('$G(x_k)-G^*$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast','FontSize',10);
pbaspect([1 0.8 1])


%% upper-level
figure (2);
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',3)
set(gcf,'DefaultLineMarkerSize',4);
set(gcf,'Position',[331,167,591,586])
f_fstar = f_vec-fstar_fmincon;

plot(time_vec,f_vec,'-','DisplayName','PB-APG','LineWidth',5);
hold on;
plot(time_vecg1,f_vecg1,'-','DisplayName','aPB-APG','LineWidth',6);
plot(time_vecstr,f_vecstr,'-','DisplayName','PB-APG-sc','LineWidth',4);
plot(time_vecg1str,f_vecg1str,'-','DisplayName','aPB-APG-sc','LineWidth',3);
plot(time_vec3,f_vec3,'--','DisplayName','BiG-SAM($\delta = 1$)');
plot(time_vec3_1,f_vec3_1,'--','DisplayName','BiG-SAM($\delta = 0.01$)');
plot(time_vec2,f_vec2,'--','DisplayName','a-IRG');
plot(time_vec6,f_vec6,'--','DisplayName','Bi-SG');

ylabel('$F(x_k)$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast','FontSize',10);
pbaspect([1 0.8 1])

%% lower and upper gaps
lower_value = [g_vec(end);g_vecg1(end);g_vecstr(end);g_vecg1str(end);g_vec3(end);g_vec3_1(end);g_vec2(end);g_vec6(end)];
lower_value_gap = [g_vec(end)-gstar;g_vecg1(end)-gstar;g_vecstr(end)-gstar;g_vecg1str(end)-gstar;g_vec3(end)-gstar;g_vec3_1(end)-gstar;g_vec2(end)-gstar;g_vec6(end)-gstar];

fprintf('Lower-level value\n');
fprintf('%.4e\n',lower_value);
fprintf('***************************\n')
fprintf('Lower-level optimal gap\n');
fprintf('%.4e\n',lower_value_gap);

upper_value = [f_vec(end);f_vecg1(end);f_vecstr(end);f_vecg1str(end);f_vec3(end);f_vec3_1(end);f_vec2(end);f_vec6(end)];
upper_value_gap = [f_vec(end)-fstar_fmincon;f_vecg1(end)-fstar_fmincon;f_vecstr(end)-fstar_fmincon;f_vecg1str(end)-fstar_fmincon;f_vec3(end)-fstar_fmincon;f_vec3_1(end)-fstar_fmincon;f_vec2(end)-fstar_fmincon;f_vec6(end)-fstar_fmincon];

fprintf('Upper-level value\n');
fprintf('%.4e\n',upper_value);
fprintf('***************************\n')
fprintf('Upper-level optimal gap\n');
fprintf('%.4e\n',upper_value_gap);

iter_vec = [iter;iterg1;iterstr;iterg1str];
fprintf('%d\n',iter_vec);


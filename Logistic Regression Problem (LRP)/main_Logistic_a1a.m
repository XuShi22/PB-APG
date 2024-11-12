%%
% min ||x||^2_2/2
%s.t. argmin logistic + 1norm ball
clear;clc;close all;
format long


%% load data
seed = 123456;
rng(seed);
addpath('./libsvm-3.3/matlab/'); % use libsvmread
[y, X] = libsvmread('a1a.t');
X = normalize(X,'range');
A = full(X); b = full(y);
% [m,n]= size(A);
A2=A(1:1000,:);A2 = [A2,ones(1000,1)];
b2=b(1:1000);
[m,n]= size(A2);
lambda_g = 10;
x0 = rand(n,1); % initial point
maxiter=1e6; % max iterations

%% function definition
% upper level function
fun_f= @(x) x'*x/2;
% lower level function
fun_g = @(x) (1/m)*ones(1,m)*log(1+exp(-(A2*x).*b2));
fun_h = @(x) 0;

% upper level gradient
grad_f= @(x) x;
% lower level gradient
grad_g = @(x) -(1/m)*A2'*(b2.*(ones(m,1) - 1./(1+exp(-(A2*x).*b2))));
% upper level lipschitz
L_f=1;
% lower level lipschitz
L_g=(0.25/m)*(norm(full(A2'*A2),2));

% Finding optimal solution
% lower level optimal: FISTA
opts.L_f = L_f;
opts.L_g = L_g;
opts.L0 = L_g;
opts.tol = 1e-16; % lower-level tolerance
proxh = @(x) ProjectOntoL1Ball(x,lambda_g);
[x_fista, gstar_fista, ~, ~] = Greedy_FISTA_init(fun_g, fun_h, grad_g, proxh, x0, opts, fun_f, fun_g);
norm(x_fista,1)

% total optimal: fmincon
f = @(x) f_logistic(x); g = @(x) g_logistic(x,A2,b2,gstar_fista,lambda_g);
A=[];b=[];Aeq=[];beq=[];lb=[];ub=[];
options = optimoptions('fmincon','Display','iter','Algorithm','active-set','EnableFeasibilityMode',true, 'HessianApproximation','lbfgs', 'SpecifyObjectiveGradient',true, 'MaxFunctionEvaluations',10e6,'MaxIterations',10e6,'ConstraintTolerance',1e-13,'FunctionTolerance',1e-13,'OptimalityTolerance',1e-13,'StepTolerance',1e-13);
[x_fmincon_log,fval,exitflag,output,lambda_fmin,grad] = fmincon(f,x0,A,b,Aeq,beq,lb,ub,g,options);
fstar = fun_f(x_fmincon_log);
fun_g(x_fmincon_log)-gstar_fista
norm(x_fmincon_log,1)
fprintf('\n');


%% PB-APG
epsilon = 1e-10; %termination criterion, can modify
gamma_tot = 100000; %penalty parameter, can modify
gamma1 = gamma_tot; %penalty parameter
L = L_f + gamma1*L_g;
opts.L_f = L_f;
opts.L_g = L_g;
opts.L0 = L;
% opts.lambda = lambda;
opts.tol = epsilon; % tolerance
fun = @(x) x'*x/2 + gamma1*(1/m)*ones(1,m)*log(1+exp(-(A2*x).*b2));
grad = @(x) x - gamma1*(1/m)*A2'*(b2.*(ones(m,1) - 1./(1+exp(-(A2*x).*b2))));
fun_h = @(x) 0;
proxh = @(x) ProjectOntoL1Ball(x,lambda_g);
opts.a = @(x) 1;

[x_fista1, gstar_fista1, f_vec, g_vec, time_vec, iter] = Greedy_FISTA_init(fun, fun_h, grad, proxh, x0, opts, fun_f, fun_g);
f_vec = [fun_f(x0);f_vec]; g_vec = [fun_g(x0);g_vec]; time_vec = [0;time_vec];

fun_f(x_fista1) - fun_f(x_fmincon_log)
fun_g(x_fista1) - gstar_fista
norm(x_fista1,1)
fprintf('total iterations are %d\n',iter);


%% aPB-APG
gamma = gamma_tot/(20^5); %penalty parameter
f_vecg1 = [fun_f(x0)]; g_vecg1 = [fun_g(x0)]; time_vecg1 = [0];
fold = 5;
x00 = x0;
tol_vec = [1e-6;1e-7;1e-8;1e-9;1e-10]; %termination criterion vector for 1e-10, can modify
% tol_vec = [1e-0;1e-1;1e-2;1e-3;1e-4]; %termination criterion vector for 1e-4, can modify
% tol_vec = [1e-3;1e-4;1e-5;1e-6;1e-7]; %termination criterion vector for 1e-7, can modify

iterg1 = 0;
opts.a = @(x) 1;
for i = 1:fold
    gamma1 = gamma*(20^i);
    L = L_f + gamma1*L_g;
    opts.L_f = L_f;
    opts.L_g = L_g;
    opts.L0 = L;
    opts.tol = tol_vec(i); % tolerance
    fun = @(x) (x'*x)/2 + gamma1*(1/m)*ones(1,m)*log(1+exp(-(A2*x).*b2));
    grad = @(x) x - gamma1*(1/m)*A2'*(b2.*(ones(m,1) - 1./(1+exp(-(A2*x).*b2))));
    fun_h = @(x) 0;
    proxh = @(x) ProjectOntoL1Ball(x,lambda_g);
    [x_fista1g11, gstar_fista1g11, f_vecg11, g_vecg11, time_vecg11, iterg11] = Greedy_FISTA_init(fun, fun_h, grad, proxh, x00, opts, fun_f, fun_g);

    x00 = x_fista1g11;
    f_vecg1 = [f_vecg1;f_vecg11];
    g_vecg1 = [g_vecg1;g_vecg11];
    time_vecg11 = time_vecg11 + time_vecg1(end);
    time_vecg1 = [time_vecg1;time_vecg11];
    
    iterg1 = iterg1 + iterg11;
end

fun_f(x_fista1g11) - fun_f(x_fmincon_log)
fun_g(x_fista1g11) - gstar_fista
norm(x_fista1g11,1)
fprintf('total iterations are %d\n',iterg1);


%% PB-APG-sc
% epsilon = 1e-10; %termination criterion, can modify
gamma1 = gamma_tot; %penalty parameter
L = L_f + gamma1*L_g;
opts.L_f = L_f;
opts.L_g = L_g;
opts.L0 = L;
opts.tol = epsilon; % tolerance
fun = @(x) x'*x/2 + gamma1*(1/m)*ones(1,m)*log(1+exp(-(A2*x).*b2));
grad = @(x) x - gamma1*(1/m)*A2'*(b2.*(ones(m,1) - 1./(1+exp(-(A2*x).*b2))));
fun_h = @(x) 0;
proxh = @(x) ProjectOntoL1Ball(x,lambda_g);

opts.a = @(x) (sqrt(L) - sqrt(1))/(sqrt(L) + sqrt(1));

[x_fista1str, gstar_fista1str, f_vecstr, g_vecstr, time_vecstr, iterstr] = Greedy_FISTA_str(fun, fun_h, grad, proxh, x0, opts, fun_f, fun_g);
f_vecstr = [fun_f(x0);f_vecstr]; g_vecstr = [fun_g(x0);g_vecstr]; time_vecstr = [0;time_vecstr];

fun_f(x_fista1str) - fun_f(x_fmincon_log)
fun_g(x_fista1str) - gstar_fista
norm(x_fista1str,1)
fprintf('total iterations are %d\n',iterstr);


%% aPB-APG-sc
gamma = gamma_tot/(20^5); %penalty parameter
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
    fun = @(x) (x'*x)/2 + gamma1*(1/m)*ones(1,m)*log(1+exp(-(A2*x).*b2));
    grad = @(x) x - gamma1*(1/m)*A2'*(b2.*(ones(m,1) - 1./(1+exp(-(A2*x).*b2))));
    fun_h = @(x) 0;
    proxh = @(x) ProjectOntoL1Ball(x,lambda_g);
    opts.a = @(x) (sqrt(L) - sqrt(1))/(sqrt(L) + sqrt(1));

    [x_fista1g11str, gstar_fista1g11str, f_vecg11str, g_vecg11str, time_vecg11str, iterg11str] = Greedy_FISTA_init(fun, fun_h, grad, proxh, x00, opts, fun_f, fun_g);

    x00 = x_fista1g11str;
    f_vecg1str = [f_vecg1str;f_vecg11str];
    g_vecg1str = [g_vecg1str;g_vecg11str];
    time_vecg11str = time_vecg11str + time_vecg1str(end);
    time_vecg1str = [time_vecg1str;time_vecg11str];
    
    iterg1str = iterg1str + iterg11str;
end

fun_f(x_fista1g11str) - fun_f(x_fmincon_log)
fun_g(x_fista1g11str) - gstar_fista
norm(x_fista1g11str,1)
fprintf('total iterations are %d\n',iterg1str);

% find max time
tt = [time_vec(end),time_vecg1(end),time_vecstr(end),time_vecg1str(end)];
max_time = max(tt);

plus_time = 0.15;


%% CG for the sub problem
param.L_g = L_g;
param.L_f = L_f;
param.lam1=lambda_g;
param.maxiter=5e5;
param.epsilonf = 1e-4;
param.epsilong = 1e-4;
param.tol = 1e-8;

param.epsilong = 1e-2;
param.lam1=lambda_g;
param.maxiter=1e5;
tic;
[x_CG , f_hist] = CG_lowerlevel(fun_g,grad_g,x0,param);
time_init = toc;


%% CG-BiO algorithm
param.epsilonf = 1e-4;
param.epsilong = 1e-4;
param.lam=lambda_g;
param.fun_g_x0 = fun_g(x_CG);
param.maxiter=1e6;
param.maxtime = max_time + plus_time;
param.inittime = time_init;
disp('CG-BiO starts');
[f_vec1,g_vec1,time_vec1,x] = CG_BiO(fun_f,grad_f,grad_g,fun_g,param,x_CG);
f_vec1 = [fun_f(x0);fun_f(x_CG);f_vec1];g_vec1 = [fun_g(x0);fun_g(x_CG);g_vec1];
disp('CG-BiO Achieved!');
time_vec1 = [0;time_init;time_vec1];


%% a-IRG Algorithm
param.maxtime = max_time + plus_time;
param.maxiter=1e6;
disp('a-IRG Algorithm starts')
[f_vec2,g_vec2,time_vec2,xlast] = Alg_Projection(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec2 = [fun_f(x0);f_vec2];g_vec2 = [fun_g(x0);g_vec2];time_vec2 = [0;time_vec2];
disp('a-IRG Solution Achieved!');


%% BiG-SAM Algorithm
param.eta_g = 1/L_g;
param.eta_f = 1/L_f;
param.gamma = 10;
param.maxiter=1e6;
param.maxtime = max_time + plus_time;
disp('BiG-SAM Algorithm starts');
[f_vec3,g_vec3,time_vec3,xlast3] = BigSAM(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec3 = [fun_f(x0);f_vec3];g_vec3 = [fun_g(x0);g_vec3];time_vec3 = [0;time_vec3];
disp('BiG-SAM Solution Achieved!');


%% MNG
param.maxtime = max_time + plus_time;
param.maxiter=1e6;
param.M = L_g;
disp('Mininum norm gradient Algorithm starts');
[f_vec4,g_vec4,time_vec4,xlast4] = MNG(fun_f,grad_f,fun_g,grad_g,param,zeros(n,1));
f_vec4 = [fun_f(zeros(n,1));f_vec4];g_vec4 = [fun_g(zeros(n,1));g_vec4];time_vec4 = [0;time_vec4];
disp('MNG Solution Achieved!')


%% DBGD
param.alpha = 1;
param.beta = 1;
param.stepsize = 1/L_g;
param.maxiter=1e6;
param.maxtime = max_time + plus_time;
disp('DBGD Algorithm starts');
[f_vec5,g_vec5,time_vec5,xlast5] = DBGD(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec5 = [fun_f(x0);f_vec5];g_vec5 = [fun_g(x0);g_vec5];time_vec5 = [0;time_vec5];
disp('DBGD Solution Achieved!');


%% Bi-SG Algorithm
param.eta_g = 1/L_g;
param.maxiter=1e6;
param.maxtime = max_time + plus_time;
disp('Bi-SG Algorithm starts');
[f_vec6,g_vec6,time_vec6,xlast6] = Bi_SG(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec6 = [fun_f(x0);f_vec6];g_vec6 = [fun_g(x0);g_vec6];time_vec6 = [0;time_vec6];
disp('Bi-SG Solution Achieved!');


%% R-APM samadi2023
epsilon = 1e-10; %termination criterion of fista
param.L_f = L_f;
param.L_g = L_g;
param.maxiter = 1e6;
eta = 1/gamma_tot; %step-size
L = L_g + eta*L_f;
param.L0 = L;
param.tol = epsilon; % tolerance
param.maxtime = max_time + plus_time;
fun_R = @(x) (1/m)*ones(1,m)*log(1+exp(-(A2*x).*b2)) + eta*x'*x/2;
grad_R = @(x) - (1/m)*A2'*(b2.*(ones(m,1) - 1./(1+exp(-(A2*x).*b2)))) + eta*x;
fun_h_R = @(x) 0;
proxh_R = @(x) ProjectOntoL1Ball(x,lambda_g);
disp('R_APM Algorithm starts');
[~, ~, f_vec7, g_vec7, time_vec7, iter7] = R_APM(fun_R, fun_h_R, grad_R, proxh_R, x0, param, fun_f, fun_g);
f_vec7 = [fun_f(x0);f_vec7]; g_vec7 = [fun_g(x0);g_vec7]; time_vec7 = [0;time_vec7];
disp('R_APM Solution Achieved!');


%% Figures
%% lower-level gap
figure (1);
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',3)
set(gcf,'DefaultLineMarkerSize',4);
set(gcf,'Position',[331,167,591,586])

semilogy(time_vec,(g_vec-gstar_fista),'-','DisplayName','PB-APG','LineWidth',5);
hold on;
semilogy(time_vecg1,(g_vecg1-gstar_fista),'-','DisplayName','aPB-APG','LineWidth',6);
semilogy(time_vecstr,(g_vecstr-gstar_fista),'-','DisplayName','PB-APG-sc','LineWidth',4);
semilogy(time_vecg1str,(g_vecg1str-gstar_fista),'-','DisplayName','aPB-APG-sc','LineWidth',3);
semilogy(time_vec4,(g_vec4-gstar_fista),'--','DisplayName','MNG');
semilogy(time_vec3,(g_vec3-gstar_fista),'--','DisplayName','BiG-SAM');
semilogy(time_vec5,(g_vec5-gstar_fista),'--','DisplayName','DBGD');
semilogy(time_vec2,(g_vec2-gstar_fista),'--','DisplayName','a-IRG');
semilogy(time_vec1,(g_vec1-gstar_fista),'--','DisplayName','CG-BiO');
semilogy(time_vec6,(g_vec6-gstar_fista),'--','DisplayName','Bi-SG');
semilogy(time_vec7,(g_vec7-gstar_fista),'--','DisplayName','R-APM');

ylabel('$G(x_k)-G^*$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast','FontSize',8);
pbaspect([1 0.8 1])


%% upper-level
figure (2);
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',3)
set(gcf,'DefaultLineMarkerSize',4);
set(gcf,'Position',[331,167,591,586])

plot(time_vec,f_vec,'-','DisplayName','PB-APG','LineWidth',5);
hold on;
plot(time_vecg1,f_vecg1,'-','DisplayName','aPB-APG','LineWidth',6);
plot(time_vecstr,f_vecstr,'-','DisplayName','PB-APG-sc','LineWidth',4);
plot(time_vecg1str,f_vecg1str,'-','DisplayName','aPB-APG-sc','LineWidth',3);
plot(time_vec4,f_vec4,'--','DisplayName','MNG');
plot(time_vec3,f_vec3,'--','DisplayName','BiG-SAM');
plot(time_vec5,f_vec5,'--','DisplayName','DBGD');
plot(time_vec2,f_vec2,'--','DisplayName','a-IRG');
plot(time_vec1,f_vec1,'--','DisplayName','CG-BiO');
plot(time_vec6,f_vec6,'--','DisplayName','Bi-SG');
plot(time_vec7,f_vec7,'--','DisplayName','R-APM');

ylabel('$F(x_k)$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast','FontSize',8);
pbaspect([1 0.8 1])


%% lower and upper gap
lower_value = [g_vec(end);g_vecg1(end);g_vecstr(end);g_vecg1str(end);g_vec4(end);g_vec3(end);g_vec5(end);g_vec2(end);g_vec1(end);g_vec6(end);g_vec7(end)];
lower_value_gap = [g_vec(end)-gstar_fista;g_vecg1(end)-gstar_fista;g_vecstr(end)-gstar_fista;g_vecg1str(end)-gstar_fista;g_vec4(end)-gstar_fista;g_vec3(end)-gstar_fista;g_vec5(end)-gstar_fista;g_vec2(end)-gstar_fista;g_vec1(end)-gstar_fista;g_vec6(end)-gstar_fista;g_vec7(end)-gstar_fista];

fprintf('Lower-level value\n');
fprintf('%.4e\n',lower_value);
fprintf('***************************\n')
fprintf('Lower-level optimal gap\n');
fprintf('%.4e\n',lower_value_gap);

upper_value = [f_vec(end);f_vecg1(end);f_vecstr(end);f_vecg1str(end);f_vec4(end);f_vec3(end);f_vec5(end);f_vec2(end);f_vec1(end);f_vec6(end);f_vec7(end)];
upper_value_gap = [f_vec(end)-fstar;f_vecg1(end)-fstar;f_vecstr(end)-fstar;f_vecg1str(end)-fstar;f_vec4(end)-fstar;f_vec3(end)-fstar;f_vec5(end)-fstar;f_vec2(end)-fstar;f_vec1(end)-fstar;f_vec6(end)-fstar;f_vec7(end)-fstar];

fprintf('Upper-level value\n');
fprintf('%.4e\n',upper_value);
fprintf('***************************\n')
fprintf('Upper-level optimal gap\n');
fprintf('%.4e\n',upper_value_gap);

iter_vec = [iter;iterg1;iterstr;iterg1str];
fprintf('%d\n',iter_vec);


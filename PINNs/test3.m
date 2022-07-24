clear,clc

tic
%% initail parameters
mu = 250;
b = 4;
theta = 5;
c_x = 58;
c_p = 110;
nu = 0;
y0 = 2000;
gama = 4;
f_0 = 0.694;
c_f = 43.9;
beta = 2;
eps = 0.2;
%% Lagrangian
f = @(x,y) -(f_0*x^beta*(y0 - y + mu*(nu - theta*(x^gama/(c_x^gama + x^gama) - 1) + (b*x^gama)/(c_p^gama + x^gama) - 1)))/(c_f^beta + x^beta);
g = @(x,y) y0 - y + mu*(nu - theta*(x^gama/(c_x^gama + x^gama) - 1) - (b*x^gama)/(c_p^gama + x^gama) + 1);
 
fx = @(x,y) (f_0*mu*x^beta*(theta*((gama*x^(gama - 1))/(c_x^gama + x^gama) - (gama*x^gama*x^(gama - 1))/(c_x^gama + x^gama)^2) - (b*gama*x^(gama - 1))/(c_p^gama + x^gama) + (b*gama*x^gama*x^(gama - 1))/(c_p^gama + x^gama)^2))/(c_f^beta + x^beta) - (beta*f_0*x^(beta - 1)*(y0 - y + mu*(nu - theta*(x^gama/(c_x^gama + x^gama) - 1) + (b*x^gama)/(c_p^gama + x^gama) - 1)))/(c_f^beta + x^beta) + (beta*f_0*x^beta*x^(beta - 1)*(y0 - y + mu*(nu - theta*(x^gama/(c_x^gama + x^gama) - 1) + (b*x^gama)/(c_p^gama + x^gama) - 1)))/(c_f^beta + x^beta)^2;
fxx = @(x,y)  (2*beta^2*f_0*x^(2*beta - 2)*(y0 - y + mu*(nu - theta*(x^gama/(c_x^gama + x^gama) - 1) + (b*x^gama)/(c_p^gama + x^gama) - 1)))/(c_f^beta + x^beta)^2 - (f_0*mu*x^beta*(theta*((2*gama^2*x^(2*gama - 2))/(c_x^gama + x^gama)^2 - (gama*x^(gama - 2)*(gama - 1))/(c_x^gama + x^gama) - (2*gama^2*x^gama*x^(2*gama - 2))/(c_x^gama + x^gama)^3 + (gama*x^gama*x^(gama - 2)*(gama - 1))/(c_x^gama + x^gama)^2) - (2*b*gama^2*x^(2*gama - 2))/(c_p^gama + x^gama)^2 + (b*gama*x^(gama - 2)*(gama - 1))/(c_p^gama + x^gama) + (2*b*gama^2*x^gama*x^(2*gama - 2))/(c_p^gama + x^gama)^3 - (b*gama*x^gama*x^(gama - 2)*(gama - 1))/(c_p^gama + x^gama)^2))/(c_f^beta + x^beta) - (beta*f_0*x^(beta - 2)*(beta - 1)*(y0 - y + mu*(nu - theta*(x^gama/(c_x^gama + x^gama) - 1) + (b*x^gama)/(c_p^gama + x^gama) - 1)))/(c_f^beta + x^beta) - (2*beta^2*f_0*x^beta*x^(2*beta - 2)*(y0 - y + mu*(nu - theta*(x^gama/(c_x^gama + x^gama) - 1) + (b*x^gama)/(c_p^gama + x^gama) - 1)))/(c_f^beta + x^beta)^3 + (2*beta*f_0*mu*x^(beta - 1)*(theta*((gama*x^(gama - 1))/(c_x^gama + x^gama) - (gama*x^gama*x^(gama - 1))/(c_x^gama + x^gama)^2) - (b*gama*x^(gama - 1))/(c_p^gama + x^gama) + (b*gama*x^gama*x^(gama - 1))/(c_p^gama + x^gama)^2))/(c_f^beta + x^beta) + (beta*f_0*x^beta*x^(beta - 2)*(beta - 1)*(y0 - y + mu*(nu - theta*(x^gama/(c_x^gama + x^gama) - 1) + (b*x^gama)/(c_p^gama + x^gama) - 1)))/(c_f^beta + x^beta)^2 - (2*beta*f_0*mu*x^beta*x^(beta - 1)*(theta*((gama*x^(gama - 1))/(c_x^gama + x^gama) - (gama*x^gama*x^(gama - 1))/(c_x^gama + x^gama)^2) - (b*gama*x^(gama - 1))/(c_p^gama + x^gama) + (b*gama*x^gama*x^(gama - 1))/(c_p^gama + x^gama)^2))/(c_f^beta + x^beta)^2;
fyx = @(x,y) (beta*f_0*x^(beta - 1))/(c_f^beta + x^beta) - (beta*f_0*x^beta*x^(beta - 1))/(c_f^beta + x^beta)^2;
fy = @(x,y) (f_0*x^beta)/(c_f^beta + x^beta);

gx = @(x,y) -mu*(theta*((gama*x^(gama - 1))/(c_x^gama + x^gama) - (gama*x^gama*x^(gama - 1))/(c_x^gama + x^gama)^2) + (b*gama*x^(gama - 1))/(c_p^gama + x^gama) - (b*gama*x^gama*x^(gama - 1))/(c_p^gama + x^gama)^2);
gy = @(x,y) -1;
gxy = @(x,y) 0;
gyy = @(x,y) 0;

s = @(x,y) (f_0*eps*x^beta)/(c_f^beta + x^beta);
sx = @(x,y) (beta*eps*f_0*x^(beta - 1))/(c_f^beta + x^beta) - (beta*eps*f_0*x^beta*x^(beta - 1))/(c_f^beta + x^beta)^2;
sxx = @(x,y) (beta*eps*f_0*x^(beta - 2)*(beta - 1))/(c_f^beta + x^beta) - (2*beta^2*eps*f_0*x^(2*beta - 2))/(c_f^beta + x^beta)^2 + (2*beta^2*eps*f_0*x^beta*x^(2*beta - 2))/(c_f^beta + x^beta)^3 - (beta*eps*f_0*x^beta*x^(beta - 2)*(beta - 1))/(c_f^beta + x^beta)^2;
sxxx = @(x,y) (6*beta^3*eps*f_0*x^(beta - 1)*x^(2*beta - 2))/(c_f^beta + x^beta)^3 - (2*beta^2*eps*f_0*x^(2*beta - 3)*(2*beta - 2))/(c_f^beta + x^beta)^2 + (2*beta^2*eps*f_0*x^beta*x^(2*beta - 3)*(2*beta - 2))/(c_f^beta + x^beta)^3 + (beta*eps*f_0*x^(beta - 3)*(beta - 1)*(beta - 2))/(c_f^beta + x^beta) - (6*beta^3*eps*f_0*x^beta*x^(beta - 1)*x^(2*beta - 2))/(c_f^beta + x^beta)^4 - (2*beta^2*eps*f_0*x^(beta - 1)*x^(beta - 2)*(beta - 1))/(c_f^beta + x^beta)^2 - (beta*eps*f_0*x^beta*x^(beta - 3)*(beta - 1)*(beta - 2))/(c_f^beta + x^beta)^2 + (2*beta^2*eps*f_0*x^beta*x^(beta - 1)*x^(beta - 2)*(beta - 1))/(c_f^beta + x^beta)^3;


L = @(x,y,u,v) (u-f(x,y)-0.5*mu^2*s(x,y)*sx(x,y))^2/(mu*s(x,y))^2 + (v-g(x,y))^2/mu^2 + fx(x,y) + mu^2*(s(x,y)*sxx(x,y)+sx(x,y)^2)/2 + gy(x,y) -(f(x,y)+mu^2*sx(x,y)*s(x,y)/2)*sx(x,y)/s(x,y) ;


%% Compute the action functional
% T = 4;
n = 200;
for i = 1 : n
    filename = ['C:\Users\hujianyu\Desktop\test3\result\u_opt',num2str(i-1),'.txt'];
    u = load(filename);
    filename = ['C:\Users\hujianyu\Desktop\test3\result\v_opt',num2str(i-1),'.txt'];
    v = load(filename);
    T = 1 + (i-1)*0.05;
    Act(i) = ActionValue(u,v,L,T);
end

%% Find the minimizer of the action functional
ind = find(Act==min(min(Act)))
% ind = find(Act==max(max(Act)));
j = (ind-1);

u_L = (b-1)^(-1/gama)*c_p;
v_L = y0+mu*(theta+nu-theta*c_p^gama/((b-1)*c_x^gama+c_p^gama));


filename = ['C:\Users\hujianyu\Desktop\test3\result\u_opt',num2str(j),'.txt'];
u = load(filename);
filename = ['C:\Users\hujianyu\Desktop\test3\result\v_opt',num2str(j),'.txt'];
v = load(filename);
L = load('LimitCycle_nu=0.txt');

% t = 1.05 : 0.05 : 11;
% plot(t,Act,'b');hold on
% plot(1+ind*0.05,Act(ind),'o')
figure
plot(u,v,'m-','Linewidth',2); hold on
plot(L(1,:),L(2,:),'r-','Linewidth',2); hold on 
plot(u_L,v_L,'o'); hold on
plot(L(1,1722),L(2,1722),'o')
xlabel('$c$', 'interpreter', 'latex', 'fontsize', 16);
ylabel('$w$', 'interpreter', 'latex', 'fontsize', 16);
% plot(L(1,j+1),L(2,j+1),'o');

A = Act(ind);
P(1,1) = u(end);
P(1,2) = v(end);
% plot(L(1,j+1),L(2,j+1),'o');
T = 1 + n*0.05;
save path_0 u v A P
% saves 
% print('-dpng','act.fig')

figure
filename = ['C:\Users\hujianyu\Desktop\test3\result\loss',num2str(j),'.txt'];
loss = load(filename);
m = length(loss);
TT = 4*10^5;
t = 0 : TT/(m-1) : TT;
plot(t,log10(loss),'b-','Linewidth',2)
xlabel('$\mbox{Number of Iterations}$', 'interpreter', 'latex', 'fontsize', 16);
ylabel('$\mbox{Order of Loss}$', 'interpreter', 'latex', 'fontsize', 16);

figure
l = 1 : (T-1)/(n-1) : T;
plot(l,Act,'g-','Linewidth',2)
xlabel('$\mbox{Transition Time}$', 'interpreter', 'latex', 'fontsize', 16);
ylabel('$\mbox{Action Value}$', 'interpreter', 'latex', 'fontsize', 16);


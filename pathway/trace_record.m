%% carbon cycle sysytem
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

%% functions 
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
% ga = @(x,y) mu;

%% stable point
u_L = (b-1)^(-1/gama)*c_p;
v_L = y0+mu*(theta+nu-theta*c_p^gama/((b-1)*c_x^gama+c_p^gama));

%% 
dis=@(x,y,x1,y1) ((x-x1)^2+(y-y1)^2)^(1/2);
T = 4;
N = 10^3;
dt = T/N;
t = 0 : dt : T;
F = @(x,y,u,v)   y*(fx(x,u)+(eps*mu)^2/2*(sxx(x,u)*s(x,u)+sx(x,u)^2))+v*fy(x,u)+2*sx(x,u)/s(x,u)*y*(y-f(x,u)-(eps*mu)^2/2*sx(x,u)*s(x,u))-sx(x,u)/s(x,u)*(y-f(x,u)-(eps*mu)^2/2*sx(x,u)*s(x,u))^2-(y-f(x,u)-(eps*mu)^2/2*sx(x,u)*s(x,u))*(fx(x,u)+(eps*mu)^2/2*(sxx(x,u)*s(x,u)+sx(x,u)^2))-s(x,u)^2*(v-g(x,u))*gx(x,u)+(eps*mu)^2*s(x,u)^2/2*(fxx(x,u)+(eps*mu)^2/2*sxxx(x,u)*s(x,u)+3*(eps*mu)^2/2*sxx(x,u)*sx(x,u)+gxy(x,u)-sx(x,u)/s(x,u)*(fx(x,u)+(eps*mu)^2/2*(sxx(x,u)*s(x,u)+sx(x,u)^2))-(sxx(x,u)*s(x,u)-sx(x,u)^2/(s(x,u)^2))*(f(x,u)+(eps*mu)^2/2*s(x,u)*sx(x,u)));
G = @(x,y,u,v)   gx(x,u)*y+gy(x,u)*v-s(x,u)^(-2)*(y-f(x,u)-(eps*mu)^2/2*sx(x,u)*s(x,u))*fy(x,u)-(v-g(x,u))*gy(x,u)+0.5*(eps*mu)^2*(fyx(x,u)+gyy(x,u)-fy(x,u)*sx(x,u)/s(x,u));

fprintf("读取数据中...");

AA = load ('LimitCycle_nu=0.csv');
BB = load ('LimitCycle_nu=0_ans.csv');

fprintf("已完成\n");

[NN1,NN2] = size(AA);
x1=zeros(NN2,N+1);
x2=zeros(1,N+1);
x3=zeros(NN2,N+1);
x4=zeros(1,N+1);
num=0;
lst=zeros(1,0);
fprintf("NN1 NN2=%d %d\n",NN1,NN2)
for I=1:NN2
    if (mod(I,100)==0)
        fprintf("%d 计算结束\n",I)
    end
    a1 = AA(1,I);
    a2 = AA(2,I);
    v1 = BB(1,I);
    v2 = BB(2,I);
    x1(I,1) = u_L;
    x2(1) = v1;
    x3(I,1) = v_L;
    x4(1) = v2;
    for i = 1 : N
        x1(I,i+1) = x1(I,i) + x2(i)*dt;
        x2(i+1) = x2(i) + F(x1(I,i),x2(i),x3(I,i),x4(i))*dt;
        x3(I,i+1) = x3(I,i) + x4(i)*dt;
        x4(i+1) = x4(i) + G(x1(I,i),x2(i),x3(I,i),x4(i))*dt;
    end
end

fprintf("保存数据中...");

csvwrite('x1_trace.csv',x1)
csvwrite('x3_trace.csv',x3)

fprintf("已完成\n");

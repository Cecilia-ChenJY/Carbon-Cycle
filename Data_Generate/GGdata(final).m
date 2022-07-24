%% carbon cycle sysytem
clear,clc

tic
%% initail parameters
mu = 250;
b = 4;
theta = 5;
c_x = 58;
c_p = 110;
nu = 0.4;
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

%% main code
% u_L = (b-1)^(-1/gama)*c_p;
% v_L = y0+mu*(theta+nu-theta*c_p^gama/((b-1)*c_x^gama+c_p^gama));
T = 4;

F = @(x,y,u,v)   y*(fx(x,u)+(eps*mu)^2/2*(sxx(x,u)*s(x,u)+sx(x,u)^2))+v*fy(x,u)+2*sx(x,u)/s(x,u)*y*(y-f(x,u)-(eps*mu)^2/2*sx(x,u)*s(x,u))-sx(x,u)/s(x,u)*(y-f(x,u)-(eps*mu)^2/2*sx(x,u)*s(x,u))^2-(y-f(x,u)-(eps*mu)^2/2*sx(x,u)*s(x,u))*(fx(x,u)+(eps*mu)^2/2*(sxx(x,u)*s(x,u)+sx(x,u)^2))-s(x,u)^2*(v-g(x,u))*gx(x,u)+(eps*mu)^2*s(x,u)^2/2*(fxx(x,u)+(eps*mu)^2/2*sxxx(x,u)*s(x,u)+3*(eps*mu)^2/2*sxx(x,u)*sx(x,u)+gxy(x,u)-sx(x,u)/s(x,u)*(fx(x,u)+(eps*mu)^2/2*(sxx(x,u)*s(x,u)+sx(x,u)^2))-(sxx(x,u)*s(x,u)-sx(x,u)^2/(s(x,u)^2))*(f(x,u)+(eps*mu)^2/2*s(x,u)*sx(x,u)));
G = @(x,y,u,v)   gx(x,u)*y+gy(x,u)*v-s(x,u)^(-2)*(y-f(x,u)-(eps*mu)^2/2*sx(x,u)*s(x,u))*fy(x,u)-(v-g(x,u))*gy(x,u)+0.5*(eps*mu)^2*(fyx(x,u)+gyy(x,u)-fy(x,u)*sx(x,u)/s(x,u));

%% main code
u_L = (b-1)^(-1/gama)*c_p;
v_L = y0+mu*(theta+nu-theta*c_p^gama/((b-1)*c_x^gama+c_p^gama));
N = 50;
dud = 0 : 2*pi/N : 2*pi;
vec1 =cos(dud);
vec2 =sin(dud);
M = 10;
vel = 40;
NN = 10^3;
dt = T/NN;
t = 0 : dt : T;
dy = @(t, y)[y(2);
             F(y(1),y(2),y(3),y(4));
             y(4);
             G(y(1),y(2),y(3),y(4))];
tspan = [0,T];


pl=0;
P=zeros(5,M*(N+1)^2);
u = load('LimitCycle_nu=0.txt');
% plot(u(1,:),u(2,:),'r'); hold on
for k = 1 : M
    fprintf("k=%d;\n",k);
    out1 = (vel*k/M) *vec1;
    out2 = (vel*k/M)*vec2;
    for i = 1 : N+1
        for j = 1 : N+1
            y0 = [u_L out1(i) v_L out2(j)];
            [x,u2] = ode45(dy,tspan,y0);
            pl=pl+1;
            P(1,pl) =out1(i);
            P(2,pl) = out2(j);
            P(3,pl) = u2(end,1);
            P(4,pl) = u2(end,3);
            P(5,pl) = x(end);
        end
    end
end
% for k = M : M
%     out1 = rur(k)*vec1;
%     out2 = rur(k)*vec2;
% for i = 1 : N+1
%     for j = 1 : N+1            
%         x1(1) = u_L;
%         x2(1) = out1(i);
%         x3(1) = v_L;
%         x4(1) = out2(j);
%         for l = 1 : NN
%             x1(l+1) = x1(l) + x2(l)*dt;
%             x2(l+1) = x2(l) + F(x1(l),x2(l),x3(l),x4(l))*dt;
%             x3(l+1) = x3(l) + x4(l)*dt;
%             x4(l+1) = x4(l) + G(x1(l),x2(l),x3(l),x4(l))*dt;
%         end
%         P(1,(k-1)*(N+1)^2+(i-1)*(N+1)+j) = out1(i);
%         P(2,(k-1)*(N+1)^2+(i-1)*(N+1)+j) = out2(j);
%         P(3,(k-1)*(N+1)^2+(i-1)*(N+1)+j) = x1(end);
%         P(4,(k-1)*(N+1)^2+(i-1)*(N+1)+j) = x3(end);
%         plot(x1,x3,'-o');hold on
%     end
% end
% end

save Original4 P

toc
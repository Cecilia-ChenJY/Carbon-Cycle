function [s] = ActionValue(u,v,L,T)
%% Compute the action functional 
%% L: Lagrangian; u,v: paths; T: Time
%T = 4;
n = length(u);
dt = T/n;

s = 0;
for i = 2 : n-1
    s = s + L(u(i),v(i),(u(i+1)-u(i-1))/(2*dt),(v(i+1)-v(i-1))/(2*dt))*dt/2;
end
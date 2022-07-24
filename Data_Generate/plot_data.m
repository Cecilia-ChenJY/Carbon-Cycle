clear, clc
uL = load('LimitCycle_nu=0.8.txt');
PP = load('Original0.8.mat');
P = PP.P;
[m n] = size(P);
for i = 1 : n
    if P(4,i) > 2000 & P(3,i) < 200 & P(4,i)<4000
    plot(P(3,i),P(4,i),'*');hold on
    end
end
plot(uL(1,:),uL(2,:));
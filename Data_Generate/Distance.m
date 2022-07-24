function d = Distance(X,Y)
%% X: a input point 
%% Y: a set of points

u = X;
v = Y;
[m n] = size(v);
d = 10^3;
for i = 1 : n
    aa = 0;
    for j = 1 : m
        aa = aa + (u(j)-v(j,i))^2;
    end
    D(i) = sqrt(aa);
    if D(i)<d
        d = D(i);
    end
end


% d = min(D);
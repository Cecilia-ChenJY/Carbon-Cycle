clear,clc

fprintf("读取数据中...");
x1 = load ('x1_trace.csv');
x3 = load ('x3_trace.csv');
fprintf("已完成\n");
index=[2390:2393,2509:2512,2672:2676,2751:3143,3284:3285,3356:3357];
x1n=x1(index,:);
x3n=x3(index,:);
% csvwrite('u_opt.csv',x1n)
% csvwrite('v_opt.csv',x1n)
save('subdatau_opt.mat','x1n')
save('subdatav_opt.mat','x3n')
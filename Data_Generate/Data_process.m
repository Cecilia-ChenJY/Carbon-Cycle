%% compute the stable point
clear,clc
load('Original0.9.mat');
SED=P;
u = load('LimitCycle_nu=0.9.txt');
plot(u(1,:),u(2,:),'r'); hold on
% plot(SED(3,:),SED(4,:),'*');hold on

[~, n] = size(SED);
k = 0;
SDE11 = zeros(n,1);
for j = 1 : n
    if  min(sum((u-SED([3,4],j)).^2)) < 50
        k = k + 1;
        SDE11(k)=j;    
    end
end
SDE1 = SED([1,2,3,4],SDE11(1:k));
%plot(SDE1(3,:),SDE1(4,:),'*');  


l=10^3; %稀疏度>0，越大越密
SDE21=SDE1([3,4],:);
[~, n] = size(SDE21);
xmin=min(SDE21(1,:));xmax=max(SDE21(1,:));
ymin=min(SDE21(2,:));ymax=max(SDE21(2,:));
SDE21=SDE21-[xmin,ymin]';
SDE21=floor(SDE21./[(xmax-xmin)/l,(ymax-ymin)/l]');
[~,ia,~]=unique(SDE21',"rows","stable");
SDE2=SDE1(:,ia);
%save sparse0.mat SDE2
plot(SDE2(3,:),SDE2(4,:),'*');   


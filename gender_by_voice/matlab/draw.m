close all;clear;
load('voive_data.mat')
x=1:3168
y=v_d( : ,10);
figure(1)
plot(x,y);

stepnum=20;
maxnum=max(y);
minnum=min(y);
step=(maxnum-minnum)/stepnum;
l=length(y);
for i = 1:l
    y(i)=int32( (y(i)-minnum)/step+1 );
end
figure(2)
plot(x,y);
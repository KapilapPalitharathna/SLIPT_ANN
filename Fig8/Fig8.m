clear; clc; close all;

ANN_8_4 = cell2mat(struct2cell(load('Data Sets/ANN_Nt8_Ne4.mat'))); 
ANN_4_4 = cell2mat(struct2cell(load('Data Sets/ANN_Nt4_Ne4.mat')));
ANN_4_2 = cell2mat(struct2cell(load('Data Sets/ANN_Nt4_Ne2.mat'))); 

BF_8_4 = cell2mat(struct2cell(load('Data Sets/Bruteforce_Nt8_Ne4.mat'))); 
BF_4_4 = cell2mat(struct2cell(load('Data Sets/Bruteforce_Nt4_Ne4.mat')));
BF_4_2 = cell2mat(struct2cell(load('Data Sets/Bruteforce_Nt4_Ne2.mat'))); 

figure(1);
plot(-1,-1,'ko-');
hold on
plot(-1,-1,'kd-');
hold on
plot(-1,-1,'b-');
hold on
plot(-1,-1,'k-');
hold on
plot(-1,-1,'r-');
hold on

plot(0:2:30, BF_8_4,'bo-');
hold on;
plot(0:2:30, BF_4_4,'ko-');
hold on;
plot(0:2:30, BF_4_2,'ro-');
hold on;
plot(0:2:30, ANN_8_4,'bd-');
hold on;
plot(0:2:30, ANN_4_4,'kd-');
hold on;
plot(0:2:30, ANN_4_2,'rd-');
hold on;

grid on
xlabel('\sigma_{\beta}^2 [degrees]');
ylabel('Average Total DC Current [A]');
legend('Bruteforce','ANN', 'N_t = 8, N_e=4', 'N_t = 4, N_e=4', 'N_t = 4, N_e=2');
axis([0 30 7 15]);

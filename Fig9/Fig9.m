clear; clc; close all;


MSE_Path = cell2mat(struct2cell(load('Data Sets/MSE_Path_Prediction_High_Res.mat'))); 

IDC_2_2_4_No = cell2mat(struct2cell(load('Data Sets/IDC_Nt4_Nc2_Ne2_No.mat'))); 
IDC_4_4_4_Com = cell2mat(struct2cell(load('Data Sets/IDC_Nt4_Nc4_Ne4_Com.mat')));
IDC_4_4_4_No = cell2mat(struct2cell(load('Data Sets/IDC_Nt4_Nc4_Ne4_No.mat'))); 

IDC_2_2_8_No = cell2mat(struct2cell(load('Data Sets/IDC_Nt8_Nc2_Ne2_No.mat'))); 
IDC_4_4_8_Com = cell2mat(struct2cell(load('Data Sets/IDC_Nt8_Nc4_Ne4_Com.mat')));
IDC_4_4_8_No = cell2mat(struct2cell(load('Data Sets/IDC_Nt8_Nc4_Ne4_No.mat'))); 


figure(1);
plot(-1,-1,'k-');
hold on
plot(-1,-1,'k:');
hold on
plot(-1,-1,'ko');
hold on
plot(-1,-1,'kd');
hold on
plot(-1,-1,'ks');
hold on

plot(MSE_Path, IDC_2_2_8_No,'k-');
hold on;
plot(MSE_Path, IDC_4_4_8_Com,'k-');
hold on;
plot(MSE_Path, IDC_4_4_8_No,'k-');
hold on;
plot(MSE_Path, IDC_2_2_4_No,'k:');
hold on;
plot(MSE_Path, IDC_4_4_4_Com,'k:');
hold on;
plot(MSE_Path, IDC_4_4_4_No,'k:');
hold on;

plot(MSE_Path(1:4:end), IDC_2_2_8_No(1:4:end),'kd');
hold on;
plot(MSE_Path(1:4:end), IDC_4_4_8_Com(1:4:end),'ks');
hold on;
plot(MSE_Path(1:4:end), IDC_4_4_8_No(1:4:end),'ko');
hold on;
plot(MSE_Path(1:4:end), IDC_2_2_4_No(1:4:end),'kd');
hold on;
plot(MSE_Path(1:4:end), IDC_4_4_4_Com(1:4:end),'ks');
hold on;
plot(MSE_Path(1:4:end), IDC_4_4_4_No(1:4:end),'ko');
hold on;

grid on
xlabel('MSE of the Path Prediction [m^2]');
ylabel('Average error of I_{DC}, [A]');
legend('N_t = 8','N_t = 4', 'N_c = 4, N_e=4, no common user', 'N_c = 2, N_e=2, no common user', 'N_c = 4, N_e=4, common user');
axis([0.08 0.2 0.065 0.16]);
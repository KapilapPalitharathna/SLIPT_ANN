close all; clear; clc;

% Import data in Brute force method
BRUTE_IDC = load('Data Sets/IDC_optm_path_test_01.mat');
BRUTE_IDC = struct2array(BRUTE_IDC);
BRUTE_W = load('Data Sets/W_opt_path_test_01.mat');
BRUTE_W = struct2array(BRUTE_W);
BRUTE_IDC_W = [BRUTE_IDC, BRUTE_W];

% Import data proposed method (LSTM + Regression)
%REG_LSTM_IDC_W_temp = load('predicted_REG_IDC_W_OPT.dat');
REG_IDC_W = load('Data Sets/predicted_REG_IDC_W_OPT_without_pathPrediction.dat');

ZF_BF_IDC4 = load('Data Sets/I_DC_opt_ZF_BF_mod_05.mat');
ZF_BF_IDC4 = struct2array(ZF_BF_IDC4);
% MRT_BF_IDC4 = load('I_DC_opt_MRT_BF_mod_01.mat');
% MRT_BF_IDC4 = struct2array(MRT_BF_IDC4);

% Adjust IDC_W matrix after path predition (Fill)
REG_LSTM_IDC_W = REG_IDC_W;

U_step = size(BRUTE_IDC_W,1);
pos = 1:U_step;

figure(1);
subplot(2,2,1); plot(pos, BRUTE_IDC_W(:,1),'--g','LineWidth',1);
hold on;
plot(pos, REG_LSTM_IDC_W(:,1),'--k','LineWidth',1.5);
plot(pos, ZF_BF_IDC4(:,1),'--r','LineWidth',0.8);
% plot(pos, MRT_BF_IDC4(:,1),'--b','LineWidth',0.8);
hold off;
legend('Brute Force','Proposed method','ZF Beamforming','Location', 'Best');
xlabel('step number'); ylabel('DC current (A)');
title('DC current variation in transmitter 1');

subplot(2,2,2); plot(pos, BRUTE_IDC_W(:,2),'--g','LineWidth',1);
hold on;
plot(pos, REG_LSTM_IDC_W(:,2),'--k','LineWidth',1.5);
plot(pos, ZF_BF_IDC4(:,2),'--r','LineWidth',0.8);
% plot(pos, MRT_BF_IDC4(:,2),'--b','LineWidth',0.8);
hold off;
legend('Brute Force','Proposed method','ZF Beamforming','Location', 'Best');
xlabel('step number'); ylabel('DC current (A)');
title('DC current variation in transmitter 2');

subplot(2,2,3); plot(pos, BRUTE_IDC_W(:,3),'--g','LineWidth',1);
hold on;
plot(pos, REG_LSTM_IDC_W(:,3),'--k','LineWidth',1.5);
plot(pos, ZF_BF_IDC4(:,3),'--r','LineWidth',0.8);
% plot(pos, MRT_BF_IDC4(:,3),'--b','LineWidth',0.8);
hold off;
legend('Brute Force','Proposed method','ZF Beamforming','Location', 'Best');
xlabel('step number'); ylabel('DC current (A)');
title('DC current variation in transmitter 3');

subplot(2,2,4); plot(pos, BRUTE_IDC_W(:,4),'--g','LineWidth',1);
hold on;
plot(pos, REG_LSTM_IDC_W(:,4),'--k','LineWidth',1.5);
plot(pos, ZF_BF_IDC4(:,4),'--r','LineWidth',0.8);
% plot(pos, MRT_BF_IDC4(:,4),'--b','LineWidth',0.8);
hold off;
legend('Brute Force','Proposed method','ZF Beamforming','Location', 'Best');
xlabel('step number'); ylabel('DC current (A)');
title('DC current variation in transmitter 4');

% suptitle('DC current variation for each transmitters');


% Plot summation of IDC
BRUTE_IDC_W_sum = sum(BRUTE_IDC_W(:,1:4),2);
REG_LSTM_IDC_W_sum = sum(REG_LSTM_IDC_W(:,1:4),2);
ZF_BF_IDC_sum4 = sum(ZF_BF_IDC4(:,1:4),2);
% MRT_BF_IDC_sum4 = sum(MRT_BF_IDC4(:,1:4),2);

REG_LSTM_IDC_W_sum_test01 = [REG_LSTM_IDC_W_sum(1:69,:); REG_LSTM_IDC_W_sum(70:340,:)+0.2;...
    REG_LSTM_IDC_W_sum(341:end,:)];

figure(2);
plot(pos', BRUTE_IDC_W_sum,'-r','LineWidth',1.5);
hold on;
plot(pos', REG_LSTM_IDC_W_sum_test01,'-k','LineWidth',1.5);
plot(pos', ZF_BF_IDC_sum4,'-b','LineWidth',1.5);
% plot(pos', MRT_BF_IDC_sum4,'-r','LineWidth',1.5);

hold off;
grid on;
% xlim([0, 560]); ylim([6, 18]);
legend('Brute-force','Proposed method','ZF beamforming','Location', 'Best');
xlabel('Step number'); ylabel('DC current[A]');
xlim([1,560]); ylim([7,20]);
% title('Summation of DC current variation in transmitters');

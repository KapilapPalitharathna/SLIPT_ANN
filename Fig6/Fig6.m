close all; clear; clc;

% Import data in Brute force method
sum_cu_1_eu_i = load('Data Sets/sum_IDC_optm_path_cu_1_eu_i.mat');sum_cu_1_eu_i = struct2array(sum_cu_1_eu_i);
sum_cu_2_eu_i = load('Data Sets/sum_IDC_optm_path_cu_2_eu_i.mat');sum_cu_2_eu_i = struct2array(sum_cu_2_eu_i);
sum_cu_3_eu_i = load('Data Sets/sum_IDC_optm_path_cu_3_eu_i.mat');sum_cu_3_eu_i = struct2array(sum_cu_3_eu_i);
sum_cu_4_eu_i = load('Data Sets/sum_IDC_optm_path_cu_4_eu_i.mat');sum_cu_4_eu_i = struct2array(sum_cu_4_eu_i);

U_step = size(sum_cu_1_eu_i,1);
pos = 1:U_step;

figure(1);
plot(pos, sum_cu_1_eu_i(:,1),'-g','LineWidth',1);
hold on;
plot(pos, sum_cu_1_eu_i(:,2),'-k','LineWidth',1);
plot(pos, sum_cu_1_eu_i(:,3),'-m','LineWidth',1);
plot(pos, sum_cu_1_eu_i(:,4),'-r','LineWidth',1);
hold off;
lgd=legend('N_c=1, N_e=1,', 'N_c=1, N_e=2,', 'N_c=1, N_e=3', 'N_c=1, N_e=4','Location', 'Best');
%lgd.NumColumns = 2;
xlabel('Step number'); ylabel('Total DC current[A]');
xlim([1,560]); ylim([4,14]);
grid on;

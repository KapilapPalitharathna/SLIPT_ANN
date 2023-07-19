clear; clc; close all;

cu_1 = load('Data Sets/avg_sum_IDC_cu_1.mat'); cu_1_fin = struct2array(cu_1);
cu_2 = load('Data Sets/avg_sum_IDC_cu_2.mat'); cu_2_fin = struct2array(cu_2);
cu_3 = load('Data Sets/avg_sum_IDC_cu_3.mat'); cu_3_fin = struct2array(cu_3);
cu_4 = load('Data Sets/avg_sum_IDC_cu_4.mat'); cu_4_fin = struct2array(cu_4);

ZF_cu_1_eu_i = load('Data Sets/OPT_IDC_ZF_cu_1.mat'); ZF_cu_1_eu_i_fin = struct2array(ZF_cu_1_eu_i);
ZF_cu_2_eu_i = load('Data Sets/OPT_IDC_ZF_cu_2.mat'); ZF_cu_2_eu_i_fin = struct2array(ZF_cu_2_eu_i);
ZF_cu_3_eu_i = load('Data Sets/OPT_IDC_ZF_cu_3.mat'); ZF_cu_3_eu_i_fin = struct2array(ZF_cu_3_eu_i);
ZF_cu_4_eu_i = load('Data Sets/OPT_IDC_ZF_cu_4.mat'); ZF_cu_4_eu_i_fin = struct2array(ZF_cu_4_eu_i);

figure(1);
plot(1:size(cu_1_fin,2), cu_1_fin,'sk','LineWidth',0.8);
hold on;
plot(1:size(cu_2_fin,2), cu_2_fin,'>k','LineWidth',0.8);
plot(1:size(cu_3_fin,2), cu_3_fin,'ok','LineWidth',0.8);
plot(1:size(cu_4_fin,2), cu_4_fin,'xk','LineWidth',0.8);
plot(1:size(cu_1_fin,2), cu_1_fin,'-k','LineWidth',0.8);
plot(1:size(ZF_cu_1_eu_i_fin,2), ZF_cu_1_eu_i_fin,'--k','LineWidth',0.8);

plot(1:size(cu_1_fin,2), cu_1_fin,'-sk','LineWidth',0.8);
% hold on;
plot(1:size(cu_2_fin,2), cu_2_fin,'->k','LineWidth',0.8);
plot(1:size(cu_3_fin,2), cu_3_fin,'-ok','LineWidth',0.8);
plot(1:size(cu_4_fin,2), cu_4_fin,'-xk','LineWidth',0.8);

plot(1:size(ZF_cu_1_eu_i_fin,2), ZF_cu_1_eu_i_fin,'--sk','LineWidth',0.8);
plot(1:size(ZF_cu_2_eu_i_fin,2), ZF_cu_2_eu_i_fin,'-->k','LineWidth',0.8);
plot(1:size(ZF_cu_4_eu_i_fin,2), ZF_cu_4_eu_i_fin,'--ok','LineWidth',0.8);
plot(1:size(ZF_cu_3_eu_i_fin,2), ZF_cu_3_eu_i_fin,'--xk','LineWidth',0.8);

hold off;
grid on; set(gca,'GridLineStyle','--');
xlabel('Number of energy users, N_e'); ylabel('Average total DC current[A]');
legend('N_c = 1', 'N_c = 2', 'N_c = 3', 'N_c = 4',...
    'Proposed ANN', 'ZF beamforming', 'Location', 'Best');
% lgd = legend('N_c (Proposed) = 1,', 'N_c (Proposed) = 2,', 'N_c (Proposed) = 3,', 'N_c (Proposed) = 4,',...
%     'N_c (ZF) = 1', 'N_c (ZF) = 2', 'N_c (ZF) = 3', 'N_c (ZF) = 4','Location', 'Best');
% lgd.NumColumns = 2;
a2 = axes();
a2.Position = [.39 .16 0.26 0.5]; % xlocation, ylocation, xsize, ysize
plot(a2, cu_1_fin(4:7),'-sk','LineWidth',0.8);
hold on;
plot(a2, cu_2_fin(4:7),'->k','LineWidth',0.8);
plot(a2, cu_3_fin(4:7),'-ok','LineWidth',0.8);
plot(a2, cu_4_fin(4:7),'-xk','LineWidth',0.8);

plot(a2, ZF_cu_1_eu_i_fin(4:7),'--sk','LineWidth',0.8);
plot(a2, ZF_cu_2_eu_i_fin(4:7),'-->k','LineWidth',0.8);
plot(a2, ZF_cu_4_eu_i_fin(4:7),'--ok','LineWidth',0.8);
plot(a2, ZF_cu_3_eu_i_fin(4:7),'--xk','LineWidth',0.8); axis tight

annotation('rectangle',[.37 .69 .3 .22],'Color','k')

grid on; set(gca,'GridLineStyle','--');
% plot(a2,ZF_cu_1_eu_i_fin(4:7)); axis tight

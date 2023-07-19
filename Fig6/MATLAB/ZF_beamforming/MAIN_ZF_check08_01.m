close all; clear; clc;
tic;
load('Position_mat_Noise_var_0_03_test_01.mat');
U_step = size(R_CU_mat,2); % Number of user steps for CUs and EUs
VLC_settings_temp; % For all user position initializations

NumCU = 4; %size(R_CU,2); % Number of CUs ##############
NumT = size(T,2); % Number of Txs
idc_res = 8;

% I_DC current values for 2 txs [Real IU=0.6A]
iu_tx = 10.5;
IL_tx1 = 0.4; IU_tx1 = iu_tx; %Tx-1 current upper and lower bounds
IL_tx2 = 0.4; IU_tx2 = iu_tx; %Tx-2 current upper and lower bounds
IL_tx3 = 0.4; IU_tx3 = iu_tx; %Tx-3 current upper and lower bounds
IL_tx4 = 0.4; IU_tx4 = iu_tx; %Tx-4 current upper and lower bounds

% w_steps = 6;
W_MAX = (IU_tx1-IL_tx1)/2;
W_MIN = 0.001; %W_MAX/(NumT*NumCU);
P_IDC = 15; % W/A: The parameter to transform current to power

% EU energy threshold values
eu_th = -13.58731356201;%-13.954519154;%-13.969458;%EU 1 Energy Threshold values
EU1_th = eu_th; %EU 1 Energy Threshold values
EU2_th = eu_th; %EU 2 Energy Threshold values
EU3_th = eu_th; %EU 3 Energy Threshold values
EU4_th = eu_th; %EU 4 Energy Threshold values
EU5_th = eu_th; %EU 1 Energy Threshold values
EU6_th = eu_th; %EU 2 Energy Threshold values
EU7_th = eu_th; %EU 3 Energy Threshold values
EU8_th = eu_th; %EU 4 Energy Threshold values
EU9_th = eu_th; %EU 3 Energy Threshold values
EU10_th = eu_th; %EU 4 Energy Threshold values

EU_TH_mat = [EU1_th, EU2_th, EU3_th, EU4_th, EU5_th, EU6_th, EU7_th, EU8_th, EU9_th, EU10_th];

% CU rate threshold values 12.36099849128935 
cu_th = 2.650588239979590; %CU Rate Threshold values
CU1_th = cu_th; %CU 1 Rate Threshold values
CU2_th = cu_th; %CU 2 Rate Threshold values
CU3_th = cu_th; %CU 3 Rate Threshold values
CU4_th = cu_th; %CU 4 Rate Threshold values
CU5_th = cu_th; %CU 3 Rate Threshold values
CU6_th = cu_th; %CU 4 Rate Threshold values

opt_idc_zf_path_eu_i = [];
sum_opt_idc_zf_path_eu_i = [];
avg_sum_opt_idc_zf_path_eu_i = [];

for num_eu = 1:size(EU_TH_mat,2)
    
NumEU = num_eu;
EU_TH = EU_TH_mat(1,1:NumEU);

opt_idc_zf_path=[];
sum_opt_idc_zf_path=[];
avg_sum_opt_idc_zf_path=[];

for path_step = 1:1:U_step % Check for each steps

fprintf('Num_cu=%d, Num_eu=%d, Path number=%d \n',NumCU, num_eu, path_step); 
    
r_cu = R_CU_mat{1,path_step}; % get CU x and y coordinates
r_eu = R_EU_mat{1,path_step}; % get EU x and y coordinates
R_CU = [r_cu(:,2), r_cu(:,3), r_cu(:,1), r_cu(:,4)];
R_EU = [r_eu(:, 5), r_eu(:, 9), r_eu(:, 7:8), r_eu(:, 3:4), r_eu(:, 2), r_eu(:, 10), r_eu(:, 6), r_eu(:, 1)];
[H_CU_temp1, H_EU_temp1, U_blokage] = LOS_channelGain(T, R_CU, R_EU);% channel gain [Tx*Rx] - Cu and EU
H_CU_rx_tx = H_CU_temp1(1:NumCU, :);
H_CU_tx_rx = H_CU_rx_tx.';
H_EU = H_EU_temp1(1:NumEU, :);

% PRECODING MATRIX
inv_H = inv(H_CU_tx_rx'*H_CU_tx_rx)*H_CU_tx_rx';  %inv(H_CU);
% inv_H = inv_H.';
w_cu = [inv_H(1,:)/norm(inv_H(1,:)); inv_H(2,:)/norm(inv_H(2,:)); inv_H(3,:)/norm(inv_H(3,:)); inv_H(4,:)/norm(inv_H(4,:))]/NumCU;
min_w_cu = min(min(w_cu));
max_w_cu = max(max(w_cu));
W_CU_temp = interp1([min_w_cu, max_w_cu],[W_MIN, W_MAX],w_cu(:));
W_CU = reshape(W_CU_temp, NumCU, NumT);
w_mat_temp = W_CU/NumCU;

% Rate calculation
idc_zf_opt=[];
rate_rx=[];

for w_ind = 1:size(w_mat_temp,1)
    w_temp = w_mat_temp(w_ind,:);
    idc_temp = IL_tx1 + w_temp;
    P_idc_temp = P_IDC*idc_temp;
    h_rx_CU = H_CU_rx_tx(w_ind,:).*P_idc_temp;
    
    rate_temp = 0.5*log2(1 + (exp(1)/2*pi)*(((h_rx_CU*w_temp').^2)/noise_var));
    rate_rx = [rate_rx, rate_temp];
end

rate_logic_CU1 = rate_rx(:,1) >= CU1_th;
rate_logic_CU2 = rate_rx(:,2) >= CU2_th;
rate_logic_CU3 = rate_rx(:,3) >= CU3_th;
rate_logic_CU4 = rate_rx(:,4) >= CU4_th;
rate_logic = rate_logic_CU1 .* rate_logic_CU2 .* rate_logic_CU3 .* rate_logic_CU4;
    
if sum(rate_logic) ~= 0
    sum_w_mat_temp = sum(w_mat_temp,1);
    w_temp = sum_w_mat_temp;
    idc_temp_lower = IL_tx1 + w_temp;
    idc_temp_upper = IU_tx1 - w_temp;
    
    i_both_sides=[];
    for i_ind = 1:size(idc_temp_lower,2)
        i_s = idc_temp_lower(:,i_ind);
        i_e = idc_temp_upper(:,i_ind);
        i_s_e = linspace(i_s,i_e,idc_res)';
        i_both_sides = [i_both_sides, i_s_e];
    end
        i_all_combinations=[];
        for ind_tx1=1:size(i_both_sides,1)
            val_tx1 = i_both_sides(ind_tx1,1);

            for ind_tx2=1:size(i_both_sides,1)
                val_tx2 = i_both_sides(ind_tx2,2);

                for ind_tx3=1:size(i_both_sides,1)
                    val_tx3 = i_both_sides(ind_tx3,3);

                    for ind_tx4=1:size(i_both_sides,1)
                        val_tx4 = i_both_sides(ind_tx4,4);

                        val_txs = [val_tx1, val_tx2, val_tx3, val_tx4];
                        i_all_combinations = [i_all_combinations; val_txs];
                    end
                end
            end
        end
    
    for ind_idc_selected = 1:size(i_all_combinations,1)
        i_selected_temp = i_all_combinations(ind_idc_selected,:);
        P_IDC_temp = P_IDC.*i_selected_temp; % convert current -> Power
        
        E=[]; % Initialize Energy matrix
        for rx_EU = 1:size(H_EU,1) % Select each Eu one by one
            h_rx_EU = H_EU(rx_EU,:); % Extract necessary channel gains for the selected EU from 4 Txs
            h_P_DC = h_rx_EU.*P_IDC_temp; % calculate tempoary value = h*I_DC or h*P_DC
            E_temp = f*v_t.*sum(h_P_DC.*log(1 + h_P_DC./I_0),2); % Calculate Energy as in Eqn.
            E = [E, E_temp]; % Store them in E matrix [rax: for each current combinations, col: for each EU]
        end
        E_dB = 10*log10(E); % Convert to dB values
        EdB_logic = E_dB > EU_TH;  % Filtering E values from constraint

        if sum(EdB_logic) == size(H_EU,1)
            idc_zf_opt = [idc_zf_opt; i_selected_temp];
        end
    end
end

sum_idc_zf_opt = sum(idc_zf_opt,2);
[min_sum_idc_zf_opt, ind_min_sum_idc_zf_opt] = min(sum_idc_zf_opt);
opt_idc_zf = idc_zf_opt(ind_min_sum_idc_zf_opt,:);

if size(opt_idc_zf,1)==0
    opt_idc_zf=ones(1,NumT)*NaN;
end

opt_idc_zf_path = [opt_idc_zf_path; opt_idc_zf];
opt_idc_zf_path = rmmissing(opt_idc_zf_path);
sum_opt_idc_zf_path = sum(opt_idc_zf_path,2);
avg_sum_opt_idc_zf_path = mean(sum_opt_idc_zf_path);
end

% opt_idc_zf_path_eu_i = [opt_idc_zf_path_eu_i, opt_idc_zf_path];
% sum_opt_idc_zf_path_eu_i = [sum_opt_idc_zf_path_eu_i, sum_opt_idc_zf_path];
avg_sum_opt_idc_zf_path_eu_i = [avg_sum_opt_idc_zf_path_eu_i, avg_sum_opt_idc_zf_path];

end

save('OPT_IDC_ZF_cu_4', 'avg_sum_opt_idc_zf_path_eu_i', '-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;
tic;
load('Position_mat_Noise_var_0_03_test_01.mat');
U_step = size(R_CU_mat,2); % Number of user steps for CUs and EUs
VLC_settings_temp; % For all user position initializations

NumCU = 3; %size(R_CU,2); % Number of CUs ##############
NumT = size(T,2); % Number of Txs
idc_res = 8;

% I_DC current values for 2 txs [Real IU=0.6A]
iu_tx = 10.5;
IL_tx1 = 0.4; IU_tx1 = iu_tx; %Tx-1 current upper and lower bounds
IL_tx2 = 0.4; IU_tx2 = iu_tx; %Tx-2 current upper and lower bounds
IL_tx3 = 0.4; IU_tx3 = iu_tx; %Tx-3 current upper and lower bounds
IL_tx4 = 0.4; IU_tx4 = iu_tx; %Tx-4 current upper and lower bounds

% w_steps = 6;
W_MAX = (IU_tx1-IL_tx1)/2;
W_MIN = 0.001; %W_MAX/(NumT*NumCU);
P_IDC = 15; % W/A: The parameter to transform current to power

% EU energy threshold values
eu_th = -13.58731356201;%-13.954519154;%-13.969458;%EU 1 Energy Threshold values
EU1_th = eu_th; %EU 1 Energy Threshold values
EU2_th = eu_th; %EU 2 Energy Threshold values
EU3_th = eu_th; %EU 3 Energy Threshold values
EU4_th = eu_th; %EU 4 Energy Threshold values
EU5_th = eu_th; %EU 1 Energy Threshold values
EU6_th = eu_th; %EU 2 Energy Threshold values
EU7_th = eu_th; %EU 3 Energy Threshold values
EU8_th = eu_th; %EU 4 Energy Threshold values
EU9_th = eu_th; %EU 3 Energy Threshold values
EU10_th = eu_th; %EU 4 Energy Threshold values

EU_TH_mat = [EU1_th, EU2_th, EU3_th, EU4_th, EU5_th, EU6_th, EU7_th, EU8_th, EU9_th, EU10_th];

% CU rate threshold values 12.36099849128935 
cu_th = 2.650588239979590; %CU Rate Threshold values
CU1_th = cu_th; %CU 1 Rate Threshold values
CU2_th = cu_th; %CU 2 Rate Threshold values
CU3_th = cu_th; %CU 3 Rate Threshold values
CU4_th = cu_th; %CU 4 Rate Threshold values
CU5_th = cu_th; %CU 3 Rate Threshold values
CU6_th = cu_th; %CU 4 Rate Threshold values

opt_idc_zf_path_eu_i = [];
sum_opt_idc_zf_path_eu_i = [];
avg_sum_opt_idc_zf_path_eu_i = [];

for num_eu = 1:size(EU_TH_mat,2)
    
NumEU = num_eu;
EU_TH = EU_TH_mat(1,1:NumEU);

opt_idc_zf_path=[];
sum_opt_idc_zf_path=[];
avg_sum_opt_idc_zf_path=[];

for path_step = 1:1:U_step % Check for each steps

fprintf('Num_cu=%d, Num_eu=%d, Path number=%d \n',NumCU, num_eu, path_step); 
    
r_cu = R_CU_mat{1,path_step}; % get CU x and y coordinates
r_eu = R_EU_mat{1,path_step}; % get EU x and y coordinates
R_CU = [r_cu(:,2), r_cu(:,3), r_cu(:,1), r_cu(:,4)];
R_EU = [r_eu(:, 5), r_eu(:, 9), r_eu(:, 7:8), r_eu(:, 3:4), r_eu(:, 2), r_eu(:, 10), r_eu(:, 6), r_eu(:, 1)];
[H_CU_temp1, H_EU_temp1, U_blokage] = LOS_channelGain(T, R_CU, R_EU);% channel gain [Tx*Rx] - Cu and EU
H_CU_rx_tx = H_CU_temp1(1:NumCU, :);
H_CU_tx_rx = H_CU_rx_tx.';
H_EU = H_EU_temp1(1:NumEU, :);

% PRECODING MATRIX
inv_H = inv(H_CU_tx_rx'*H_CU_tx_rx)*H_CU_tx_rx';  %inv(H_CU);
% inv_H = inv_H.';
w_cu = [inv_H(1,:)/norm(inv_H(1,:)); inv_H(2,:)/norm(inv_H(2,:)); inv_H(3,:)/norm(inv_H(3,:))]/NumCU;%, inv_H(4,:)/norm(inv_H(4,:))]/NumCU; %; inv_H(4,:)/norm(inv_H(4,:))]/NumCU;
min_w_cu = min(min(w_cu));
max_w_cu = max(max(w_cu));
W_CU_temp = interp1([min_w_cu, max_w_cu],[W_MIN, W_MAX],w_cu(:));
W_CU = reshape(W_CU_temp, NumCU, NumT);
w_mat_temp = W_CU/NumCU;

% Rate calculation
idc_zf_opt=[];
rate_rx=[];

for w_ind = 1:size(w_mat_temp,1)
    w_temp = w_mat_temp(w_ind,:);
    idc_temp = IL_tx1 + w_temp;
    P_idc_temp = P_IDC*idc_temp;
    h_rx_CU = H_CU_rx_tx(w_ind,:).*P_idc_temp;
    
    rate_temp = 0.5*log2(1 + (exp(1)/2*pi)*(((h_rx_CU*w_temp').^2)/noise_var));
    rate_rx = [rate_rx, rate_temp];
end

rate_logic_CU1 = rate_rx(:,1) >= CU1_th;
rate_logic_CU2 = rate_rx(:,2) >= CU2_th;
rate_logic_CU3 = rate_rx(:,3) >= CU3_th;
rate_logic = rate_logic_CU1 .* rate_logic_CU2 .* rate_logic_CU3;
    
if sum(rate_logic) ~= 0
    sum_w_mat_temp = sum(w_mat_temp,1);
    w_temp = sum_w_mat_temp;
    idc_temp_lower = IL_tx1 + w_temp;
    idc_temp_upper = IU_tx1 - w_temp;
    
    i_both_sides=[];
    for i_ind = 1:size(idc_temp_lower,2)
        i_s = idc_temp_lower(:,i_ind);
        i_e = idc_temp_upper(:,i_ind);
        i_s_e = linspace(i_s,i_e, idc_res)';
        i_both_sides = [i_both_sides, i_s_e];
    end
        i_all_combinations=[];
        for ind_tx1=1:size(i_both_sides,1)
            val_tx1 = i_both_sides(ind_tx1,1);

            for ind_tx2=1:size(i_both_sides,1)
                val_tx2 = i_both_sides(ind_tx2,2);

                for ind_tx3=1:size(i_both_sides,1)
                    val_tx3 = i_both_sides(ind_tx3,3);

                    for ind_tx4=1:size(i_both_sides,1)
                        val_tx4 = i_both_sides(ind_tx4,4);

                        val_txs = [val_tx1, val_tx2, val_tx3, val_tx4];
                        i_all_combinations = [i_all_combinations; val_txs];
                    end
                end
            end
        end
    
    for ind_idc_selected = 1:size(i_all_combinations,1)
        i_selected_temp = i_all_combinations(ind_idc_selected,:);
        P_IDC_temp = P_IDC.*i_selected_temp; % convert current -> Power
        
        E=[]; % Initialize Energy matrix
        for rx_EU = 1:size(H_EU,1) % Select each Eu one by one
            h_rx_EU = H_EU(rx_EU,:); % Extract necessary channel gains for the selected EU from 4 Txs
            h_P_DC = h_rx_EU.*P_IDC_temp; % calculate tempoary value = h*I_DC or h*P_DC
            E_temp = f*v_t.*sum(h_P_DC.*log(1 + h_P_DC./I_0),2); % Calculate Energy as in Eqn.
            E = [E, E_temp]; % Store them in E matrix [rax: for each current combinations, col: for each EU]
        end
        E_dB = 10*log10(E); % Convert to dB values
        EdB_logic = E_dB > EU_TH;  % Filtering E values from constraint

        if sum(EdB_logic) == size(H_EU,1)
            idc_zf_opt = [idc_zf_opt; i_selected_temp];
        end
    end
end

sum_idc_zf_opt = sum(idc_zf_opt,2);
[min_sum_idc_zf_opt, ind_min_sum_idc_zf_opt] = min(sum_idc_zf_opt);
opt_idc_zf = idc_zf_opt(ind_min_sum_idc_zf_opt,:);

if size(opt_idc_zf,1)==0
    opt_idc_zf=ones(1,NumT)*NaN;
end

opt_idc_zf_path = [opt_idc_zf_path; opt_idc_zf];
opt_idc_zf_path = rmmissing(opt_idc_zf_path);
sum_opt_idc_zf_path = sum(opt_idc_zf_path,2);
avg_sum_opt_idc_zf_path = mean(sum_opt_idc_zf_path);
end

% opt_idc_zf_path_eu_i = [opt_idc_zf_path_eu_i, opt_idc_zf_path];
% sum_opt_idc_zf_path_eu_i = [sum_opt_idc_zf_path_eu_i, sum_opt_idc_zf_path];
avg_sum_opt_idc_zf_path_eu_i = [avg_sum_opt_idc_zf_path_eu_i, avg_sum_opt_idc_zf_path];

end

save('OPT_IDC_ZF_cu_3', 'avg_sum_opt_idc_zf_path_eu_i', '-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;
tic;
load('Position_mat_Noise_var_0_03_test_01.mat');
U_step = size(R_CU_mat,2); % Number of user steps for CUs and EUs
VLC_settings_temp; % For all user position initializations

NumCU = 2; %size(R_CU,2); % Number of CUs ##############
NumT = size(T,2); % Number of Txs
idc_res = 8;

% I_DC current values for 2 txs [Real IU=0.6A]
iu_tx = 10.5;
IL_tx1 = 0.4; IU_tx1 = iu_tx; %Tx-1 current upper and lower bounds
IL_tx2 = 0.4; IU_tx2 = iu_tx; %Tx-2 current upper and lower bounds
IL_tx3 = 0.4; IU_tx3 = iu_tx; %Tx-3 current upper and lower bounds
IL_tx4 = 0.4; IU_tx4 = iu_tx; %Tx-4 current upper and lower bounds

% w_steps = 6;
W_MAX = (IU_tx1-IL_tx1)/2;
W_MIN = 0.001; %W_MAX/(NumT*NumCU);
P_IDC = 15; % W/A: The parameter to transform current to power

% EU energy threshold values
eu_th = -13.58731356201;%-13.954519154;%-13.969458;%EU 1 Energy Threshold values
EU1_th = eu_th; %EU 1 Energy Threshold values
EU2_th = eu_th; %EU 2 Energy Threshold values
EU3_th = eu_th; %EU 3 Energy Threshold values
EU4_th = eu_th; %EU 4 Energy Threshold values
EU5_th = eu_th; %EU 1 Energy Threshold values
EU6_th = eu_th; %EU 2 Energy Threshold values
EU7_th = eu_th; %EU 3 Energy Threshold values
EU8_th = eu_th; %EU 4 Energy Threshold values
EU9_th = eu_th; %EU 3 Energy Threshold values
EU10_th = eu_th; %EU 4 Energy Threshold values

EU_TH_mat = [EU1_th, EU2_th, EU3_th, EU4_th, EU5_th, EU6_th, EU7_th, EU8_th, EU9_th, EU10_th];

% CU rate threshold values 12.36099849128935 
cu_th = 2.650588239979590; %CU Rate Threshold values
CU1_th = cu_th; %CU 1 Rate Threshold values
CU2_th = cu_th; %CU 2 Rate Threshold values
CU3_th = cu_th; %CU 3 Rate Threshold values
CU4_th = cu_th; %CU 4 Rate Threshold values
CU5_th = cu_th; %CU 3 Rate Threshold values
CU6_th = cu_th; %CU 4 Rate Threshold values

opt_idc_zf_path_eu_i = [];
sum_opt_idc_zf_path_eu_i = [];
avg_sum_opt_idc_zf_path_eu_i = [];

for num_eu = 1:size(EU_TH_mat,2)
    
NumEU = num_eu;
EU_TH = EU_TH_mat(1,1:NumEU);

opt_idc_zf_path=[];
sum_opt_idc_zf_path=[];
avg_sum_opt_idc_zf_path=[];

for path_step = 1:1:U_step % Check for each steps

fprintf('Num_cu=%d, Num_eu=%d, Path number=%d \n',NumCU, num_eu, path_step); 
    
r_cu = R_CU_mat{1,path_step}; % get CU x and y coordinates
r_eu = R_EU_mat{1,path_step}; % get EU x and y coordinates
R_CU = [r_cu(:,2), r_cu(:,3), r_cu(:,1), r_cu(:,4)];
R_EU = [r_eu(:, 5), r_eu(:, 9), r_eu(:, 7:8), r_eu(:, 3:4), r_eu(:, 2), r_eu(:, 10), r_eu(:, 6), r_eu(:, 1)];
[H_CU_temp1, H_EU_temp1, U_blokage] = LOS_channelGain(T, R_CU, R_EU);% channel gain [Tx*Rx] - Cu and EU
H_CU_rx_tx = H_CU_temp1(1:NumCU, :);
H_CU_tx_rx = H_CU_rx_tx.';
H_EU = H_EU_temp1(1:NumEU, :);

% PRECODING MATRIX
inv_H = inv(H_CU_tx_rx'*H_CU_tx_rx)*H_CU_tx_rx';  %inv(H_CU);
% inv_H = inv_H.';
w_cu = [inv_H(1,:)/norm(inv_H(1,:)); inv_H(2,:)/norm(inv_H(2,:))]/NumCU;
min_w_cu = min(min(w_cu));
max_w_cu = max(max(w_cu));
W_CU_temp = interp1([min_w_cu, max_w_cu],[W_MIN, W_MAX],w_cu(:));
W_CU = reshape(W_CU_temp, NumCU, NumT);
w_mat_temp = W_CU/NumCU;

% Rate calculation
idc_zf_opt=[];
rate_rx=[];

for w_ind = 1:size(w_mat_temp,1)
    w_temp = w_mat_temp(w_ind,:);
    idc_temp = IL_tx1 + w_temp;
    P_idc_temp = P_IDC*idc_temp;
    h_rx_CU = H_CU_rx_tx(w_ind,:).*P_idc_temp;
    
    rate_temp = 0.5*log2(1 + (exp(1)/2*pi)*(((h_rx_CU*w_temp').^2)/noise_var));
    rate_rx = [rate_rx, rate_temp];
end

rate_logic_CU1 = rate_rx(:,1) >= CU1_th;
rate_logic_CU2 = rate_rx(:,2) >= CU2_th;
rate_logic = rate_logic_CU1 .* rate_logic_CU2;
    
if sum(rate_logic) ~= 0
    sum_w_mat_temp = sum(w_mat_temp,1);
    w_temp = sum_w_mat_temp;
    idc_temp_lower = IL_tx1 + w_temp;
    idc_temp_upper = IU_tx1 - w_temp;
    
    i_both_sides=[];
    for i_ind = 1:size(idc_temp_lower,2)
        i_s = idc_temp_lower(:,i_ind);
        i_e = idc_temp_upper(:,i_ind);
        i_s_e = linspace(i_s,i_e, idc_res)';
        i_both_sides = [i_both_sides, i_s_e];
    end
        i_all_combinations=[];
        for ind_tx1=1:size(i_both_sides,1)
            val_tx1 = i_both_sides(ind_tx1,1);

            for ind_tx2=1:size(i_both_sides,1)
                val_tx2 = i_both_sides(ind_tx2,2);

                for ind_tx3=1:size(i_both_sides,1)
                    val_tx3 = i_both_sides(ind_tx3,3);

                    for ind_tx4=1:size(i_both_sides,1)
                        val_tx4 = i_both_sides(ind_tx4,4);

                        val_txs = [val_tx1, val_tx2, val_tx3, val_tx4];
                        i_all_combinations = [i_all_combinations; val_txs];
                    end
                end
            end
        end
    
    for ind_idc_selected = 1:size(i_all_combinations,1)
        i_selected_temp = i_all_combinations(ind_idc_selected,:);
        P_IDC_temp = P_IDC.*i_selected_temp; % convert current -> Power
        
        E=[]; % Initialize Energy matrix
        for rx_EU = 1:size(H_EU,1) % Select each Eu one by one
            h_rx_EU = H_EU(rx_EU,:); % Extract necessary channel gains for the selected EU from 4 Txs
            h_P_DC = h_rx_EU.*P_IDC_temp; % calculate tempoary value = h*I_DC or h*P_DC
            E_temp = f*v_t.*sum(h_P_DC.*log(1 + h_P_DC./I_0),2); % Calculate Energy as in Eqn.
            E = [E, E_temp]; % Store them in E matrix [rax: for each current combinations, col: for each EU]
        end
        E_dB = 10*log10(E); % Convert to dB values
        EdB_logic = E_dB > EU_TH;  % Filtering E values from constraint

        if sum(EdB_logic) == size(H_EU,1)
            idc_zf_opt = [idc_zf_opt; i_selected_temp];
        end
    end
end

sum_idc_zf_opt = sum(idc_zf_opt,2);
[min_sum_idc_zf_opt, ind_min_sum_idc_zf_opt] = min(sum_idc_zf_opt);
opt_idc_zf = idc_zf_opt(ind_min_sum_idc_zf_opt,:);

if size(opt_idc_zf,1)==0
    opt_idc_zf=ones(1,NumT)*NaN;
end

opt_idc_zf_path = [opt_idc_zf_path; opt_idc_zf];
opt_idc_zf_path = rmmissing(opt_idc_zf_path);
sum_opt_idc_zf_path = sum(opt_idc_zf_path,2);
avg_sum_opt_idc_zf_path = mean(sum_opt_idc_zf_path);
end

% opt_idc_zf_path_eu_i = [opt_idc_zf_path_eu_i, opt_idc_zf_path];
% sum_opt_idc_zf_path_eu_i = [sum_opt_idc_zf_path_eu_i, sum_opt_idc_zf_path];
avg_sum_opt_idc_zf_path_eu_i = [avg_sum_opt_idc_zf_path_eu_i, avg_sum_opt_idc_zf_path];

end

save('OPT_IDC_ZF_cu_2', 'avg_sum_opt_idc_zf_path_eu_i', '-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;
tic;
load('Position_mat_Noise_var_0_03_test_01.mat');
U_step = size(R_CU_mat,2); % Number of user steps for CUs and EUs
VLC_settings_temp; % For all user position initializations

NumCU = 1; %size(R_CU,2); % Number of CUs ##############
NumT = size(T,2); % Number of Txs
idc_res = 8;

% I_DC current values for 2 txs [Real IU=0.6A]
iu_tx = 10.5;
IL_tx1 = 0.4; IU_tx1 = iu_tx; %Tx-1 current upper and lower bounds
IL_tx2 = 0.4; IU_tx2 = iu_tx; %Tx-2 current upper and lower bounds
IL_tx3 = 0.4; IU_tx3 = iu_tx; %Tx-3 current upper and lower bounds
IL_tx4 = 0.4; IU_tx4 = iu_tx; %Tx-4 current upper and lower bounds

% w_steps = 6;
W_MAX = (IU_tx1-IL_tx1)/2;
W_MIN = 0.001; %W_MAX/(NumT*NumCU);
P_IDC = 15; % W/A: The parameter to transform current to power

% EU energy threshold values
eu_th = -13.58731356201;%-13.954519154;%-13.969458;%EU 1 Energy Threshold values
EU1_th = eu_th; %EU 1 Energy Threshold values
EU2_th = eu_th; %EU 2 Energy Threshold values
EU3_th = eu_th; %EU 3 Energy Threshold values
EU4_th = eu_th; %EU 4 Energy Threshold values
EU5_th = eu_th; %EU 1 Energy Threshold values
EU6_th = eu_th; %EU 2 Energy Threshold values
EU7_th = eu_th; %EU 3 Energy Threshold values
EU8_th = eu_th; %EU 4 Energy Threshold values
EU9_th = eu_th; %EU 3 Energy Threshold values
EU10_th = eu_th; %EU 4 Energy Threshold values

EU_TH_mat = [EU1_th, EU2_th, EU3_th, EU4_th, EU5_th, EU6_th, EU7_th, EU8_th, EU9_th, EU10_th];

% CU rate threshold values 12.36099849128935 
cu_th = 2.650588239979590; %CU Rate Threshold values
CU1_th = cu_th; %CU 1 Rate Threshold values
CU2_th = cu_th; %CU 2 Rate Threshold values
CU3_th = cu_th; %CU 3 Rate Threshold values
CU4_th = cu_th; %CU 4 Rate Threshold values
CU5_th = cu_th; %CU 3 Rate Threshold values
CU6_th = cu_th; %CU 4 Rate Threshold values

opt_idc_zf_path_eu_i = [];
sum_opt_idc_zf_path_eu_i = [];
avg_sum_opt_idc_zf_path_eu_i = [];

for num_eu = 1:size(EU_TH_mat,2)
    
NumEU = num_eu;
EU_TH = EU_TH_mat(1,1:NumEU);

opt_idc_zf_path=[];
sum_opt_idc_zf_path=[];
avg_sum_opt_idc_zf_path=[];

for path_step = 1:1:U_step % Check for each steps

fprintf('Num_cu=%d, Num_eu=%d, Path number=%d \n',NumCU, num_eu, path_step); 
    
r_cu = R_CU_mat{1,path_step}; % get CU x and y coordinates
r_eu = R_EU_mat{1,path_step}; % get EU x and y coordinates
R_CU = [r_cu(:,2), r_cu(:,3), r_cu(:,1), r_cu(:,4)];
R_EU = [r_eu(:, 5), r_eu(:, 9), r_eu(:, 7:8), r_eu(:, 3:4), r_eu(:, 2), r_eu(:, 10), r_eu(:, 6), r_eu(:, 1)];
[H_CU_temp1, H_EU_temp1, U_blokage] = LOS_channelGain(T, R_CU, R_EU);% channel gain [Tx*Rx] - Cu and EU
H_CU_rx_tx = H_CU_temp1(1:NumCU, :);
H_CU_tx_rx = H_CU_rx_tx.';
H_EU = H_EU_temp1(1:NumEU, :);

% PRECODING MATRIX
inv_H = inv(H_CU_tx_rx'*H_CU_tx_rx)*H_CU_tx_rx';  %inv(H_CU);
% inv_H = inv_H.';
w_cu = [inv_H(1,:)/norm(inv_H(1,:))]/NumCU;
min_w_cu = min(min(w_cu));
max_w_cu = max(max(w_cu));
W_CU_temp = interp1([min_w_cu, max_w_cu],[W_MIN, W_MAX],w_cu(:));
W_CU = reshape(W_CU_temp, NumCU, NumT);
w_mat_temp = W_CU/NumCU;

% Rate calculation
idc_zf_opt=[];
rate_rx=[];

for w_ind = 1:size(w_mat_temp,1)
    w_temp = w_mat_temp(w_ind,:);
    idc_temp = IL_tx1 + w_temp;
    P_idc_temp = P_IDC*idc_temp;
    h_rx_CU = H_CU_rx_tx(w_ind,:).*P_idc_temp;
    
    rate_temp = 0.5*log2(1 + (exp(1)/2*pi)*(((h_rx_CU*w_temp').^2)/noise_var));
    rate_rx = [rate_rx, rate_temp];
end

rate_logic_CU1 = rate_rx(:,1) >= CU1_th;
rate_logic = rate_logic_CU1;
    
if sum(rate_logic) ~= 0
    sum_w_mat_temp = sum(w_mat_temp,1);
    w_temp = sum_w_mat_temp;
    idc_temp_lower = IL_tx1 + w_temp;
    idc_temp_upper = IU_tx1 - w_temp;
    
    i_both_sides=[];
    for i_ind = 1:size(idc_temp_lower,2)
        i_s = idc_temp_lower(:,i_ind);
        i_e = idc_temp_upper(:,i_ind);
        i_s_e = linspace(i_s,i_e, idc_res)';
        i_both_sides = [i_both_sides, i_s_e];
    end
        i_all_combinations=[];
        for ind_tx1=1:size(i_both_sides,1)
            val_tx1 = i_both_sides(ind_tx1,1);

            for ind_tx2=1:size(i_both_sides,1)
                val_tx2 = i_both_sides(ind_tx2,2);

                for ind_tx3=1:size(i_both_sides,1)
                    val_tx3 = i_both_sides(ind_tx3,3);

                    for ind_tx4=1:size(i_both_sides,1)
                        val_tx4 = i_both_sides(ind_tx4,4);

                        val_txs = [val_tx1, val_tx2, val_tx3, val_tx4];
                        i_all_combinations = [i_all_combinations; val_txs];
                    end
                end
            end
        end
    
    for ind_idc_selected = 1:size(i_all_combinations,1)
        i_selected_temp = i_all_combinations(ind_idc_selected,:);
        P_IDC_temp = P_IDC.*i_selected_temp; % convert current -> Power
        
        E=[]; % Initialize Energy matrix
        for rx_EU = 1:size(H_EU,1) % Select each Eu one by one
            h_rx_EU = H_EU(rx_EU,:); % Extract necessary channel gains for the selected EU from 4 Txs
            h_P_DC = h_rx_EU.*P_IDC_temp; % calculate tempoary value = h*I_DC or h*P_DC
            E_temp = f*v_t.*sum(h_P_DC.*log(1 + h_P_DC./I_0),2); % Calculate Energy as in Eqn.
            E = [E, E_temp]; % Store them in E matrix [rax: for each current combinations, col: for each EU]
        end
        E_dB = 10*log10(E); % Convert to dB values
        EdB_logic = E_dB > EU_TH;  % Filtering E values from constraint

        if sum(EdB_logic) == size(H_EU,1)
            idc_zf_opt = [idc_zf_opt; i_selected_temp];
        end
    end
end

sum_idc_zf_opt = sum(idc_zf_opt,2);
[min_sum_idc_zf_opt, ind_min_sum_idc_zf_opt] = min(sum_idc_zf_opt);
opt_idc_zf = idc_zf_opt(ind_min_sum_idc_zf_opt,:);

if size(opt_idc_zf,1)==0
    opt_idc_zf=ones(1,NumT)*NaN;
end

opt_idc_zf_path = [opt_idc_zf_path; opt_idc_zf];
opt_idc_zf_path = rmmissing(opt_idc_zf_path);
sum_opt_idc_zf_path = sum(opt_idc_zf_path,2);
avg_sum_opt_idc_zf_path = mean(sum_opt_idc_zf_path);
end

% opt_idc_zf_path_eu_i = [opt_idc_zf_path_eu_i, opt_idc_zf_path];
% sum_opt_idc_zf_path_eu_i = [sum_opt_idc_zf_path_eu_i, sum_opt_idc_zf_path];
avg_sum_opt_idc_zf_path_eu_i = [avg_sum_opt_idc_zf_path_eu_i, avg_sum_opt_idc_zf_path];

end

save('OPT_IDC_ZF_cu_1', 'avg_sum_opt_idc_zf_path_eu_i', '-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

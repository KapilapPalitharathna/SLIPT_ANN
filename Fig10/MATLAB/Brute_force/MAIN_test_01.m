close all; clear; clc;
tic;
load('Position_mat_Noise_var_0_03_test_01.mat');
U_step = size(R_CU_mat,2); % Number of user steps for CUs and EUs
VLC_settings_temp; % For all user position initializations

NumCU = 4; %size(R_CU,2); % Number of CUs ##############
NumT = size(T,2); % Number of Txs

IDC_optm_path_eu_i = [];
sum_IDC_optm_path_eu_i = [];
avg_sum_IDC_optm_path_eu_i = [];

% I_DC current values for 2 txs [Real IU=0.6A]
iu_tx = 6;
IL_tx1 = 0.4; IU_tx1 = iu_tx; %Tx-1 current upper and lower bounds
IL_tx2 = 0.4; IU_tx2 = iu_tx; %Tx-2 current upper and lower bounds
IL_tx3 = 0.4; IU_tx3 = iu_tx; %Tx-3 current upper and lower bounds
IL_tx4 = 0.4; IU_tx4 = iu_tx; %Tx-4 current upper and lower bounds

IDC_step_tx1 = 30;%50
IDC_step_tx2 = 30;
IDC_step_tx3 = 30;
IDC_step_tx4 = 30;

Num_raw_W_temp = 10;%15; % Resolution of W values (Number of steps of beamforming value (W) from min to max)
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
EU9_th = eu_th; %EU 4 Energy Threshold values
EU10_th = eu_th; %EU 4 Energy Threshold values

EU_TH_mat = [EU1_th, EU2_th, EU3_th, EU4_th, EU5_th, EU6_th, EU7_th, EU8_th, EU9_th, EU10_th];

% CU rate threshold values 12.36099849128935 
cu_th = 2.650576505433320; %CU Rate Threshold values
CU1_th = cu_th; %CU 1 Rate Threshold values
CU2_th = cu_th; %CU 2 Rate Threshold values
CU3_th = cu_th; %CU 3 Rate Threshold values
CU4_th = cu_th; %CU 4 Rate Threshold values
CU5_th = cu_th; %CU 3 Rate Threshold values
CU6_th = cu_th; %CU 4 Rate Threshold values

for num_eu = 1:size(EU_TH_mat,2)
    
NumEU = num_eu;
EU_TH = EU_TH_mat(1,1:NumEU);

IDC_optm_path = []; % Optimum DC current matrix
W_opt_path = []; % Optimum Pre-coding matrix
EdB_logic_path = [];
E_dB_i_path=[];
r_eu_i_path=[];
I_DC_combination_mat=[];
No_of_IDC_selected_path=[];
Blockage_mat_test_01=[];

for path_step = 1:1:U_step % 556:U_step %1:1:U_step % Check for each steps

IDC_opt_temp=[]; % I_DC optimum temporay values
IDC_optm_sum_iter = []; % Summation of I_DC optimum temporay values 
W_opt_temp=[]; % W optimum temporay values 

r_cu = R_CU_mat{1,path_step}; % get CU x and y coordinates
r_eu = R_EU_mat{1,path_step}; % get EU x and y coordinates
R_CU = r_cu;
R_EU = [r_eu(:, 5), r_eu(:, 9), r_eu(:, 7:8), r_eu(:, 3:4), r_eu(:, 2), r_eu(:, 10), r_eu(:, 6), r_eu(:, 1)];
[H_CU_temp1, H_EU_temp1, U_blokage] = LOS_channelGain(T, R_CU, R_EU);% channel gain [Tx*Rx] - Cu and EU
H_CU = H_CU_temp1(1:NumCU, :);
H_EU = H_EU_temp1(1:NumEU, :);

U_blokage1 = 1-U_blokage;
Blockage_mat_test_01 = [Blockage_mat_test_01; U_blokage1];

% Using above 2 parameters, define new(filtered) DC current vector
I_DC_tx1_temp = linspace(IL_tx1,IU_tx1,IDC_step_tx1)'; I_DC_tx1 = I_DC_tx1_temp(2:end-1,:);
I_DC_tx2_temp = linspace(IL_tx2,IU_tx2,IDC_step_tx2)'; I_DC_tx2 = I_DC_tx2_temp(2:end-1,:);
I_DC_tx3_temp = linspace(IL_tx3,IU_tx3,IDC_step_tx3)'; I_DC_tx3 = I_DC_tx3_temp(2:end-1,:);
I_DC_tx4_temp = linspace(IL_tx4,IU_tx4,IDC_step_tx4)'; I_DC_tx4 = I_DC_tx4_temp(2:end-1,:);

%%%%%% Energy Users - EU %%%%%%
[p, q, r, s] = ndgrid(I_DC_tx1, I_DC_tx2, I_DC_tx3, I_DC_tx4); % get all I_DC combinations
I_DC_temp = [s(:),r(:),q(:),p(:)]; % re-order all combinations
P_IDC_temp = P_IDC.*I_DC_temp; % convert current -> Power

E=[]; % Initialize Energy matrix
for rx_EU = 1:size(H_EU,1) % Select each Eu one by one
    h_rx_EU = H_EU(rx_EU,:); % Extract necessary channel gains for the selected EU from 4 Txs
    h_P_DC = h_rx_EU.*P_IDC_temp; % calculate tempoary value = h*I_DC or h*P_DC
    E_temp = f*v_t.*sum(h_P_DC.*log(1 + h_P_DC./I_0),2); % Calculate Energy as in Eqn.
    E = [E, E_temp]; % Store them in E matrix [rax: for each current combinations, col: for each EU]
end
E_dB = 10*log10(E); % Convert to dB values

E_dB_i = max(E_dB); % 1st EU
E_dB_i_path = [E_dB_i_path; E_dB_i];
r_eu_i = R_EU(:)';
r_eu_i_path = [r_eu_i_path; r_eu_i];

EdB_logic = E_dB > EU_TH;  % Filtering E values from constraint #############
EdB_logic_path = [EdB_logic_path; sum(EdB_logic)];
sum_EdB_logic = sum(EdB_logic,2); % Summation energy of all 4 Txs
[x,y] = find(sum_EdB_logic==NumEU); % Check satisfied indexes of the Energy constraint for all 4 Txs
I_dc_selected = I_DC_temp(x,:); % Filtered all current combinations which satsfied the Energy constraint

% Check there are no any current combination to match the Energy constraint and if not code is terminated
I_dc_selected_flag = sum(sum(I_dc_selected)); 
if I_dc_selected_flag == 0
    fprintf('Energy constrint is not satisfied for any current combination \n');
    break;
end
sum_I_DC_opt_temp = sum(I_dc_selected,2); % get sum for each temp opt IDC 
[I_DC_min,ind_I_DC_min] = sort(sum_I_DC_opt_temp);

ind_I_DC_min = ind_I_DC_min(1:round(size(ind_I_DC_min,1)*0.5),:);
No_of_IDC_selected_path = [No_of_IDC_selected_path;size(ind_I_DC_min,1)];

for ind_i_dc_min_temp = 1:size(ind_I_DC_min,1)
ind_I_DC_opt_temp = ind_I_DC_min(ind_i_dc_min_temp);
IDC_optm_temp = I_dc_selected(ind_I_DC_opt_temp,:); %find IDC optimum temporary

W_TH_TX = min([IDC_optm_temp-[IL_tx1, IL_tx2, IL_tx3, IL_tx4]],[[IU_tx1, IU_tx1, IU_tx1, IU_tx1]-IDC_optm_temp]); % Get max value of W elements from DC current values

W_temp = ones(Num_raw_W_temp-1, NumT)*0.0001;
for w_ind_temp = 1:size(W_TH_TX,2)
    W_TH_TX_temp = W_TH_TX(w_ind_temp);
    W_tx_i_temp1 = linspace(0,W_TH_TX_temp,Num_raw_W_temp)';
    W_tx_i_temp = W_tx_i_temp1(2:end,:);
    W_temp(:,w_ind_temp) = W_tx_i_temp;
end

len_W = Num_raw_W_temp-1; % No.of raws in W_temp matrix
W_mat = [];%Initialize W matrix
for w1 = 1:len_W
    W1 = W_temp(w1,:);
    W1_mat = repmat(W1,len_W,1);
    for w2 = 1:len_W
        W2 = W_temp(w2,:);
        W2_mat = repmat(W2,len_W,1);
        for w3 = 1:len_W
            W3 = W_temp(w3,:);
            W3_mat = repmat(W3,len_W,1);
            W4_mat = W_temp;
            W_mat_temp = [W1_mat, W2_mat, W3_mat, W4_mat];
            W_mat = [W_mat; W_mat_temp];
            fprintf('Path number=%d, selected opt idc temp=%d/%d\n',...
                path_step,ind_i_dc_min_temp,size(ind_I_DC_min,1));
        end
    end
end

W_mat = W_mat/NumCU;
rate_rx = [];
P_IDC_temp2 = 1;%P_IDC.*IDC_optm_temp;
%%%% Communication Users - CU
for Hcu_ind = 1:size(H_CU,1) % Select the CU
    H_CU_temp = H_CU(Hcu_ind,:);
    h_rx_CU = P_IDC_temp2.*H_CU_temp;
    w_desi_mat = W_mat(:,1:4);
    w_intf_mat = W_mat(:,5:8) + W_mat(:,9:12) + W_mat(:,13:16);
    
    % Rate calculation
    rate_temp = 0.5*log2(1 + (exp(1)/2*pi)*((w_desi_mat*h_rx_CU').^2)./(((w_intf_mat*h_rx_CU').^2) + noise_var));
	rate_rx = [rate_rx, rate_temp];
% 	fprintf('CU_path_step=%d, Hcu_ind=%d \n',path_step, Hcu_ind);
end

rate_logic_CU1 = rate_rx(:,1) >= CU1_th;
rate_logic_CU2 = rate_rx(:,2) >= CU2_th; %###########
rate_logic_CU3 = rate_rx(:,3) >= CU3_th;
rate_logic_CU4 = rate_rx(:,4) >= CU4_th;
% rate_logic_CU5 = rate_rx(:,5) >= CU5_th;
% rate_logic_CU6 = rate_rx(:,6) >= CU6_th;
rate_logic = rate_logic_CU1.*rate_logic_CU2.*rate_logic_CU3.*rate_logic_CU4;%##########

if sum(rate_logic) ~= 0
    IDC_optm = IDC_optm_temp;
    % TO find W opt
    ind_rate_logic= find(rate_logic == 1); %Find rate index from logic matrix
    mid_rate_logic = round(length(ind_rate_logic)/2); % find middle
	ind_mid_rate_logic = ind_rate_logic(mid_rate_logic); % get middle index value from indexes set 
	W_opt = W_mat(ind_mid_rate_logic,:); % get W_opt raw
	W_OPT = reshape(W_opt,NumT,NumCU)'; % reshape W_opt as (no. of user * no. of tx)
% 	fprintf('The satisfied IDC combination is: %d \n',ind_i_dc_min_temp);
    I_DC_combination_mat = [I_DC_combination_mat; ind_i_dc_min_temp];
	break;
end

if ind_i_dc_min_temp == size(ind_I_DC_min,1)
    fprintf('There are no any satisfied IDC combination for step no.: %d \n',path_step);
    IDC_optm = [];
end
end

disp(W_OPT);
W_opt_path = [W_opt_path; W_opt];

IDC_optm_path = [IDC_optm_path; IDC_optm];
sum_IDC_optm_path = sum(IDC_optm_path,2);
avg_sum_IDC_optm_path = mean(sum_IDC_optm_path);
end

IDC_optm_path_eu_i = [IDC_optm_path_eu_i, IDC_optm_path];
sum_IDC_optm_path_eu_i = [sum_IDC_optm_path_eu_i, sum_IDC_optm_path];
avg_sum_IDC_optm_path_eu_i = [avg_sum_IDC_optm_path_eu_i, avg_sum_IDC_optm_path];

end

save('OPT_IDC_cu_4', 'IDC_optm_path_eu_i', 'sum_IDC_optm_path_eu_i', 'avg_sum_IDC_optm_path_eu_i', '-v7.3');
toc;

figure(1);
plot(1:size(EU_TH_mat,2), avg_sum_IDC_optm_path_eu_i,'-ok','LineWidth',0.8);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;
tic;
load('Position_mat_Noise_var_0_03_test_01.mat');
U_step = size(R_CU_mat,2); % Number of user steps for CUs and EUs
VLC_settings_temp; % For all user position initializations

NumCU = 3; %size(R_CU,2); % Number of CUs ##############
NumT = size(T,2); % Number of Txs

IDC_optm_path_eu_i = [];
sum_IDC_optm_path_eu_i = [];
avg_sum_IDC_optm_path_eu_i = [];

% I_DC current values for 2 txs [Real IU=0.6A]
iu_tx = 6;
IL_tx1 = 0.4; IU_tx1 = iu_tx; %Tx-1 current upper and lower bounds
IL_tx2 = 0.4; IU_tx2 = iu_tx; %Tx-2 current upper and lower bounds
IL_tx3 = 0.4; IU_tx3 = iu_tx; %Tx-3 current upper and lower bounds
IL_tx4 = 0.4; IU_tx4 = iu_tx; %Tx-4 current upper and lower bounds

IDC_step_tx1 = 30;%50
IDC_step_tx2 = 30;
IDC_step_tx3 = 30;
IDC_step_tx4 = 30;

Num_raw_W_temp = 10;%15; % Resolution of W values (Number of steps of beamforming value (W) from min to max)
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
cu_th = 2.650576505433320; %CU Rate Threshold values
CU1_th = cu_th; %CU 1 Rate Threshold values
CU2_th = cu_th; %CU 2 Rate Threshold values
CU3_th = cu_th; %CU 3 Rate Threshold values
CU4_th = cu_th; %CU 4 Rate Threshold values
CU5_th = cu_th; %CU 3 Rate Threshold values
CU6_th = cu_th; %CU 4 Rate Threshold values

for num_eu = 1:size(EU_TH_mat,2)
    
NumEU = num_eu;
EU_TH = EU_TH_mat(1,1:NumEU);

IDC_optm_path = []; % Optimum DC current matrix
W_opt_path = []; % Optimum Pre-coding matrix
EdB_logic_path = [];
E_dB_i_path=[];
r_eu_i_path=[];
I_DC_combination_mat=[];
No_of_IDC_selected_path=[];
Blockage_mat_test_01=[];

for path_step = 1:1:U_step % 556:U_step %1:1:U_step % Check for each steps

IDC_opt_temp=[]; % I_DC optimum temporay values
IDC_optm_sum_iter = []; % Summation of I_DC optimum temporay values 
W_opt_temp=[]; % W optimum temporay values 

r_cu = R_CU_mat{1,path_step}; % get CU x and y coordinates
r_eu = R_EU_mat{1,path_step}; % get EU x and y coordinates
R_CU = r_cu;
R_EU = [r_eu(:, 5), r_eu(:, 9), r_eu(:, 7:8), r_eu(:, 3:4), r_eu(:, 2), r_eu(:, 10), r_eu(:, 6), r_eu(:, 1)];
[H_CU_temp1, H_EU_temp1, U_blokage] = LOS_channelGain(T, R_CU, R_EU);% channel gain [Tx*Rx] - Cu and EU
H_CU = H_CU_temp1(1:NumCU, :);
H_EU = H_EU_temp1(1:NumEU, :);

U_blokage1 = 1-U_blokage;
Blockage_mat_test_01 = [Blockage_mat_test_01; U_blokage1];

% Using above 2 parameters, define new(filtered) DC current vector
I_DC_tx1_temp = linspace(IL_tx1,IU_tx1,IDC_step_tx1)'; I_DC_tx1 = I_DC_tx1_temp(2:end-1,:);
I_DC_tx2_temp = linspace(IL_tx2,IU_tx2,IDC_step_tx2)'; I_DC_tx2 = I_DC_tx2_temp(2:end-1,:);
I_DC_tx3_temp = linspace(IL_tx3,IU_tx3,IDC_step_tx3)'; I_DC_tx3 = I_DC_tx3_temp(2:end-1,:);
I_DC_tx4_temp = linspace(IL_tx4,IU_tx4,IDC_step_tx4)'; I_DC_tx4 = I_DC_tx4_temp(2:end-1,:);

%%%%%% Energy Users - EU %%%%%%
[p, q, r, s] = ndgrid(I_DC_tx1, I_DC_tx2, I_DC_tx3, I_DC_tx4); % get all I_DC combinations
I_DC_temp = [s(:),r(:),q(:),p(:)]; % re-order all combinations
P_IDC_temp = P_IDC.*I_DC_temp; % convert current -> Power

E=[]; % Initialize Energy matrix
for rx_EU = 1:size(H_EU,1) % Select each Eu one by one
    h_rx_EU = H_EU(rx_EU,:); % Extract necessary channel gains for the selected EU from 4 Txs
    h_P_DC = h_rx_EU.*P_IDC_temp; % calculate tempoary value = h*I_DC or h*P_DC
    E_temp = f*v_t.*sum(h_P_DC.*log(1 + h_P_DC./I_0),2); % Calculate Energy as in Eqn.
    E = [E, E_temp]; % Store them in E matrix [rax: for each current combinations, col: for each EU]
end
E_dB = 10*log10(E); % Convert to dB values

E_dB_i = max(E_dB); % 1st EU
E_dB_i_path = [E_dB_i_path; E_dB_i];
r_eu_i = R_EU(:)';
r_eu_i_path = [r_eu_i_path; r_eu_i];

EdB_logic = E_dB > EU_TH;  % Filtering E values from constraint #############
EdB_logic_path = [EdB_logic_path; sum(EdB_logic)];
sum_EdB_logic = sum(EdB_logic,2); % Summation energy of all 4 Txs
[x,y] = find(sum_EdB_logic==NumEU); % Check satisfied indexes of the Energy constraint for all 4 Txs
I_dc_selected = I_DC_temp(x,:); % Filtered all current combinations which satsfied the Energy constraint

% Check there are no any current combination to match the Energy constraint and if not code is terminated
I_dc_selected_flag = sum(sum(I_dc_selected)); 
if I_dc_selected_flag == 0
    fprintf('Energy constrint is not satisfied for any current combination \n');
    break;
end
sum_I_DC_opt_temp = sum(I_dc_selected,2); % get sum for each temp opt IDC 
[I_DC_min,ind_I_DC_min] = sort(sum_I_DC_opt_temp);

ind_I_DC_min = ind_I_DC_min(1:round(size(ind_I_DC_min,1)*0.5),:);
No_of_IDC_selected_path = [No_of_IDC_selected_path;size(ind_I_DC_min,1)];

for ind_i_dc_min_temp = 1:size(ind_I_DC_min,1)
ind_I_DC_opt_temp = ind_I_DC_min(ind_i_dc_min_temp);
IDC_optm_temp = I_dc_selected(ind_I_DC_opt_temp,:); %find IDC optimum temporary

W_TH_TX = min([IDC_optm_temp-[IL_tx1, IL_tx2, IL_tx3, IL_tx4]],[[IU_tx1, IU_tx1, IU_tx1, IU_tx1]-IDC_optm_temp]); % Get max value of W elements from DC current values

W_temp = ones(Num_raw_W_temp-1, NumT)*0.0001;
for w_ind_temp = 1:size(W_TH_TX,2)
    W_TH_TX_temp = W_TH_TX(w_ind_temp);
    W_tx_i_temp1 = linspace(0,W_TH_TX_temp,Num_raw_W_temp)';
    W_tx_i_temp = W_tx_i_temp1(2:end,:);
    W_temp(:,w_ind_temp) = W_tx_i_temp;
end

len_W = Num_raw_W_temp-1; % No.of raws in W_temp matrix
W_mat = [];%Initialize W matrix
for w1 = 1:len_W
    W1 = W_temp(w1,:);
    W1_mat = repmat(W1,len_W,1);
    for w2 = 1:len_W
        W2 = W_temp(w2,:);
        W2_mat = repmat(W2,len_W,1);
        W3_mat = W_temp;
        W_mat_temp = [W1_mat, W2_mat, W3_mat];
        W_mat = [W_mat; W_mat_temp];
        fprintf('Path number=%d, selected opt idc temp=%d/%d\n',path_step,ind_i_dc_min_temp,size(ind_I_DC_min,1));
    end
end
W_mat = W_mat/NumCU;
rate_rx = [];
P_IDC_temp2 = P_IDC.*IDC_optm_temp;
%%%% Communication Users - CU
for Hcu_ind = 1:size(H_CU,1) % Select the CU
    H_CU_temp = H_CU(Hcu_ind,:);
    h_rx_CU = P_IDC_temp2.*H_CU_temp;
    w_desi_mat = W_mat(:,1:4);
    w_intf_mat = W_mat(:,5:8) + W_mat(:,9:12);
    
    % Rate calculation
    rate_temp = 0.5*log2(1 + (exp(1)/2*pi)*((w_desi_mat*h_rx_CU').^2)./(((w_intf_mat*h_rx_CU').^2) + noise_var));
	rate_rx = [rate_rx, rate_temp];
end

rate_logic_CU1 = rate_rx(:,1) >= CU1_th;
rate_logic_CU2 = rate_rx(:,2) >= CU2_th; 
rate_logic_CU3 = rate_rx(:,3) >= CU3_th;

rate_logic = rate_logic_CU1.*rate_logic_CU2.*rate_logic_CU3;

if sum(rate_logic) ~= 0
    IDC_optm = IDC_optm_temp;
    % TO find W opt
    ind_rate_logic= find(rate_logic == 1); %Find rate index from logic matrix
    mid_rate_logic = round(length(ind_rate_logic)/2); % find middle
	ind_mid_rate_logic = ind_rate_logic(mid_rate_logic); % get middle index value from indexes set 
	W_opt = W_mat(ind_mid_rate_logic,:); % get W_opt raw
	W_OPT = reshape(W_opt,NumT,NumCU)'; % reshape W_opt as (no. of user * no. of tx)
    I_DC_combination_mat = [I_DC_combination_mat; ind_i_dc_min_temp];
	break;
end

if ind_i_dc_min_temp == size(ind_I_DC_min,1)
    fprintf('There are no any satisfied IDC combination for step no.: %d \n',path_step);
    IDC_optm = [];
end
end

disp(W_OPT);
W_opt_path = [W_opt_path; W_opt];

IDC_optm_path = [IDC_optm_path; IDC_optm];
sum_IDC_optm_path = sum(IDC_optm_path,2);
avg_sum_IDC_optm_path = mean(sum_IDC_optm_path);
end

IDC_optm_path_eu_i = [IDC_optm_path_eu_i, IDC_optm_path];
sum_IDC_optm_path_eu_i = [sum_IDC_optm_path_eu_i, sum_IDC_optm_path];
avg_sum_IDC_optm_path_eu_i = [avg_sum_IDC_optm_path_eu_i, avg_sum_IDC_optm_path];

end

save('OPT_IDC_cu_3', 'IDC_optm_path_eu_i', 'sum_IDC_optm_path_eu_i', 'avg_sum_IDC_optm_path_eu_i', '-v7.3');
toc;

figure(1);
plot(1:size(EU_TH_mat,2), avg_sum_IDC_optm_path_eu_i,'-ok','LineWidth',0.8);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;
tic;
load('Position_mat_Noise_var_0_03_test_01.mat');
U_step = size(R_CU_mat,2); % Number of user steps for CUs and EUs
VLC_settings_temp; % For all user position initializations

NumCU = 2; %size(R_CU,2); % Number of CUs ##############
NumT = size(T,2); % Number of Txs

IDC_optm_path_eu_i = [];
sum_IDC_optm_path_eu_i = [];
avg_sum_IDC_optm_path_eu_i = [];

% I_DC current values for 2 txs [Real IU=0.6A]
iu_tx = 6;
IL_tx1 = 0.4; IU_tx1 = iu_tx; %Tx-1 current upper and lower bounds
IL_tx2 = 0.4; IU_tx2 = iu_tx; %Tx-2 current upper and lower bounds
IL_tx3 = 0.4; IU_tx3 = iu_tx; %Tx-3 current upper and lower bounds
IL_tx4 = 0.4; IU_tx4 = iu_tx; %Tx-4 current upper and lower bounds

IDC_step_tx1 = 30;%50
IDC_step_tx2 = 30;
IDC_step_tx3 = 30;
IDC_step_tx4 = 30;

Num_raw_W_temp = 10;%15; % Resolution of W values (Number of steps of beamforming value (W) from min to max)
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
cu_th = 2.650576505433320; %CU Rate Threshold values
CU1_th = cu_th; %CU 1 Rate Threshold values
CU2_th = cu_th; %CU 2 Rate Threshold values
CU3_th = cu_th; %CU 3 Rate Threshold values
CU4_th = cu_th; %CU 4 Rate Threshold values
CU5_th = cu_th; %CU 3 Rate Threshold values
CU6_th = cu_th; %CU 4 Rate Threshold values

for num_eu = 1:size(EU_TH_mat,2)
    
NumEU = num_eu;
EU_TH = EU_TH_mat(1,1:NumEU);

IDC_optm_path = []; % Optimum DC current matrix
W_opt_path = []; % Optimum Pre-coding matrix
EdB_logic_path = [];
E_dB_i_path=[];
r_eu_i_path=[];
I_DC_combination_mat=[];
No_of_IDC_selected_path=[];
Blockage_mat_test_01=[];

for path_step = 1:1:U_step % 556:U_step %1:1:U_step % Check for each steps

IDC_opt_temp=[]; % I_DC optimum temporay values
IDC_optm_sum_iter = []; % Summation of I_DC optimum temporay values 
W_opt_temp=[]; % W optimum temporay values 

r_cu = R_CU_mat{1,path_step}; % get CU x and y coordinates
r_eu = R_EU_mat{1,path_step}; % get EU x and y coordinates
R_CU = r_cu;
R_EU = [r_eu(:, 5), r_eu(:, 9), r_eu(:, 7:8), r_eu(:, 3:4), r_eu(:, 2), r_eu(:, 10), r_eu(:, 6), r_eu(:, 1)];
[H_CU_temp1, H_EU_temp1, U_blokage] = LOS_channelGain(T, R_CU, R_EU);% channel gain [Tx*Rx] - Cu and EU
H_CU = H_CU_temp1(1:NumCU, :);
H_EU = H_EU_temp1(1:NumEU, :);

U_blokage1 = 1-U_blokage;
Blockage_mat_test_01 = [Blockage_mat_test_01; U_blokage1];

% Using above 2 parameters, define new(filtered) DC current vector
I_DC_tx1_temp = linspace(IL_tx1,IU_tx1,IDC_step_tx1)'; I_DC_tx1 = I_DC_tx1_temp(2:end-1,:);
I_DC_tx2_temp = linspace(IL_tx2,IU_tx2,IDC_step_tx2)'; I_DC_tx2 = I_DC_tx2_temp(2:end-1,:);
I_DC_tx3_temp = linspace(IL_tx3,IU_tx3,IDC_step_tx3)'; I_DC_tx3 = I_DC_tx3_temp(2:end-1,:);
I_DC_tx4_temp = linspace(IL_tx4,IU_tx4,IDC_step_tx4)'; I_DC_tx4 = I_DC_tx4_temp(2:end-1,:);

%%%%%% Energy Users - EU %%%%%%
[p, q, r, s] = ndgrid(I_DC_tx1, I_DC_tx2, I_DC_tx3, I_DC_tx4); % get all I_DC combinations
I_DC_temp = [s(:),r(:),q(:),p(:)]; % re-order all combinations
P_IDC_temp = P_IDC.*I_DC_temp; % convert current -> Power

E=[]; % Initialize Energy matrix
for rx_EU = 1:size(H_EU,1) % Select each Eu one by one
    h_rx_EU = H_EU(rx_EU,:); % Extract necessary channel gains for the selected EU from 4 Txs
    h_P_DC = h_rx_EU.*P_IDC_temp; % calculate tempoary value = h*I_DC or h*P_DC
    E_temp = f*v_t.*sum(h_P_DC.*log(1 + h_P_DC./I_0),2); % Calculate Energy as in Eqn.
    E = [E, E_temp]; % Store them in E matrix [rax: for each current combinations, col: for each EU]
end
E_dB = 10*log10(E); % Convert to dB values

E_dB_i = max(E_dB); % 1st EU
E_dB_i_path = [E_dB_i_path; E_dB_i];
r_eu_i = R_EU(:)';
r_eu_i_path = [r_eu_i_path; r_eu_i];

EdB_logic = E_dB > EU_TH;  % Filtering E values from constraint #############
EdB_logic_path = [EdB_logic_path; sum(EdB_logic)];
sum_EdB_logic = sum(EdB_logic,2); % Summation energy of all 4 Txs
[x,y] = find(sum_EdB_logic==NumEU); % Check satisfied indexes of the Energy constraint for all 4 Txs
I_dc_selected = I_DC_temp(x,:); % Filtered all current combinations which satsfied the Energy constraint

% Check there are no any current combination to match the Energy constraint and if not code is terminated
I_dc_selected_flag = sum(sum(I_dc_selected)); 
if I_dc_selected_flag == 0
    fprintf('Energy constrint is not satisfied for any current combination \n');
    break;
end
sum_I_DC_opt_temp = sum(I_dc_selected,2); % get sum for each temp opt IDC 
[I_DC_min,ind_I_DC_min] = sort(sum_I_DC_opt_temp);

ind_I_DC_min = ind_I_DC_min(1:round(size(ind_I_DC_min,1)*0.5),:);
No_of_IDC_selected_path = [No_of_IDC_selected_path;size(ind_I_DC_min,1)];

for ind_i_dc_min_temp = 1:size(ind_I_DC_min,1)
ind_I_DC_opt_temp = ind_I_DC_min(ind_i_dc_min_temp);
IDC_optm_temp = I_dc_selected(ind_I_DC_opt_temp,:); %find IDC optimum temporary

W_TH_TX = min([IDC_optm_temp-[IL_tx1, IL_tx2, IL_tx3, IL_tx4]],[[IU_tx1, IU_tx1, IU_tx1, IU_tx1]-IDC_optm_temp]); % Get max value of W elements from DC current values

W_temp = ones(Num_raw_W_temp-1, NumT)*0.0001;
for w_ind_temp = 1:size(W_TH_TX,2)
    W_TH_TX_temp = W_TH_TX(w_ind_temp);
    W_tx_i_temp1 = linspace(0,W_TH_TX_temp,Num_raw_W_temp)';
    W_tx_i_temp = W_tx_i_temp1(2:end,:);
    W_temp(:,w_ind_temp) = W_tx_i_temp;
end

len_W = Num_raw_W_temp-1; % No.of raws in W_temp matrix
W_mat = [];%Initialize W matrix
for w1 = 1:len_W
    W1 = W_temp(w1,:);
    W1_mat = repmat(W1,len_W,1);
    W2_mat = W_temp;
    W_mat_temp = [W1_mat, W2_mat];
    W_mat = [W_mat; W_mat_temp];
    fprintf('Path number=%d, selected opt idc temp=%d/%d\n',path_step,ind_i_dc_min_temp,size(ind_I_DC_min,1));
end
W_mat = W_mat/NumCU;
rate_rx = [];
P_IDC_temp2 = P_IDC.*IDC_optm_temp;
%%%% Communication Users - CU
for Hcu_ind = 1:size(H_CU,1) % Select the CU
    H_CU_temp = H_CU(Hcu_ind,:);
    h_rx_CU = P_IDC_temp2.*H_CU_temp;
    w_desi_mat = W_mat(:,1:4);
    w_intf_mat = W_mat(:,5:8);
    
    % Rate calculation
    rate_temp = 0.5*log2(1 + (exp(1)/2*pi)*((w_desi_mat*h_rx_CU').^2)./(((w_intf_mat*h_rx_CU').^2) + noise_var));
	rate_rx = [rate_rx, rate_temp];
end

rate_logic_CU1 = rate_rx(:,1) >= CU1_th;
rate_logic_CU2 = rate_rx(:,2) >= CU2_th; 

rate_logic = rate_logic_CU1.*rate_logic_CU2;

if sum(rate_logic) ~= 0
    IDC_optm = IDC_optm_temp;
    % TO find W opt
    ind_rate_logic= find(rate_logic == 1); %Find rate index from logic matrix
    mid_rate_logic = round(length(ind_rate_logic)/2); % find middle
	ind_mid_rate_logic = ind_rate_logic(mid_rate_logic); % get middle index value from indexes set 
	W_opt = W_mat(ind_mid_rate_logic,:); % get W_opt raw
	W_OPT = reshape(W_opt,NumT,NumCU)'; % reshape W_opt as (no. of user * no. of tx)
    I_DC_combination_mat = [I_DC_combination_mat; ind_i_dc_min_temp];
	break;
end

if ind_i_dc_min_temp == size(ind_I_DC_min,1)
    fprintf('There are no any satisfied IDC combination for step no.: %d \n',path_step);
    IDC_optm = [];
end
end

disp(W_OPT);
W_opt_path = [W_opt_path; W_opt];

IDC_optm_path = [IDC_optm_path; IDC_optm];
sum_IDC_optm_path = sum(IDC_optm_path,2);
avg_sum_IDC_optm_path = mean(sum_IDC_optm_path);
end

IDC_optm_path_eu_i = [IDC_optm_path_eu_i, IDC_optm_path];
sum_IDC_optm_path_eu_i = [sum_IDC_optm_path_eu_i, sum_IDC_optm_path];
avg_sum_IDC_optm_path_eu_i = [avg_sum_IDC_optm_path_eu_i, avg_sum_IDC_optm_path];

end

save('OPT_IDC_cu_2', 'IDC_optm_path_eu_i', 'sum_IDC_optm_path_eu_i', 'avg_sum_IDC_optm_path_eu_i', '-v7.3');
toc;

figure(1);
plot(1:size(EU_TH_mat,2), avg_sum_IDC_optm_path_eu_i,'-ok','LineWidth',0.8);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;
tic;
load('Position_mat_Noise_var_0_03_test_01.mat');
U_step = size(R_CU_mat,2); % Number of user steps for CUs and EUs
VLC_settings_temp; % For all user position initializations

NumCU = 1; %size(R_CU,2); % Number of CUs ##############
NumT = size(T,2); % Number of Txs

IDC_optm_path_eu_i = [];
sum_IDC_optm_path_eu_i = [];
avg_sum_IDC_optm_path_eu_i = [];

% I_DC current values for 2 txs [Real IU=0.6A]
iu_tx = 6;
IL_tx1 = 0.4; IU_tx1 = iu_tx; %Tx-1 current upper and lower bounds
IL_tx2 = 0.4; IU_tx2 = iu_tx; %Tx-2 current upper and lower bounds
IL_tx3 = 0.4; IU_tx3 = iu_tx; %Tx-3 current upper and lower bounds
IL_tx4 = 0.4; IU_tx4 = iu_tx; %Tx-4 current upper and lower bounds

IDC_step_tx1 = 30;%50
IDC_step_tx2 = 30;
IDC_step_tx3 = 30;
IDC_step_tx4 = 30;

Num_raw_W_temp = 10;%15; % Resolution of W values (Number of steps of beamforming value (W) from min to max)
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
cu_th = 2.650576505433320; %CU Rate Threshold values
CU1_th = cu_th; %CU 1 Rate Threshold values
CU2_th = cu_th; %CU 2 Rate Threshold values
CU3_th = cu_th; %CU 3 Rate Threshold values
CU4_th = cu_th; %CU 4 Rate Threshold values
CU5_th = cu_th; %CU 3 Rate Threshold values
CU6_th = cu_th; %CU 4 Rate Threshold values

for num_eu = 1:size(EU_TH_mat,2)
    
NumEU = num_eu;
EU_TH = EU_TH_mat(1,1:NumEU);

IDC_optm_path = []; % Optimum DC current matrix
W_opt_path = []; % Optimum Pre-coding matrix
EdB_logic_path = [];
E_dB_i_path=[];
r_eu_i_path=[];
I_DC_combination_mat=[];
No_of_IDC_selected_path=[];
Blockage_mat_test_01=[];

for path_step = 1:1:U_step % 556:U_step %1:1:U_step % Check for each steps

IDC_opt_temp=[]; % I_DC optimum temporay values
IDC_optm_sum_iter = []; % Summation of I_DC optimum temporay values 
W_opt_temp=[]; % W optimum temporay values 

r_cu = R_CU_mat{1,path_step}; % get CU x and y coordinates
r_eu = R_EU_mat{1,path_step}; % get EU x and y coordinates
R_CU = r_cu;
R_EU = [r_eu(:, 5), r_eu(:, 9), r_eu(:, 7:8), r_eu(:, 3:4), r_eu(:, 2), r_eu(:, 10), r_eu(:, 6), r_eu(:, 1)];
[H_CU_temp1, H_EU_temp1, U_blokage] = LOS_channelGain(T, R_CU, R_EU);% channel gain [Tx*Rx] - Cu and EU
H_CU = H_CU_temp1(1:NumCU, :);
H_EU = H_EU_temp1(1:NumEU, :);

U_blokage1 = 1-U_blokage;
Blockage_mat_test_01 = [Blockage_mat_test_01; U_blokage1];

% Using above 2 parameters, define new(filtered) DC current vector
I_DC_tx1_temp = linspace(IL_tx1,IU_tx1,IDC_step_tx1)'; I_DC_tx1 = I_DC_tx1_temp(2:end-1,:);
I_DC_tx2_temp = linspace(IL_tx2,IU_tx2,IDC_step_tx2)'; I_DC_tx2 = I_DC_tx2_temp(2:end-1,:);
I_DC_tx3_temp = linspace(IL_tx3,IU_tx3,IDC_step_tx3)'; I_DC_tx3 = I_DC_tx3_temp(2:end-1,:);
I_DC_tx4_temp = linspace(IL_tx4,IU_tx4,IDC_step_tx4)'; I_DC_tx4 = I_DC_tx4_temp(2:end-1,:);

%%%%%% Energy Users - EU %%%%%%
[p, q, r, s] = ndgrid(I_DC_tx1, I_DC_tx2, I_DC_tx3, I_DC_tx4); % get all I_DC combinations
I_DC_temp = [s(:),r(:),q(:),p(:)]; % re-order all combinations
P_IDC_temp = P_IDC.*I_DC_temp; % convert current -> Power

E=[]; % Initialize Energy matrix
for rx_EU = 1:size(H_EU,1) % Select each Eu one by one
    h_rx_EU = H_EU(rx_EU,:); % Extract necessary channel gains for the selected EU from 4 Txs
    h_P_DC = h_rx_EU.*P_IDC_temp; % calculate tempoary value = h*I_DC or h*P_DC
    E_temp = f*v_t.*sum(h_P_DC.*log(1 + h_P_DC./I_0),2); % Calculate Energy as in Eqn.
    E = [E, E_temp]; % Store them in E matrix [rax: for each current combinations, col: for each EU]
end
E_dB = 10*log10(E); % Convert to dB values

E_dB_i = max(E_dB); % 1st EU
E_dB_i_path = [E_dB_i_path; E_dB_i];
r_eu_i = R_EU(:)';
r_eu_i_path = [r_eu_i_path; r_eu_i];

EdB_logic = E_dB > EU_TH;  % Filtering E values from constraint #############
EdB_logic_path = [EdB_logic_path; sum(EdB_logic)];
sum_EdB_logic = sum(EdB_logic,2); % Summation energy of all 4 Txs
[x,y] = find(sum_EdB_logic==NumEU); % Check satisfied indexes of the Energy constraint for all 4 Txs
I_dc_selected = I_DC_temp(x,:); % Filtered all current combinations which satsfied the Energy constraint

% Check there are no any current combination to match the Energy constraint and if not code is terminated
I_dc_selected_flag = sum(sum(I_dc_selected)); 
if I_dc_selected_flag == 0
    fprintf('Energy constrint is not satisfied for any current combination \n');
    break;
end
sum_I_DC_opt_temp = sum(I_dc_selected,2); % get sum for each temp opt IDC 
[I_DC_min,ind_I_DC_min] = sort(sum_I_DC_opt_temp);

ind_I_DC_min = ind_I_DC_min(1:round(size(ind_I_DC_min,1)*0.5),:);
No_of_IDC_selected_path = [No_of_IDC_selected_path;size(ind_I_DC_min,1)];

for ind_i_dc_min_temp = 1:size(ind_I_DC_min,1)
ind_I_DC_opt_temp = ind_I_DC_min(ind_i_dc_min_temp);
IDC_optm_temp = I_dc_selected(ind_I_DC_opt_temp,:); %find IDC optimum temporary

W_TH_TX = min([IDC_optm_temp-[IL_tx1, IL_tx2, IL_tx3, IL_tx4]],[[IU_tx1, IU_tx1, IU_tx1, IU_tx1]-IDC_optm_temp]); % Get max value of W elements from DC current values

W_temp = ones(Num_raw_W_temp-1, NumT)*0.0001;
for w_ind_temp = 1:size(W_TH_TX,2)
    W_TH_TX_temp = W_TH_TX(w_ind_temp);
    W_tx_i_temp1 = linspace(0,W_TH_TX_temp,Num_raw_W_temp)';
    W_tx_i_temp = W_tx_i_temp1(2:end,:);
    W_temp(:,w_ind_temp) = W_tx_i_temp;
end

len_W = Num_raw_W_temp-1; % No.of raws in W_temp matrix
W_mat = [];%Initialize W matrix
for w1 = 1:len_W
    W1 = W_temp(w1,1);
    W1_mat = repmat(W1,len_W,1);
    for w2 = 1:len_W
        W2 = W_temp(w2,2);
        W2_mat = repmat(W2,len_W,1);
        for w3 = 1:len_W
            W3 = W_temp(w3,3);
            W3_mat = repmat(W3,len_W,1);
            W4_mat = W_temp(:,4);   
            W_mat_temp = [W1_mat, W2_mat, W3_mat, W4_mat];
            W_mat = [W_mat; W_mat_temp];
            fprintf('Path number=%d, selected opt idc temp=%d/%d, \n',path_step,ind_i_dc_min_temp,size(ind_I_DC_min,1)); 
        end
    end
end
W_mat = W_mat/NumCU;
rate_rx = [];
P_IDC_temp2 = P_IDC.*IDC_optm_temp;
%%%% Communication Users - CU
for Hcu_ind = 1:size(H_CU,1) % Select the CU
    H_CU_temp = H_CU(Hcu_ind,:);
    h_rx_CU = P_IDC_temp2.*H_CU_temp;
    w_desi_mat = W_mat(:,1:4);
    
    % Rate calculation
    rate_temp = 0.5*log2(1 + (exp(1)/2*pi)*((w_desi_mat*h_rx_CU').^2)./(0 + noise_var));%((w_intf_mat*h_rx_CU').^2)
	rate_rx = [rate_rx, rate_temp];
	fprintf('CU_path_step=%d, Hcu_ind=%d \n',path_step, Hcu_ind);
end

rate_logic_CU1 = rate_rx(:,1) >= CU1_th;
rate_logic = rate_logic_CU1;

if sum(rate_logic) ~= 0
    IDC_optm = IDC_optm_temp;
    % TO find W opt
    ind_rate_logic= find(rate_logic == 1); %Find rate index from logic matrix
    mid_rate_logic = round(length(ind_rate_logic)/2); % find middle
	ind_mid_rate_logic = ind_rate_logic(mid_rate_logic); % get middle index value from indexes set 
	W_opt = W_mat(ind_mid_rate_logic,:); % get W_opt raw
	W_OPT = reshape(W_opt,NumT,NumCU)'; % reshape W_opt as (no. of user * no. of tx)
    I_DC_combination_mat = [I_DC_combination_mat; ind_i_dc_min_temp];
	break;
end

if ind_i_dc_min_temp == size(ind_I_DC_min,1)
    fprintf('There are no any satisfied IDC combination for step no.: %d \n',path_step);
    IDC_optm = [];
end
end

disp(W_OPT);
W_opt_path = [W_opt_path; W_opt];

IDC_optm_path = [IDC_optm_path; IDC_optm];
sum_IDC_optm_path = sum(IDC_optm_path,2);
avg_sum_IDC_optm_path = mean(sum_IDC_optm_path);
end

IDC_optm_path_eu_i = [IDC_optm_path_eu_i, IDC_optm_path];
sum_IDC_optm_path_eu_i = [sum_IDC_optm_path_eu_i, sum_IDC_optm_path];
avg_sum_IDC_optm_path_eu_i = [avg_sum_IDC_optm_path_eu_i, avg_sum_IDC_optm_path];

end

save('OPT_IDC_cu_1', 'IDC_optm_path_eu_i', 'sum_IDC_optm_path_eu_i', 'avg_sum_IDC_optm_path_eu_i', '-v7.3');
toc;

%##  MATLAB Codes to Calculate the LOS Channel Gain
function [H_CU, H_EU, U_blokage] = LOS_channelGain(T, R_CU, R_EU)

channel_settings;
Adet_CU = 1e-4; %photo detector physical area of a PD (1 cm^2)
Adet_EU = 4e-2; %photo detector physical area of a PD (300 cm^2)


Num_T = size(T,2);
Num_CU = size(R_CU,2);
Num_EU = size(R_EU,2);

h_blk_user = 1.5; % avg height of a man = 1.7 m
h_loc = h; % Height of the room
r_blk_user = 20/100; % avg width of a man = 30 cm

all_users = [R_CU, R_EU];
I_blockage = zeros(size(all_users,2),Num_T); % initialize d matrix
des_user_mat = [];
blk_user_mat = [];

% for ind_users = 1:size(all_users,2)
%     des_user = all_users(:,ind_users); % Select the desired user
%     des_user_mat = [des_user_mat; des_user];
%     blk_user = all_users;
%     blk_user(:,ind_users) = []; % Select rest of blockage users
%     blk_user_mat = [blk_user_mat; blk_user];
% end


for ind_users = 1:size(all_users,2)
    des_user_xy = all_users(:,ind_users); % Select the desired user
    blk_user = all_users;
    blk_user(:,ind_users) = []; % Select rest of blockage users
    
    for ind_blk_user = 1:size(blk_user,2)
        blk_user_xy = blk_user(:,ind_blk_user);
        for ind_T = 1:Num_T
            T_xy = T(:,ind_T);
            T_min_des_user = T_xy - des_user_xy;
            T_mul_des_user = T_xy .* flipud(des_user_xy); %swap EU

            d_temp = abs(T_min_des_user(2,1)*blk_user_xy(1,1) - T_min_des_user(1,1)*blk_user_xy(2,1) + T_mul_des_user(1,1) - T_mul_des_user(2,1)) / norm(T_min_des_user);

            if d_temp > r_blk_user % if true, no blockage
                I_temp = 1;
            else % if not true, there is a blockage through top view
                theta = atan(h_loc/norm(T_min_des_user)); % calculate theta matrix
                m_rd = sqrt(r_blk_user^2 - d_temp^2);
                alpha = atan(h_blk_user / (sqrt((norm(T_min_des_user))^2 - d_temp^2)-m_rd));
                if theta > alpha
                    I_temp = 1;
                else
                    I_temp = 0;
                end
            end
            I_blockage(ind_users,ind_T) = I_temp;
        end
    end
end

U_blokage = I_blockage(:)';

% For CUs channel
H_rec_cu=[];
for rx = 1:size(R_CU,2)
    distance_2D = sqrt(sum((R_CU(:,rx)-T).^2));
    distance_3D =  sqrt(distance_2D.^2 + h^2);
    cosphi_A1 = h./distance_3D; % angle vector
    H_A1 = (m+1)*Adet_CU.*cosphi_A1.^(m+1)./(2*pi.*distance_3D.^2); % ##channel DC gain for source 1
    H_rec_cu = [H_rec_cu; H_A1.*Ts.*G_Con];
end
%disp(H_rec);

% For EUs channel
H_rec_eu=[];
for rx = 1:size(R_EU,2)
    distance_2D = sqrt(sum((R_EU(:,rx)-T).^2));
    distance_3D =  sqrt(distance_2D.^2 + h^2);
    cosphi_A1 = h./distance_3D; % angle vector
    H_A1 = (m+1)*Adet_EU.*cosphi_A1.^(m+1)./(2*pi.*distance_3D.^2); % ##channel DC gain for source 1
    H_rec_eu = [H_rec_eu; H_A1.*Ts.*G_Con];
end

H_CU_EU = [H_rec_cu ;H_rec_eu];
H_CU_EU_blk = H_CU_EU .* I_blockage;
H_CU = H_CU_EU_blk(1:Num_CU,:);
H_EU = H_CU_EU_blk(Num_CU+1:end,:);
end
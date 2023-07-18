%%%%% Initialize Tx, CU and EU positions

% Tx position
XT1=2.5; YT1= 2.5; % 1 tx positions
XT2=-2.5; YT2=2.5; % 2 tx positions
XT3=-2.5; YT3=-2.5; % 1 tx positions
XT4=2.5; YT4=-2.5; % 2 tx positions

T = [[XT1; YT1], [XT2; YT2], [XT3; YT3], [XT4; YT4]];

figure(1);
for fig=1:U_step
R_CU = R_CU_mat{1,fig};
R_EU = R_EU_mat{1,fig};
 plot(T(1,:),T(2,:),'or','MarkerSize',10); % Plot EU location
 hold on;
 plot(R_CU(1,:),R_CU(2,:),'*g','MarkerSize',10); % Plot EU location
 plot(R_EU(1,:),R_EU(2,:),'+b','MarkerSize',10); % Plot EU location

 xlim([-5,5]); ylim([-5,5]); grid on;
 legend('Tx','R-CU', 'R-EU','Location', 'Best');
end
 hold off;

noise_var = 1e-15; % sigma_n_th = 2eB(I_rp + I_bgI_2) + sigma_th^2 and I_rp = SUM(I_DC*h_kn)
f = 0.75; % Fill factor(C_FF)
v_t = 0.025; % Thermal Voltage (25mV)
I_0 = 1e-9; % Dark saturation current of PD (I_0) 


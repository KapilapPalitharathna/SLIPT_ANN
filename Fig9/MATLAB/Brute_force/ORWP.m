function [P, theta, Omega] = ORWP()

room    = [5 5];                        % Room size
n       = 2;                            % n-th instantaneous location for calculations
k       = 1;                            % k-th random user location
P_i     = [0 0];                        % Initial position
P_f     = [5 5];                        % Final position
P_k     = P_i;                          % Initial position
N_run   = 100;                          % Number of runs 
v       = 0.5;                          % Speed of the UE (m/s)
Tc      = 0.13;                         % coherence time of the polar angle (130 ms)
Ts      = 0.0013;                       % sampling time (1.3 ms)
mu      = 29.67;                        % mean value of the polar angle (29.67 degrees)
sigma   = 7.78;                         % standard deviation of the polar angle (7.78 degrees)
theta_n_1 = 0;
theta   = theta_n_1;
D       = norm(P_f-P_i);
Omega   = [];
P       = [];
Omega_n = atan((P_f(2)-P_i(2))/(P_f(1)-P_i(1)));
for k = 1:N_run
    P_n_1   = P_k;
    t_move  = 0;
    while t_move<=(D/v)
        P_n     = [P_n_1(1)+v*Ts*cos(Omega_n) P_n_1(2)+v*Ts*sin(Omega_n)];
        P       = [P; P_n(1) P_n(2)];
        theta_n = corr_gaussian_RP(theta_n_1);
        theta   = [theta; theta_n];
        Omega   = [Omega; Omega_n];
        theta_n_1 = theta_n;
        t_move  = t_move + Ts;
        P_n_1   = P_n;
    end
end
save('Data/positions_angles.mat','Omega','P','theta')
end

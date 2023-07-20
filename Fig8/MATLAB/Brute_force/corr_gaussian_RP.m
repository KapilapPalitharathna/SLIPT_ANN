function theta_n = corr_gaussian_RP(theta_n_1)
Tc      = 0.13;     % coherence time of the polar angle (130 ms)
Ts      = 0.00013; % sampling time (130 us)
mu      = 29.67;    % mean value of the polar angle (29.67 degrees)
sigma   = 7.78;
c1      = 0.05^(Ts/Tc);
c0      = (1-c1)*mu;
sigma_w = sqrt(1-c1^2)*sigma;
w_n = wgn(1,1,10*log(sigma_w^2));
theta_n = c0+c1*theta_n_1+w_n;
theta_n = theta_n*pi/180;
end


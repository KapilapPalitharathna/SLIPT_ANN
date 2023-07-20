H_full =zeros(4*floor(num/3),4);
pos_r_t=zeros(4,3);
for i =1:floor(num/3)
pos_r_t(1,1) = (-d_rx/2)*cos(theta(i))*cos(Omega(i))+P(i,1);
pos_r_t(1,2) = (-d_rx/2)*cos(theta(i))*sin(Omega(i))+P(i,2);
pos_r_t(1,3) = (d_rx/2)*sin(theta(i));
pos_r_t(2,1) = (-d_rx/2)*cos(theta(i))*cos(Omega(i))+P(i,1);
pos_r_t(2,2) = (+d_rx/2)*cos(theta(i))*sin(Omega(i))+P(i,2);
pos_r_t(2,3) = (d_rx/2)*sin(theta(i));
pos_r_t(3,1) = (+d_rx/2)*cos(theta(i))*cos(Omega(i))+P(i,1);
pos_r_t(3,2) = (-d_rx/2)*cos(theta(i))*sin(Omega(i))+P(i,2);
pos_r_t(3,3) = (d_rx/2)*sin(theta(i));
pos_r_t(4,1) = (+d_rx/2)*cos(theta(i))*cos(Omega(i))+P(i,1);
pos_r_t(4,2) = (+d_rx/2)*cos(theta(i))*sin(Omega(i))+P(i,2);
pos_r_t(4,3) = (d_rx/2)*sin(theta(i));

H_full((4*(i-1)+1):(4*(i-1)+4),1:4) = Channel_gain(pos_t, pos_r_t, [sin(theta(i))*cos(Omega(i)) sin(theta(i))*sin(Omega(i)) cos(theta(i))], phi_half, FOV, A_pd);
end
save('Data/H_full','H_full')
%%%%% Channel parameters
theta = 60; % semi-angle(in degrees) at half power
m = -log10(2)/log10(cosd(theta)); %Lambertian order of emission
% P_total = 20; %transmitted optical power by individual LED
Ts = 1; %gain of an optical filter; ignore if no filter is used
refractive_index = 1.5; %refractive index of a lens at a PD; ignore if no lens is used
FOV = 60*pi/180; %FOV of a receiver(in radians)
G_Con = (refractive_index^2)/sin(FOV); % ## gain of an optical concentrator; ignore if no lens is used
lx=10; ly=10; lz=3; % room dimension in metre
h = 3; %the distance between source and receiver plane

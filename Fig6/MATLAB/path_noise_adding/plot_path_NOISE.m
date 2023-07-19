clear; clc;

no_path_step = 559;

Traj1 = load('path01.dat');
Traj1 = Traj1(1:no_path_step,1:2)+(randn(no_path_step,2)*0.03);
plot(Traj1(:,1),Traj1(:,2),'.');
hold on;

Traj2 = load('path02.dat');
Traj2 = Traj2(1:no_path_step,1:2)+(randn(no_path_step,2)*0.03);
plot(Traj2(:,1),Traj2(:,2),'.');

Traj3 = load('path03.dat');
Traj3 = Traj3(1:no_path_step,1:2)+(randn(no_path_step,2)*0.03);
plot(Traj3(:,1),Traj3(:,2),'.');

Traj4 = load('path04.dat');
Traj4 = Traj4(1:no_path_step,1:2)+(randn(no_path_step,2)*0.03);
plot(Traj4(:,1),Traj4(:,2),'.');

Traj5 = load('path05.dat');
Traj5 = Traj5(1:no_path_step,1:2)+(randn(no_path_step,2)*0.03);
plot(Traj5(:,1),Traj5(:,2),'+');

Traj6 = load('path06.dat');
Traj6 = Traj6(1:no_path_step,1:2)+(randn(no_path_step,2)*0.03);
plot(Traj6(:,1),Traj6(:,2),'+');

Traj7 = load('path07.dat');
Traj7 = flipud(Traj7);
Traj7 = Traj7(1:no_path_step,1:2)+(randn(no_path_step,2)*0.03);
plot(Traj7(:,1),Traj7(:,2),'+');

Traj8 = load('path08.dat');
Traj8 = flipud(Traj8);
Traj8 = Traj8(1:no_path_step,1:2)+(randn(no_path_step,2)*0.03);
plot(Traj8(:,1),Traj8(:,2),'+');
hold off;

xlim([-5,5]);ylim([-5,5]);
save('Position_Data_test_var_0_03_test_01','Traj1', 'Traj2', 'Traj3', 'Traj4', 'Traj5', 'Traj6', 'Traj7', 'Traj8', '-v7.3');
title('Approximated path as in AR');
legend('CU-1','CU-2','CU-3','CU-4','EU-1','EU-2','EU-3','EU-4');

R_CU_mat=cell(1,no_path_step);
R_EU_mat=cell(1,no_path_step);

for x=1:size(Traj1,1)
    temp1 = [Traj1(x,:)', Traj2(x,:)', Traj3(x,:)', Traj4(x,:)'];
    R_CU_mat{1,x} = temp1;
    
    temp2 = [Traj5(x,:)', Traj6(x,:)', Traj7(x,:)', Traj8(x,:)'];
    R_EU_mat{1,x} = temp2;
end

save('Position_mat_Noise_var_0_03_test_01', 'R_CU_mat', 'R_EU_mat', '-v7.3');

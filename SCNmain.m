%% I. 清空环境变量
clear all 
warning off;
clc
format long;%设置初始损失
%%
X = xlsread('input');
Y = xlsread('output');
%% 导入数据
% 训练集—
%temp = randperm(size(X,1));%1代表多少行，2代表多少列
P_train= X((1:80),:)';%冒号代表取出来是整行或者整列，'代表转置
P_test = X((81:end),:)';
M1= size(P_train,2);
% 测试集—
T_train= Y((1:80),:)';
T_test = Y((81:end),:)';
N = size(T_test,2);
%% 数据归一化
[p_train, ps_input] = mapminmax(P_train,0,1);
p_test = mapminmax('apply',P_test,ps_input);

[t_train, ps_output] = mapminmax(T_train,0,1);
t_test = mapminmax('apply',T_test,ps_output);
%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';
%% 定义目标函数和其变量
Function_name = 'branin'; % 定义你的目标函数（这里是"Branin"，一个目标）
[Vmin, Vmax, nV, Function] = ObjectiveFunction(Function_name);
ObjectiveFunction = @(x) Function(x);
%% 定义算法参数
ns = 10;                                              %% 星体的数量
SN = 1;                                               %% 信噪比
%%注意：(ns*SN) = TS算法的种群数量
maxcycle = 100;                                        %% 最大迭代次数
%% Transit Search优化算法
disp('Transit Search正在运行...')
[Bests] = TransitSearch(ObjectiveFunction, Vmin, Vmax, nV, ns, SN, maxcycle);
%% 绘制迭代曲线
Best_Cost = Bests(maxcycle).Cost;
Best_Solution = Bests(maxcycle).Location;
y=zeros(maxcycle,1);
for i = 1:maxcycle
    y(i,1) = Bests(i).Cost;
end
figure
plot(y,'linewidth',1.5);
grid on
xlabel('迭代次数')
ylabel('适应度函数')
title('TransitSearch-SCN收敛曲线')
disp(['寻优得到的最佳alpha参数:',num2str(abs(Best_Solution(1)))]);
%% 参数设置
L_max = round(Best_Solution(3));                    % 最大隐藏层神经元个数
tol = Best_Solution(2);                             % 容忍度,目标误差
T_max = 100;                                        % 随机配置的最大次数
Lambdas = [0.5, 1, 5, 10, ...
    30, 50, 100, 150, 200, 250];                    % 随机权重范围,线性网格搜索
r =  [ 0.9, 0.99, 0.999, ...
    0.9999, 0.99999, 0.999999];                     % 配置参数
nB = 1;                                             % 批大小 ,不可以修改 

%% 最优alpha参数赋值
best_alpha = Best_Solution(1); 
%% 模型初始化
M = SCN(L_max, T_max, tol, Lambdas, r , nB,best_alpha);
disp(M);
%% 模型训练
% M 是训练的模型
% per 包含相对于递增 L 的训练误差
[M, per] = M.Regression(p_train, t_train);
disp(M);
%% 模型预测
t_sim1 = M.GetOutput(p_train);
t_sim2 = M.GetOutput(p_test);
%% 数据反归一化
T_sim1 = mapminmax('reverse',t_sim1,ps_output);
T_sim2 = mapminmax('reverse',t_sim2,ps_output);

%%  均方根误差 RMSE
error1 = sqrt(sum((T_sim1' - T_train).^2)./M1);
error2 = sqrt(sum((T_test - T_sim2').^2)./N);

%%
%决定系数
R1 = sqrt(1 - (sum((T_sim1' - T_train).^2) / sum((T_sim1' - mean(T_train)).^2)));

R2 = sqrt(1 - (sum((T_sim2' - T_test).^2) / sum((T_sim2' - mean(T_test)).^2)));

%%
%均方误差 MSE
mse1 = sum((T_sim1' - T_train).^2)./M1;
mse2 = sum((T_sim2' - T_test).^2)./N;
%%
%RPD 剩余预测残差
SE1=std(T_sim1'-T_train);
RPD=std(T_train)/SE1;

SE=std(T_sim2'-T_test);
RPD1=std(T_test)/SE;
%% 平均绝对误差MAE
MAE1 = mean(abs(T_train - T_sim1'));
MAE2 = mean(abs(T_test - T_sim2'));
%% 平均绝对百分比误差MAPE
MAPE1 = mean(abs((T_train - T_sim1')./T_train));
MAPE2 = mean(abs((T_test - T_sim2')./T_test));
%%  训练集绘图
figure
plot(1:M1,T_train,'r-*',1:M1,T_sim1,'b-o','LineWidth',1.5)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'训练集预测结果对比';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD) ')' ]};
title(string)
%% 预测集绘图
figure
plot(1:N,T_test,'r-*',1:N,T_sim2,'b-o','LineWidth',1.5)
legend('真实值','预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD1= ' num2str(RPD1) ')']};
title(string)
%% 训练误差结果
figure;
plot(per.Error, 'r-*','LineWidth',1.5)
xlabel('迭代次数');
ylabel('RMSE');
legend('训练的 RMSE变化');
string={'训练误差收敛曲线'};
title(string)
%% 绘制线性拟合图
%% 训练集拟合效果图
figure
plot(T_train,T_sim1,'*r');
xlabel('真实值')
ylabel('预测值')
string = {'训练集效果图';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% 预测集拟合效果图
figure
plot(T_test,T_sim2,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'测试集效果图';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% 求平均
R3=(R1+R2)./2;
error3=(error1+error2)./2;
%% 总数据线性预测拟合图
tsim=[T_sim1',T_sim2']'
figure
plot(Y,tsim,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'所有样本拟合预测图';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
%% 打印出评价指标
disp(['-----------------------误差计算--------------------------'])
disp(['随机配置网络TransitSearch-SCN的预测集的评价结果如下所示：'])
disp(['平均绝对误差MAE为：',num2str(MAE2)])
disp(['均方误差MSE为：       ',num2str(mse2)])
disp(['均方根误差RMSEP为：  ',num2str(error2)])
disp(['决定系数R^2为：  ',num2str(R2)])
disp(['剩余预测残差RPD为：  ',num2str(RPD1)])
disp(['平均绝对百分比误差MAPE为：  ',num2str(MAPE2)])
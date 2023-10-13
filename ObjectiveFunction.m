function [Vmin, Vmax, nV, Function] = ObjectiveFunction(F)
    switch F
        case 'branin'
            Function = @objectiveFunction1;
            nV = 3;                                              %% 有几个需要优化的参数就是几维
            Vmin = [0.01,0.001,10];                              %% 寻优参数下限;
            Vmax = [1,0.01,300];                                 %% 寻优参数上限;
    end
end

% 基准函数名称： objectiveFunction1
function [fitness] = objectiveFunction1(x)
%% 参数
P_train = evalin('base', 'p_train');
T_train = evalin('base', 't_train');
P_test = evalin('base', 'p_test');
T_test = evalin('base', 't_test');
alpha = abs(x(1));
tol   = abs(x(2));
L_max = round(abs(x(3)));
%% 参数设置
% L_max = 250;                    % 最大隐藏层神经元个数
% tol = 0.01;                    % 容忍度,目标误差
T_max = 100;                    % 随机配置的最大次数
Lambdas = [0.5, 1, 5, 10, ...
    30, 50, 100, 150, 200, 250];% 随机权重范围,线性网格搜索
r =  [ 0.9, 0.99, 0.999, ...
    0.9999, 0.99999, 0.999999]; % 配置参数
nB = 1;       % 批大小 ,不可以修改

M = SCN(L_max, T_max, tol, Lambdas, r , nB, alpha);
%% 仿真测试
[M, ~] = M.Regression(P_train, T_train);
T_sim1 = M.GetOutput(P_train);
T_sim2 = M.GetOutput(P_test);

error1 = T_sim1 - T_train;
error2 = T_sim2 - T_test;
%% 误差
fitness = mse(error1)+mse(error2);
end


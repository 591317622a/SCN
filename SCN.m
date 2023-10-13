%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stochastic Configuration Netsworks Class (Matlab)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2017
classdef SCN
    properties
        Name = 'Stochastic Configuration Networks';
        version = '1.0 beta';
        % Basic parameters (networks structure)
        L       % hidden node number / start with 1
        W       % input weight matrix
        b       % hidden layer bias vector
        Beta    % output weight vector
        % Configurational parameters
        r       % regularization parameter 正则化参数
        tol     % tolerance
        Lambdas % random weights range, linear grid search 线性网格搜索法思想
        L_max   % maximum number of hidden neurons
        T_max   % Maximum times of random configurations
        % Else
        nB = 1 % how many node need to be added in the network in one loop
        verbose = 50 % display frequency
        COST = 0   % final error
        alpha      % 正则化参数
    end
    %% Funcitons and algorithm
    methods
        %% Initialize one SCN model
        function obj = SCN(L_max, T_max, tol, Lambdas, r , nB, alpha)
            format long; % 控制输出格式为16位有效数字
            
            obj.L = 1;
  
            if ~exist('L_max', 'var') || isempty(L_max) % 判断用户是否定义了变量L_max的值
                obj.L_max = 100;
            else
                obj.L_max = L_max;
                if L_max > 5000
                    obj.verbose = 500; % does not need too many output
                end
            end
            if ~exist('T_max', 'var') || isempty(T_max)
                obj.T_max=  100;
            else
                obj.T_max = T_max;
            end
            if ~exist('tol', 'var') || isempty(tol)
                obj.tol=  1e-4;
            else
                obj.tol = tol;
            end
            if ~exist('Lambdas', 'var') || isempty(Lambdas)
                obj.Lambdas=  [0.5, 1, 3, 5, 7, 9, 15, 25, 50, 100, 150, 200];
            else
                obj.Lambdas = Lambdas;
            end
            if ~exist('r', 'var') || isempty(r)
                obj.r =  [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.99999];
            else
                obj.r = r;
            end
            if ~exist('nB', 'var') || isempty(nB)
                obj.nB = 1;
            else
                obj.nB = nB;
            end
            if ~exist('alpha', 'var') || isempty(alpha)
                obj.alpha = 0.5;
            else
                obj.alpha = alpha;
            end
        end
                
        %% inequality equation return the ksi/自定义函数，解决增加节点使误差越来越小导致的新权值和偏差设定困难的问题
        function  [obj, ksi] = InequalityEq(obj, eq, gk, r_L)
            ksi = ((eq'*gk)^2)/(gk'*gk) - (1 - r_L)*(eq'*eq);
        end
        %% Search for {WB,bB} of nB nodes
        function [WB, bB, Flag] = SC_Search(obj, X, E0)
            Flag =  0;% 0: continue; 1: stop; return a good node /or stop training by set Flag = 1
            WB  = [];
            bB  = [];
            [~,d] = size(X); % Get Sample and feature number
            [~,m] = size(E0);
            % Linear search for better nodes
            C = []; % container of ksi
            for i_Lambdas = 1: length(obj.Lambdas)  % i index lambda
                Lambda = obj.Lambdas(i_Lambdas);    % Get the random weight and bias range
                % Generate candidates T_max vectors of w and b for selection
                WT = Lambda*( 2*rand(d, [obj.T_max])-1 ); % WW is d-by-T_max 确保随机数范围对称
                bT = Lambda*( 2*rand(1, [obj.T_max])-1 ); % bb is 1-by-T_max
                HT = logsig(bsxfun(@plus, X*WT, bT)); % logsig是隐层的sigmoid函数，bsxfun是对不同维矩阵进行操作的函数，plus代表+
                for i_r = 1:length(obj.r)
                    r_L = obj.r(i_r); % get the regularization value
                    % calculate the Ksi value
                    for t = 1: obj.T_max % searching one by one
                        % Calculate H_t
                        H_t = HT(:,t); % H_t隐层节点的输出
                        % Calculate ksi_1 ... ksi_m
                        ksi_m = zeros(1, m); % m为输出节点数量
                        for i_m = 1:m                            
                            eq = E0(:,i_m);
                            gk = H_t;
                            [obj, ksi_m(i_m)] = obj.InequalityEq(eq, gk, r_L);
                        end
                        Ksi_t = sum(ksi_m);                        
                        if min(ksi_m) > 0
                            C = cat(2,C, Ksi_t); % cat(2,a,b)相当于[a,b],cat(1,a,b)相当于[a;b]
                            WB  = cat(2, WB, WT(:,t)); % 获得当前节点的w和b值
                            bB  = cat(2, bB, bT(:,t));
                        end
                    end
                    nC = length(C); 
                    if nC >= obj.nB 
                        break; % r loop
                    else
                        continue; % 如果该正则化参数训练最大次数时扔不满足，则取下一个正则化参数...
                    end
                end %(r)
                if nC >= obj.nB 
                    break; % lambda loop
                else
                    continue; % 如果正则化参数都不满足，则取下一个随机权范围
                end
            end % (lambda)
            % Return the good node / or stop the training.
            if nC>= obj.nB
                [~, I] = sort(C, 'descend'); % 对c进行降序排列
                I_nb = I(1:obj.nB);
                WB = WB(:, I_nb); % 取最好的w和b
                bB = bB(:, I_nb);
            end
            if nC == 0 || nC<obj.nB % discard w b 抛弃
                disp('End Searching ...');
                Flag = 1;
            end
        end
        
        %% Add nodes to the model
        function obj = AddNodes(obj, w_L, b_L)
            obj.W = cat(2,obj.W, w_L);
            obj.b = cat(2,obj.b, b_L);
            obj.L = length(obj.b);
        end
        
        %% Compute the Beta, Output, ErrorVector and Cost
        function [obj, O, E, Error] = UpgradeSCN(obj, X, T)
            H = obj.GetH(X); % 求隐层输出
            %%%%正则化参数%%%
            %alpha=0.5;
            obj = obj.ComputeBeta(H,T,obj.alpha); % 求输出权值
            O = H*obj.Beta; % 求网络输出
            E = T - O; % 求出实际输出与网络输出之差
            Error =  Tools.RMSE(E); % 求出误差准则函数值
            obj.COST = Error;
        end     
        
        %% ComputeBeta
        function [obj, Beta] = ComputeBeta(obj, H, T,alpha)
            [a,b]=size(H);
            Beta = pinv(H+alpha*eye(a,b))*T; %pinv为伪逆，最小二乘法的广义逆矩阵求法，SCN-3
            obj.Beta = Beta;
        end              
        %% Regression
        function [obj, per] = Regression(obj, X, T)             
            per.Error = [];
            E = T;
            Error =  Tools.RMSE(E);
            disp(obj.Name);
            while (obj.L < obj.L_max) && (Error > obj.tol) % 隐层节点数小于最大节点数且误差不满足要求的时候           
                if mod(obj.L, obj.verbose) == 0 % mod取余函数
                    fprintf('L:%d\t\tRMSE:%.6f \r', obj.L, Error ); % 每增加五十个节点输出一次误差值
                end
                [w_L, b_L, Flag] = SC_Search(obj, X, E);% Search for candidate node / Hidden Parameters
                if Flag == 1
                    break;% could not find enough node
                end
                obj = AddNodes(obj, w_L, b_L);                 
                [obj, ~ , E, Error ] = obj.UpgradeSCN(X, T); % Calculate Beta/ Update all                
                %log
                per.Error = cat(2, per.Error, repmat(Error, 1, obj.nB)); % 可以直接写Error，代替repmat repmat为将Error复制1*nB块
            end% while
            fprintf('#L:%d\t\tRMSE:%.6f \r', obj.L, Error );
            disp(repmat('*', 1,30));
        end
        
        %% Output Matrix of hidden layer
        function H = GetH(obj, X)
            H =  obj.ActivationFun(X);
        end
        % Sigmoid function
        function H = ActivationFun(obj,  X)
            H = logsig(bsxfun(@plus, X*[obj.W],[obj.b]));              
        end
        %% Get Output
        function O = GetOutput(obj, X)
            H = obj.GetH(X); % 隐层输出
            O = H*[obj.Beta]; % 网络输出
        end
        %% Get Label
        function O = GetLabel(obj, X)
            O = GetOutput(obj, X);
            O = Tools.OneHotMatrix(O); % 分类，将每次输出的最大值变为1，其余输出值为零
        end
        %% Get Accuracy
        function [Rate, O] = GetAccuracy(obj, X, T)
            O = obj.GetLabel(X);
            Rate = 1- confusion(T',O'); % confusion为输出错误率
        end
        %% Get Error, Output and Hidden Matrix
        function [Error, O, H, E] = GetResult(obj, X, T)
            % X, T are test data or validation data
            H = obj.GetH(X);
            O = H*(obj.Beta);
            E = T - O;
            Error =  Tools.RMSE(E);
        end
 
    end % methods
end % class

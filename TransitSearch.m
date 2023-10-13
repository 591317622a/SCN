function [Bests] = TransitSearch (CostFunction,Vmin,Vmax,nV,ns,SN,maxcycle)
%% 初始化
Empty.Location = [];
Empty.Cost = inf;
Galaxy_Center = repmat (Empty, 1, 1);
region = repmat (Empty, ns*SN, 1);
selected_regions = repmat (Empty, ns, 1);
Stars = repmat (Empty, ns, 1);
Stars_sorted = zeros(ns,1);
Ranks = 1:1:ns;
Stars_Ranks = zeros(ns,1);
Luminosity = zeros(ns,1);
Star_RanksNormal = zeros(ns,1);
Distance = zeros(ns,1);
Transit0 = zeros(ns,1);
SN_P = repmat (Empty, SN, 1);
Bests=region;
if length(Vmin) ~= nV
    Vmin=Vmin*ones(1,nV);
    Vmax=Vmax*ones(1,nV);
end
%% 银河系阶段

% 银河系中心的初始位置
Galaxy_Center.Location = unifrnd(Vmin,Vmax,1,nV);
Galaxy_Center.Cost = CostFunction(Galaxy_Center.Location);

% 银河系居住区
for l = 1:(ns*SN)
    zone = randi(2);
    if zone ==1
        difference = rand().*(Galaxy_Center.Location)-(unifrnd(Vmin,Vmax,1,nV));
    else
        difference = rand().*(Galaxy_Center.Location)+(unifrnd(Vmin,Vmax,1,nV));
    end
    Noise = ((rand(1,nV)).^3).*(unifrnd(Vmin,Vmax,1,nV));
    region(l).Location = Galaxy_Center.Location + difference - Noise;
    region(l).Location = max(region(l).Location, Vmin);
    region(l).Location = min(region(l).Location, Vmax);
    region(l).Cost = CostFunction(region(l).Location);
end
% 从银河系的银河栖息区挑选恒星的百分比
[Sort,index]=sort([region.Cost]);
for i = 1:ns
    selected_regions(i) = region(index(1,i));
    for k = 1:SN
        zone = randi(2);
        if zone ==1
            difference = rand().*(selected_regions(i).Location)-rand().*(unifrnd(Vmin,Vmax,1,nV));
        else
            difference = rand().*(selected_regions(i).Location)+rand().*(unifrnd(Vmin,Vmax,1,nV));
        end
        Noise = ((rand(1,nV)).^3).*(unifrnd(Vmin,Vmax,1,nV));
        new.Location = selected_regions(i).Location + difference - Noise;
        new.Location = max(new.Location, Vmin);
        new.Location = min(new.Location, Vmax);
        new.Cost = CostFunction(new.Location);
        if new.Cost < Stars(i).Cost
            Stars(i) = new;
        end
    end
end
% 最佳行星的初始位置（起点：其恒星）
Best_Planets = Stars;

%  最佳星球的格式
[Sort,index]=sort([Best_Planets.Cost]);
Best_Planet = Best_Planets(index(1,1));

% 望远镜位置
Telescope.Location = unifrnd(Vmin,Vmax,1,nV);

%恒星光度的测定
for i = 1:ns
    Stars_sorted(i,1) = Stars(i).Cost;
end
Stars_sorted = sort (Stars_sorted);
for i = 1:ns
    for ii = 1:ns
        if Stars(i).Cost == Stars_sorted(ii,1)
            Stars_Ranks(i,1) = Ranks(1,ii);
            Star_RanksNormal(i,1) = (Stars_Ranks(i,1))./ns;
        end
    end
    Distance(i,1) = sum((Stars(i).Location-Telescope.Location).^2).^0.5;
    Luminosity(i,1) = Star_RanksNormal(i,1)/((Distance(i,1))^2);
end
Luminosity_new = Luminosity;
Stars2 = Stars;
%% TS 算法循环
for it = 1:maxcycle
    
    %% 过渡阶段
    Transit = Transit0;
    Luminosity = Luminosity_new;
    
    for i = 1:ns
        difference = (2*rand()-1).*(Stars(i).Location);
        Noise = ((rand(1,nV)).^3).*(unifrnd(Vmin,Vmax,1,nV));
        Stars2(i).Location = Stars(i).Location + difference - Noise;
        Stars2(i).Location = max(Stars2(i).Location, Vmin);
        Stars2(i).Location = min(Stars2(i).Location, Vmax);
        Stars2(i).Cost = CostFunction(Stars2(i).Location);
    end
    
    for i = 1:ns
        Stars_sorted(i,1) = Stars2(i).Cost;
    end
    Stars_sorted = sort (Stars_sorted);
    for i = 1:ns
        for ii = 1:ns
            if Stars2(i).Cost == Stars_sorted(ii,1)
                Stars_Ranks(i,1) = Ranks(1,ii);
                Star_RanksNormal(i,1) = (Stars_Ranks(i,1))./ns;
            end
        end
        Distance(i,1) = sum((Stars2(i).Location-Telescope.Location).^2).^0.5;
        Luminosity_new(i,1) = Star_RanksNormal(i,1)/((Distance(i,1))^2);
        if Luminosity_new(i,1) < Luminosity(i,1)
            Transit (i,1) = 1;      % Has transit been observed?  0 = No; 1 = Yes
        end
    end
    Stars = Stars2;
    
      %% 定位阶段（勘探）
    for i = 1:ns
        if Transit (i,1) == 1
            
               % 确定行星位置
            Luminosity_Ratio = Luminosity_new(i,1)/Luminosity(i,1);
            Planet.Location = (rand().*Telescope.Location + Luminosity_Ratio.*Stars(i).Location)./2;
            
            for k = 1:SN
                zone = randi(3);
                if zone ==1
                    new.Location = Planet.Location - (2*rand()-1).*(unifrnd(Vmin,Vmax,1,nV));
                elseif zone ==2
                    new.Location = Planet.Location + (2*rand()-1).*(unifrnd(Vmin,Vmax,1,nV));
                else
                    new.Location = Planet.Location + (2.*rand(1,nV)-1).*(unifrnd(Vmin,Vmax,1,nV));
                end
                new.Location = max(new.Location, Vmin);
                new.Location = min(new.Location, Vmax);
                %                             new.Cost = CostFunction(new.Location);
                SN_P(k) = new;
            end
            SUM = 0;
            for k = 1:SN
                SUM = SUM+SN_P(k).Location;
            end
            new.Location = SUM./SN;
            new.Cost = CostFunction(new.Location);
            
            if new.Cost < Best_Planets(i).Cost
                Best_Planets(i) = new;
            end
            
        else   % 未观测到凌日： 邻近行星
            
            Neighbor.Location = (rand().*Stars(i).Location + rand().*(unifrnd(Vmin,Vmax,1,nV)))./2;
            
            for k = 1:SN
                zone = randi(3);
                if zone ==1
                    Neighbor.Location = Neighbor.Location - (2*rand()-1).*(unifrnd(Vmin,Vmax,1,nV));
                elseif zone ==2
                    Neighbor.Location = Neighbor.Location + (2*rand()-1).*(unifrnd(Vmin,Vmax,1,nV));
                else
                    Neighbor.Location = Neighbor.Location + (2.*rand(1,nV)-1).*(unifrnd(Vmin,Vmax,1,nV));
                end
                Neighbor.Location = max(Neighbor.Location, Vmin);
                Neighbor.Location = min(Neighbor.Location, Vmax);
                Neighbor.Cost = CostFunction (Neighbor.Location);
                SN_P(k) = Neighbor;
            end
            SUM = 0;
            for k = 1:SN
                SUM = SUM+SN_P(k).Location;
            end
            Neighbor.Location = SUM./SN;
            Neighbor.Cost = CostFunction (Neighbor.Location);
            
            if Neighbor.Cost < Best_Planets(i).Cost
                Best_Planets(i) = Neighbor;
            end
        end
    end
    
    %% 最佳行星的信号放大（探索）
    for i = 1:ns
        for k = 1:SN
            RAND = randi(2 );
            if RAND ==1
                Power = randi(SN*ns);
                Coefficient = 2*rand();
                Noise = ((rand(1,nV)).^Power).*(unifrnd(Vmin,Vmax,1,nV));
            else
                Power = randi(SN*ns);
                Coefficient = 2*rand();
                Noise = -((rand(1,nV)).^Power).*(unifrnd(Vmin,Vmax,1,nV));
            end
            %                         
            chance = randi(2);
            if chance ==1
                new.Location = Best_Planets(i).Location - Coefficient.*Noise;
            else
                new.Location = (rand().*Best_Planets(i).Location) - Coefficient.*Noise;
            end
            new.Location = max(new.Location, Vmin);
            new.Location = min(new.Location, Vmax);
            new.Cost = CostFunction(new.Location);
            %                         
            
            if new.Cost < Best_Planets(i).Cost
                Best_Planets(i) = new;
            end
        end
        if Best_Planets(i).Cost < Best_Planet.Cost
            Best_Planet = Best_Planets(i);
        end
    end
    
    % 结果
    Bests(it)=Best_Planet;
end
end
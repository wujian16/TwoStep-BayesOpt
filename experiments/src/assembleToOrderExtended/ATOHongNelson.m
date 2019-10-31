function [ fnAvg ]=ATOHongNelson(BaseStockLevel,runlength,seed,~)

%% function [FnAvg, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov]=ATOHongNelson(BaseStockLevel,runlength,seed,~) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Written on 11/1/2003, by L. Jeff Hong                      %
%  Assemble to order systems with each component produce to   %
%  stock with a base stock level. Customers come in with      %
%  different taste. Each of them has a set of key components  %
%  and a set of nonkey components. If any of the key          %
%  components is unavailable, he leaves. But he accepts the   %
%  sell with unavailable nonkey items.                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% FnVar = NaN;
% FnGrad = NaN;
% FnGradCov = NaN;
% constraint = NaN;
% ConstraintCov = NaN;
% ConstraintGrad = NaN;
% ConstraintGradCov = NaN;

% Parameters of Components

NumComponentType=8;               % number of component types

ProTimeMean=[0.15,0.4,0.25,0.15,0.25,0.08,0.13,0.4];       
                                % mean production time assume normal distribution
ProTimeStd=0.15*ProTimeMean;    % standard deviation of mean production time
Profit=1:8;
HoldingCost=2*ones(1, NumComponentType);
                                % holding cost of each component in
                                % inventory
                            
% Parameters of Customers

ArrivalRate=12;             % assume Possion arrival
%NumCustomerType=5;             % number of customer types
%CustomerProb=[0.3,0.25,0.2,0.15,0.1];   % probability of each customer
KeyComponent=[1,0,0,1,0,1,0,0;
              1,0,0,0,1,1,0,0;
              0,1,0,1,0,1,0,0;
              0,0,1,1,0,1,0,0;
              0,0,1,0,1,1,0,0;];
                            % number of components required
NonKeyComponent=[0,0,0,0,0,0,1,0;
                 0,0,0,0,0,0,1,0;
                 0,0,0,0,0,0,0,0;
                 0,0,0,0,0,0,0,1;
                 0,0,0,0,0,0,1,0;];

% Set the seed for the random times

TimeStream = RandStream.create('mrg32k3a');
TimeStream.Substream = seed;
RandStream.setGlobalStream(TimeStream);


% Simulation
fnSum = 0.0

for k=1:runlength

    EventTime=1e5*ones(1,1+NumComponentType);     % arrival+components completion
    EventTime(1)=-log(rand)/ArrivalRate;
    TotalProfit=0;
    TotalCost=0;
    Inventory=BaseStockLevel;
    WarmUp=20;
    TotalTime=70;
    Clock=0;

    while Clock<TotalTime
        OldInventory=Inventory;
        OldClock=Clock;
        [Clock,event]=min(EventTime);
        if event==1
            temp=rand;
            if temp<0.3 
                CustomerType=1;
            elseif temp<0.55 
                CustomerType=2;
            elseif temp<0.75 
                CustomerType=3;
            elseif temp<0.9 
                CustomerType=4;
            else
                CustomerType=5;
            end
            KeyOrder=KeyComponent(CustomerType,:);
            NonKeyOrder=NonKeyComponent(CustomerType,:);
            Sell=1;
            for i=1:NumComponentType
                if Inventory(i)-KeyOrder(i)<0
                    Sell=0;
                end
                if Inventory(i)-NonKeyOrder(i)<0
                    NonKeyOrder(i)=Inventory(i);   % if nonkey>1, customer buys whatever they can get
                end
            end
            if Sell==1
                Inventory=Inventory-KeyOrder-NonKeyOrder;
                if Clock>WarmUp
                    TotalProfit=TotalProfit+Profit*(KeyOrder+NonKeyOrder)';
                end
            end
            
            % determine the time of next event
            
            EventTime(1)=Clock-log(rand)/ArrivalRate;
            if Sell==1
                for i=1:NumComponentType
                    if (Inventory(i)<BaseStockLevel(i))&&(EventTime(i+1)>1e4)
                        EventTime(i+1)=Clock+max(0,ProTimeMean(i)+randn*ProTimeStd(i));
                    end
                end
            end
        else
            ComponentType=event-1;
            Inventory(ComponentType)=Inventory(ComponentType)+1;
            if Inventory(ComponentType)>=BaseStockLevel(ComponentType);
                EventTime(event)=1e5;
            else
            end
        end
        if Clock>WarmUp
            TotalCost=TotalCost+(Clock-OldClock)*OldInventory*HoldingCost';
        end
    end
    fn = (TotalProfit-TotalCost)/(TotalTime-WarmUp);
    fnSum = fnSum + fn;
end
fnAvg = fnSum / runlength;

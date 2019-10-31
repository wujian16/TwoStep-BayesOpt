%   ***************************************
%   *** Code written by German Gutierrez***
%   ***         gg92@cornell.edu        ***
%   ***   Code edited by Bryan Chong    ***
%   ***        bhc34@cornell.edu        ***
%   ***************************************

%Returns Profit and Profit after T = 20.
function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = ATO(x, runlength, seed, ~)
% function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = ATO(x, runlength, seed, other);
% x is a vector containing the amounts of intermediate products to be processed ahead of time
% runlength is the number of hours of simulated time to simulate
% seed is the index of the substreams to use (integer >= 1)
% other is not used

FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

if (sum(x < 0)>0 ) || (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('x (row vector with %d components)\nx components should be between 0 and 1\nrunlength should be positive and real,\nseed should be a positive integer\n', nAmbulances*2);
    fn = NaN;
    FnVar = NaN;
    
else % main simulation
    % price, holding cost, avg prod time, std prod time, capacity
    items = [  1,      2,      .15,   .0225,   20;
        2,      2,      .40,    .06,    20;
        3,      2,      .25,    .0375,  20;
        4,      2,      .15,    .0225,  20;
        5,      2,      .25,    .0375,  20;
        6,      2,      .08,    .012,   20;
        7,      2,      .13,    .0195,  20;
        8,      2,      .40,    .06,    20];
    
    products = [   3.6,    1,  0,  0,  1,  0,  1,  1,  0;
        3,      1,  0,  0,  0,  1,  1,  1,  0;
        2.4,    0,  1,  0,  1,  0,  1,  0,  0;
        1.8,    0,  0,  1,  1,  0,  1,  0,  1;
        1.2,    0,  0,  1,  0,  1,  1,  1,  0];
    %Both matrices as defined in problem statement
    
    %Any proposed solution must satisfy bk<=ck
    bk=x'; % Starting Sol: items(:,5)
    nItems = size(items,1);
    nProducts=5; numberkey=6; numbernk=nItems-numberkey;            %# of products, key items and non key items respectively.
    Tmax=70; %70                                         %Length of simulation
    
    
    
    
    Profit=zeros(runlength,1);
    ProfitAfter20=zeros(runlength,1);
    
    nGen=10*Tmax*round(sum(products(:,1))); %upper bound on number of generated outputs
    
    % Generate new streams for call arrivals, call
    [ArrivalStream, ProductionKeyStream, ProductionNonKeyStream] = RandStream.create('mrg32k3a', 'NumStreams', 3);
    
    % Set the substream to the "seed"
    ArrivalStream.Substream = seed;
    ProductionKeyStream.Substream = seed;
    ProductionNonKeyStream.Substream = seed;
    
    % Generate random data
    OldStream = RandStream.setGlobalStream(ArrivalStream); % Temporarily store old stream
    
    Arrival=zeros(nProducts,nGen,runlength);
    %Generate time of next order arrival per product
    for k=1:nProducts
        Arrival(k,:,:)=exprnd(1/products(k,1),1,nGen,runlength);
    end
    
    
    % Generate production times
    RandStream.setGlobalStream(ProductionKeyStream);
    ProdTimeKey= normrnd(0,1,nGen,runlength);
    
    %Generate Uniforms used to generate orders
    RandStream.setGlobalStream(ProductionNonKeyStream);
    ProdTimeNonKey= normrnd(0,1,nGen,runlength);
    
    % Restore old random number stream
    RandStream.setGlobalStream(OldStream);
    
    for k = 1:runlength
        % Initialize this replication
        Inventory= bk;   %Tracks available inventory for each item
        Orders=zeros(nProducts,1); %Next order arrival times
        for i=1:nProducts
            Orders(i)=Arrival(i,1,k);
        end
        % ItemAvailability contains the time at which a replenishment order for a
        % given item(row) will be ready. Each order replenishes one unit of
        % one item.
        itemAvailability=zeros(nItems,1);
        A=ones(1,nProducts)+1; % index of next arrival time to use
        prevMinTime=0;      %Holds the time of previous order fulfilled (to calculate holding cost)
        p=1; % Index for Key item production times
        q=1; % Index for non-Key production times
        
        % Main simulation:
        %**** Loop through orders, as they happened (smallest time first) and identify ****
        %**** whether key and non-key items are available, calculate profit, etc.      ****
        
        %While there are orders to be satisfied.
        while(min(Orders)<=Tmax)
            % find next order to satisfy, i.e. smallest time of orders not yet
            % satisfied.
            minTime=Tmax;                            %keeps track of minimum order time found so far.
            for j=1:nProducts
                if(Orders(j)<=minTime)
                    minTime=Orders(j);                            % Time of next order to fulfill
                    minProd=j;                                    % Product of next order to fulfill
                end
            end
            Orders(minProd)=Orders(minProd)+Arrival(minProd,A(minProd),k); %generate time of next order
            A(minProd)=A(minProd)+1;
            
            if minTime >= Tmax
                break
            end
            
            %update inventory levels up to time of next order (order to be
            %fulfilled in this iteration) i.e. add however many items are available
            %by minTime.
            
            %Delete all zero columns
            itemAvailability=sort(itemAvailability,2);
            maxEntries=0;
            for i=1:nItems
                if nnz(itemAvailability(i,:)) >= maxEntries
                    maxEntries=nnz(itemAvailability(i,:));
                end
            end
            sizeIA=size(itemAvailability,2);
            if maxEntries>0
                itemAvailability(:,1:sizeIA-maxEntries)=[];
            end
            
            %Add inventory that is now ready to be used.
            sizeIA=size(itemAvailability,2);
            for i=1:nItems
                for j=1:sizeIA
                    if(itemAvailability(i,j)~=0 && itemAvailability(i,j)<=minTime)
                        Inventory(i)=Inventory(i)+1;
                        itemAvailability(i,j)=0;
                    end
                end
            end
            
            %if(all key items available) make product, calculate profit, update
            %inventory and trigger replenishment orders
            
            %loop to check if all key items are available:
            keyavail=0;
            for j=1:numberkey
                if products(minProd,j+1)<=Inventory(j)
                    keyavail=keyavail+1;
                end
            end
            if(keyavail==numberkey)             %all key items available
                Profit(k)=Profit(k)+products(minProd,2:numberkey+1)*items(1:numberkey,1);               %Add profit made from making this product
                if(minTime>=20)
                    %To keep track of profit made after time 20
                    ProfitAfter20(k)=ProfitAfter20(k)+products(minProd,2:numberkey+1)*items(1:numberkey,1);
                end
                for r=1:numberkey
                    % Decrease inventory and place replenishment orders for the amount of key items used
                    if products(minProd,r+1)~=0
                        num = products(minProd,r+1);
                        Inventory(r)=Inventory(r)-num;
                        tempSize = sizeIA;
                        for g = 1:num
                            itemAvailability(r,tempSize+1)=max(minTime,itemAvailability(r,tempSize))+(items(r,4)*ProdTimeKey(p,k)+items(r,3));
                            tempSize = tempSize + 1;
                            p=p+1;
                        end
                    end
                end
                % For each non-key item available, use it, decrease inventory, increase profit and place replenishment order.
                for j=1:numbernk
                    num=products(minProd,j+numberkey+1);
                    if(num<=Inventory(j+numberkey) && num~=0)
                        %Update profit, inventory and place replenishment orders
                        Profit(k)=Profit(k)+items(j+numberkey,1);
                        if(minTime>=20)
                            ProfitAfter20(k)=ProfitAfter20(k)+items(j+numberkey,1)*num;
                        end
                        Inventory(j+numberkey)=Inventory(j+numberkey)-num;
                        tempSize = sizeIA;
                        for g=1:num
                            itemAvailability(j+numberkey,tempSize+1)=max(minTime,itemAvailability(j+numberkey,tempSize))+(items(j+numberkey,4)*ProdTimeNonKey(q,k)+items(j+numberkey,3));
                            tempSize = tempSize + 1;
                            q=q+1;
                        end
                    end
                end
            end
            Profit(k)=Profit(k)-sum(Inventory'*items(:,2))*(minTime-prevMinTime);
            if(minTime>=20)
                ProfitAfter20(k)=ProfitAfter20(k)-sum(Inventory'*items(:,2))*(minTime-prevMinTime);
            end
            prevMinTime=minTime;
        end
    end
    dailyProfitAfter20 = ProfitAfter20/(Tmax-20);
    fn=mean(dailyProfitAfter20);
    FnVar = var(dailyProfitAfter20)/runlength;
end




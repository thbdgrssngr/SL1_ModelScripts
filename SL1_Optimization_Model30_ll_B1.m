%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function find the best fitting model/parameters
% Thibaud Griessinger, Paris, March, 2015
% Adapted from : "Thibaud Griessinger, Los Angeles, November, 2014"

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Version :  model test 3.0 - TEST PARALLEL RUN on CLUSTER
%% Contains 13 Models (call "SL1_Computational_Model30.m" a script summarizing the 13 Models tested before (with p(0)= 1/2): Model#1: Q-RL (2FP) M#2: Q-RL CounterF. (3FP) M#3: EWA (3FP) M#4: Q-RL 2states#2 (2FP) M#5: Q-RL 2states#5 (2FP) M#6: Q-RL 2states#6 (2FP) M#7: Q-RL CounterF.+2states#2 (3FP) M#8: Q-RL CounterF.+2states#5 (3FP) M#9: Q-RL CounterF.+2states#6 (3FP) M#10: Fictitious (2FP) M#11: IM (3FP) M#12: 1-Inf (3FP) M#13: 2-Inf (4FP) )

clear all
close all
load('rIG_data.mat');

modelversion = 30;
FileExt = '_AllM30_GS10_ll_B1'; % All 13 Models previously tested, Size Grid Search Fmincon: 10 Iterations
NITER = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Organise input data b 

%Clust repartition in screens
MtoRun= input('Model to Run (1-13):');
if MtoRun== 1 % Q-RL
    whichmodel= 1;
elseif MtoRun== 2 % Q-RL counter
    whichmodel= 2; 
elseif MtoRun== 3 % EWA
    whichmodel= 3; 
elseif MtoRun== 4 % Q-RL 2states #2 
    whichmodel= 4; 
elseif MtoRun== 5 % Q-RL 2states #5
    whichmodel= 5; 
elseif MtoRun== 6 % Q-RL 2states #6
    whichmodel= 6; 
elseif MtoRun== 7 % Q-RL counter + 2states #2
    whichmodel= 7; 
elseif MtoRun== 8 % Q-RL counter + 2states #5
    whichmodel= 8; 
elseif MtoRun== 9 % Q-RL counter + 2states #6
    whichmodel= 9; 
elseif MtoRun== 10 % Fictitous
    whichmodel= 10; 
elseif MtoRun== 11 % IM
    whichmodel= 11; 
elseif MtoRun== 12 % 1-Inf
    whichmodel= 12; 
elseif MtoRun== 13 % 2-Inf
    whichmodel= 13; 
end

subjects=[1:64]; % subjects to include in the analysis
block = 1; % data from communication block 1 (100 trials) (block 2 -> Script "SL1_Optimization_Model30_ll_B2")  

%% Data compilation: 
% choice and outcome data of each subject : 
for sub = subjects;
    for t = 1:100 % for each trial
        n = sub-1;
        % condition / state of each trial -> here only 1 state
        con = zeros(100, 1)+1;
        % action chosen  
        if block == 1
            N = 0;
        elseif block == 2
            N = 6400; % data in rIG_data.mat -> Block1 (64subx100t), then block 2 (64subx100t)
        end 
        ch = rIG_Data.subChoiceA(t+(N+(n*100))); % Choice made by the subject: either = 1 (P1: working, P2: not inspecting) or = -1 (P1: not working, P2: inspecting)
        % Choices converted in 1 and 2, instead of 1 and -1 (for Qvalues columns in Computation_Model)
        if ch == 1
             cho(t,1) = 1; 
        elseif ch == -1
             cho(t,1) = 2; 
        else % ch == 2
             cho(t,1) = 0; % If subject did not choose within the time limit 
        end  
        ro(t,1) = rIG_Data.subRole(t+(N+(n*100))); % Role endorsed by the subject, (Player) 1 aka employee, (Player 2) aka employer
        out(t,1) = (rIG_Data.subPayoff(t+(N+(n*100)))/100)*2; % Outcome experienced by the subject: originally 0, 50 or 100 pts, here converted to 0, 1, 2 units (as in Hampton)
    end
    % concatenate all data for each trial in cell arrays for each subject 
    rol_ = double(ro);
    rol_s{sub}= rol_;
    cho_ = double(cho);
    cho_s{sub}= cho_; 
    out_ = double(out);
    out_s{sub}= out_;  
    con_ = double(con);
    con_s{sub}= con_;
end
% choice and outcome data of the opponent of each subject : 
for opp = subjects;
    for t = 1:100
        n = opp-1;
        con = zeros(100, 1)+1;
        if block == 1
            N = 0;
        elseif block == 2
            N = 6400;
        end   
        ch = rIG_Data.oppChoiceA(t+(N+(n*100)));
        % Choices converted in 1 and 2, instead of 1 and -1
        if ch == 1
             cho(t,1) = 1;
        elseif ch == -1
             cho(t,1) = 2;
        else % % ch == 2 i.e. Opponent missed -> more complex here since the subject did not know when the opponent missed (did not choose in time), but experienced the best outcome associtated to her choice as if the opponent played the less rewarding action given the subject's choice 
            if (rIG_Data.subChoiceA(t+(N+(n*100)))== 1 && rIG_Data.subRole(t+(N+(n*100)))== 1) || (rIG_Data.subChoiceA(t+(N+(n*100)))== -1 && rIG_Data.subRole(t+(N+(n*100)))== 2) % this therefore depends on the subject's role and her choice (see payoff matrix)
                cho(t,1) = 2; % opponent's action "experienced" through the best outcome the subject received when the other missed -> this opponent's action lead to the best outcome for the subject depending on the action she selected and the role she endorsed in the game (specified in the if condition)
            elseif (rIG_Data.subChoiceA(t+(N+(n*100)))== -1 && rIG_Data.subRole(t+(N+(n*100)))== 1) || (rIG_Data.subChoiceA(t+(N+(n*100)))== 1 && rIG_Data.subRole(t+(N+(n*100)))== 2)
                cho(t,1) = 1; % opponent's action "experienced" for this subject's role / choice combination
            elseif rIG_Data.subChoiceA(t+(N+(n*100)))== 2 % in case they both missed, the subject (as the opponent) did not experience the opponent's action (no outcome displayed, but 0 pts outcome experienced as instructed, on the matrix display at time of the ouctome was just written "both missed" in red) 
                cho(t,1) = 0;
            end
        end
        ro(t,1) = rIG_Data.oppRole(t+(N+(n*100)));
        out(t,1) = (rIG_Data.oppPayoff(t+(N+(n*100)))/100)*2; % here converted to 0, 1, 2 units (as in Hampton)
    end
    % concatenate all data for each trial in cell arrays for each subject 
    rol_ = double(ro);
    rol_o{opp}= rol_; 
    cho_ = double(cho);
    cho_o{opp}= cho_; 
    out_ = double(out);
    out_o{opp}= out_; 
    con_ = double(con);
    con_o{opp}= con_;
end

%% Grid Search - Gradient Descent (fmincon function - Matlab 2012)
options = optimset('Algorithm','interior-point', 'MaxIter', 10000); % Option as in Mehdi & Stefano's optimization scripts 
%% fmincon parameter estimation from different starting points
SFb=[1:(NITER*2)];
SFa=[1:NITER];
sB0= linspace(0, 5, (numel(SFb))); % generate a value from the vector [1, 30] one step by one by step size of max value divided by SF or the number of time fmincon is played on the interval [1, 30]
sA10= linspace(0, 1, (numel(SFa)));  % from 0 to 1
sA20= linspace(0, 1, (numel(SFa)));  % from 0 to 1
sA30= linspace(0, 1, (numel(SFa)));  % from 0 to 1
% sE0= linspace(-0.5, 0.5, (numel(SF)));   % from -0.5 to 0.5
for SF0 = SFb; % Each beta
    sB= sB0(1, SF0);
    for SF1 = SFa; % Each alpha1
        sA1= sA10(1, SF1);
        for SF2 = SFa; % Each alpha2
            sA2= sA20(1, SF2);
            for SF3 = SFa; % Each alpha2
                sA3= sA30(1, SF3);
                for sub = subjects;
                    modelN= 0;
                    for model = whichmodel;
                        modelN= modelN+1; % Ref for merge 3 models parameters2 matrices in one via: parameters2= cat(3, parameters2_1, parameters2_2, parameters2_3) in AnalysisOptOutput script;
                        if model == 13 % 4 FreeParams
                            [parameters(sub,:,modelN, SF0, SF1, SF2, SF3),ll(sub,modelN, SF0, SF1, SF2, SF3),report(sub,modelN, SF0, SF1, SF2, SF3), ~, ~, grad(sub,:,modelN, SF0, SF1, SF2, SF3), hess2(:,:,sub, modelN, SF0, SF1, SF2, SF3)]=fmincon(@(x) SL1_Computational_Model30(x,rol_s{sub}, con_s{sub},cho_s{sub},out_s{sub}, con_o{sub},cho_o{sub},out_o{sub}, model),[sB sA1 sA2 sA3],[],[],[],[],[0 0 0 0],[Inf 1 1 1],[], options);
                        elseif model == 2 || model == 3 || model == 7 || model == 8 || model == 9 || model == 11 || model == 12 % 3 FP
                            [parameters(sub,:,modelN, SF0, SF1, SF2, SF3),ll(sub,modelN, SF0, SF1, SF2, SF3),report(sub,modelN, SF0, SF1, SF2, SF3), ~, ~, grad(sub,:,modelN, SF0, SF1, SF2, SF3), hess2(:,:,sub, modelN, SF0, SF1, SF2, SF3)]=fmincon(@(x) SL1_Computational_Model30(x,rol_s{sub}, con_s{sub},cho_s{sub},out_s{sub}, con_o{sub},cho_o{sub},out_o{sub}, model),[sB sA1 sA2 sA3],[],[],[],[],[0 0 0 0],[Inf 1 1 0],[], options);                        
                        else % model == 1 || model == 4 || model == 5 || model == 6 || model == 10 % 2 FP 
                            [parameters(sub,:,modelN, SF0, SF1, SF2, SF3),ll(sub,modelN, SF0, SF1, SF2, SF3),report(sub,modelN, SF0, SF1, SF2, SF3), ~, ~, grad(sub,:,modelN, SF0, SF1, SF2, SF3), hess2(:,:,sub, modelN, SF0, SF1, SF2, SF3)]=fmincon(@(x) SL1_Computational_Model30(x,rol_s{sub}, con_s{sub},cho_s{sub},out_s{sub}, con_o{sub},cho_o{sub},out_o{sub}, model),[sB sA1 sA2 sA3],[],[],[],[],[0 0 0 0],[Inf 1 0 0],[], options);
                        end
                    end
                end
            end
        end
    end
end

%% Save Data
save(strcat('SL1_Model', num2str(modelversion), FileExt, '_', num2str(MtoRun)))

%% End
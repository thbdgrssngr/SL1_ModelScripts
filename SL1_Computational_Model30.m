%% Model 3.0: Script summarizing the 13 Models tested before (with p(0)= 1/2): Model#1: Q-RL (2FP) M#2: Q-RL CounterF. (3FP) M#3: EWA (3FP) M#4: Q-RL 2states#2 (2FP) M#5: Q-RL 2states#5 (2FP) M#6: Q-RL 2states#6 (2FP) M#7: Q-RL CounterF.+2states#2 (3FP) M#8: Q-RL CounterF.+2states#5 (3FP) M#9: Q-RL CounterF.+2states#6 (3FP) M#10: Fictitious (2FP) M#11: IM (3FP) M#12: 1-Inf (3FP) M#13: 2-Inf (4FP)
function lik = SL1_Computational_Model30(params, pl, ss, as, rs, so, ao, ro, model) % input vectors: pl= player role of subject, (s= state, a= action, r= reward) of subject (s) and opponent (o), at each trial
% This function estimates the inverse loglikelihood of each model for each parameter value that fmincon tests in the optimization process
% Thibaud Griessinger, Paris, November, 2014
% Adapted from : "Thibaud Griessinger, Los Angeles, February, 2014 - adapted from "Stefano Palminteri, Paris, October, 2013""

% 0, 50, 100 pts converted in 0, 1, 2 units in outcome data and models 

%% Parameters in each model 
Beta = params(1);               % choice (inverse) temperature (1/T). Note : High temperatures (beta -> 0) cause all actions to be nearly equiprobable, whereas low temperatures (beta -> Inf) cause greedy action selections                                   
lr1  = params(2);                 
lr2  = params(3);
lr3  = params(4);

%% Initial Values 
pstar  = 1/2; % random level opponent's action 1 and 2

% Initializing Q table for each pair (state, action) 
Q      = zeros(2,2); % Expected values of actions 1 and 2 in state 1. 
% Qvalue initial = initial expected value depending on the inital belief over the opponent's action proba. Therefore depends on the role of the subject 
if pl == 1 % employee (Q(1,1) -> Q-value action "work", Q(1,2) -> Q-value action "not work")
    Q(:,1)  = 1*(1-pstar); 
    Q(:,2)  = 2*(pstar); 
else % pl == 2 % employer (Q(1,1) -> Q-value action "inspect", Q(1,2) -> Q-value action "not inspect")
    Q(:,1)  = 2*(pstar); 
    Q(:,2)  = 1*(1-pstar); 
end

%% Initial cumulating LogLikelihood
lik    = 0; 

% "Normalization" parameter
epsilon = 10^-6;

for t = 1:length(as);

% [missed trials] All models does not consider trials missed by the subject (as(i)== 0) (and thôse both missed (ao(i)== 0))

    %% Q-Learning classic 
    if model == 1 
        if as(t)~= 0 
            % Cumulated Log likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta)))); % Note that this is equivalent to Stefano-fromDaw10's original script where lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            % Factual Update: 
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(1,as(t)); % (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(1,as(t)) = Q(1,as(t)) + (lr1*RPE);
        end


    %% Counterfactual Q-RL (as in Stefano2015_inprep)
    elseif model == 2
        if as(t)~= 0 
            % Cumulated Log likelihood of the Model: 
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            
            % Factual Update: 
            RPE =  rs(t) - Q(1,as(t)); % (Reward Obtained for action selected - Q-Value stored of this action)
            Q(1,as(t)) = Q(1,as(t)) + (lr1*RPE);
            
            % Counterfactual Update: 
            % 0) Model needs to generate counterfactual rewards. What would have been obtained if chose differently 
            if pl(t) == 1 % Role 1: model as agent is the employee (Hampton's nomenclature : p = subject-as-employer's proba action, and p** = subject-as-employer's action proba simulated by the opponent-as-employee)
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 1;
                end
            elseif pl(t) == 2 % Role 2
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 1;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 0;    
                end
            end
            % 1) Model generates a Counterfactual Reward Prediction Error from the reward that could have been experienced by the subject if chose the other option
            RPE_u =  rs_u(t) - Q(1,(mod(as(t),2)+1)); % (Counterfactual Reward Obtained for action not-selected - Q-Value stored of the other action too)
            % 2) Model updates estimated Q-value of the action the subject did not selected: 
            Q(1,(mod(as(t),2)+1)) = Q(1,(mod(as(t),2)+1)) + (lr2*RPE_u);
        end


    %% EWA (Zhu_2012 Adaptation + Simplification suggested in Camerer2007, see below and "[SL1] Results Analysis 1 - comitémithèse")
    elseif model == 3
        phi = lr1; % belief about the speed of adaptation of the opponent
        delta = lr2; % weight between foregone payoffs and actual payoffs when updating value
        if as(t)~= 0
            % Cumulated Log likelihood of the Model: 
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta)))); 
            
            % Estimate Counterfactual Reward given subject and the opponent's choices: 
            % 0) generates counterfactual reward. What would have been obtained if chose differently 
            if pl(t) == 1 % Role 1
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 1;
                end
            elseif pl(t) == 2 % Role 2
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 1;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 0;    
                end
            end

            % Action Values update given weight RL/"BB" - 1 value updated each trial: Q (originally also N but:)
            N = 1; % N(t) = (rho * NN(t)) +1; The initial experience N (0) was included in the original EWA model so that Bayesian learning models are nested as a special case—N (0) represents the strength of prior beliefs. We restrict N (0) = 1 here because its influence fades rapidly as an experimentprogresses and mostsubjects come to experiments with weak priors anyway. T
            Q(1,as(t)) = (phi * N * Q(1,as(t)) + rs(t)) /N;
            Q(1,(mod(as(t),2)+1)) = (phi * N * Q(1,(mod(as(t),2)+1)) + (delta*rs_u(t))) /N;

        end


    %% Q-RL 2-States#2 (state is the same as at previous trial - assume opponent repeat his choice) 
    elseif model == 4
        if  as(t)~= 0 
            
            % State selection: What state are we in at the time of choice (i.e. which option the opponen would choose) ? 
            if t > 1 && (ao(t-1) ~= 0) % if as(t)~= 0 then ao(t) must be ~= 0 since when t= 0 ao(t-1) not possible ! 
                s = ao(t-1); % Same state than previous one (option the opponent selected in the previous trial)
            else 
                s = RandomChoice([0.5 0.5]); % first trial of the block the state is selected randomly - [alternative] can be fixed in trial 1 of block 2 as a prior over the current state 
            end
            
            % Cumulated Log likelihood of the Model to chose the same option than the subject: 
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); 
            
            % Factual Update: 
            RPE =  rs(t) - Q(ao(t),as(t)); % Note that the state here is the other's action observed at the time of the outcome 
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);
        end


    %% Q-RL 2-States #5 (state containing the highest Q-value)
    elseif model == 5
        if  as(t)~= 0 
            
            % State selection:
            if max(Q(1,:)) ~= max(Q(2,:)) % if only one Q-value max 
                mx = mod(find((Q(:,:)) == max(max(Q(:,:)))),2); % State in which the max Q-Value in all Q matrix 
                if mx == 0 
                    s = 2;
                else 
                    s= 1;
                end
            else 
                s= RandomChoice([0.5 0.5]); % State picked randomly if 2 max q-values equal in the 2 states 
            end
            
            % Cumulated Log likelihood: 
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            
            % Factual Update: 
            RPE =  rs(t) - Q(ao(t),as(t)); 
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE); % State = the other's action observed at the time of the outcome 
        end


    %% Q-RL 2-States #6 ("clever" - state the most freq, option chosen the most in the past opponent's choices history)
    elseif model == 6
        if  as(t)~= 0 
            % State selection:
            freq_o = sum(ao(1:t))/t; % Counts frequency of opponent's choices from trial 1 - Takes a value between 1 and 2, then mean = 1.5, thus: 
            if freq_o < 1.5
                s = 1;
            elseif freq_o > 1.5
                s = 2;
            else
                s= RandomChoice([0.5 0.5]); % if each option chosen equally frequently in the past pick randomly - [alternative] can take the most frequent since trial= 2. 
            end
            
            % Cumulated Log likelihood: 
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); 
            
            % Factual Update: 
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);
        end


    %% Q-RL CounterF. + 2-States #2 (state is the same as at previous trial - assume opponent repeat his choice + Counter Factual update of Q-value unchosen) 
    elseif model == 7
        if  as(t)~= 0 
            
            % State selection:
            if t > 1 && (ao(t-1) ~= 0) 
                s = ao(t-1); 
            else 
                s= RandomChoice([0.5 0.5]);
            end
            
            % Cumulated Log likelihood: 
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); 

            % Factual Update: 
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);

            % Counterfactual Update
            % 0) Counterfactual reward: 
            if pl(t) == 1 
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 1;
                end
            elseif pl(t) == 2 
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 1;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 0;    
                end
            end
            % 1) Counterfactual Reward Prediction Error:
            RPE_u =  rs_u(t) - Q(ao(t),(mod(as(t),2)+1));
            % 2) Model updates unchosen Q-value:
            Q(ao(t),(mod(as(t),2)+1)) = Q(ao(t),(mod(as(t),2)+1)) + (lr2*RPE_u);

        end


    %% Q-RL CounterF. + 2-States #5 (state with highest Q-value + Counter Factual update of Q-value unchosen)
    elseif model == 8
        if  as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % State selection:
            if max(Q(1,:)) ~= max(Q(2,:)) 
                mx = mod(find((Q(:,:)) == max(max(Q(:,:)))),2);
                if mx == 0 
                    s = 2;
                else 
                    s= 1;
                end
            else 
                s= RandomChoice([0.5 0.5]);
            end
            
            % Cumulated Log likelihood: 
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); 

            % Factual Update: 
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);

            % Counterfactual Update
            % 0) Counterfactual reward: 
            if pl(t) == 1 
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 1;
                end
            elseif pl(t) == 2 
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 1;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 0;    
                end
            end
            % 1) Counterfactual Reward Prediction Error:
            RPE_u =  rs_u(t) - Q(ao(t),(mod(as(t),2)+1));
            % 2) Model updates unchosen Q-value:
            Q(ao(t),(mod(as(t),2)+1)) = Q(ao(t),(mod(as(t),2)+1)) + (lr2*RPE_u);


        end


    %% Q-RL Counter + 2-States #6 ("clever" - state the most freq, option chosen the most in the past opponent's choices history +  Counter Factual update of Q-value unchosen)
    elseif model == 9
        if  as(t)~= 0
            
            % State selection:
            freq_o = sum(ao(1:t))/t; 
            if freq_o < 1.5
                s = 1;
            elseif freq_o > 1.5
                s = 2;
            else
                s= RandomChoice([0.5 0.5]);
            end
            
            % Cumulated Log likelihood: 
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); 

            % Factual Update: 
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);

            % Counterfactual Update
            % 0) Counterfactual reward: 
            if pl(t) == 1 
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 1;
                end
            elseif pl(t) == 2 
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 1;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 0;    
                end
            end
            % 1) Counterfactual Reward Prediction Error:
            RPE_u =  rs_u(t) - Q(ao(t),(mod(as(t),2)+1));
            % 2) Model updates unchosen Q-value:
            Q(ao(t),(mod(as(t),2)+1)) = Q(ao(t),(mod(as(t),2)+1)) + (lr2*RPE_u);

        end


    %% Fictitious Play 
    elseif model == 10
        % Agent observes the opponent's action and generates an Action Prediction Error over her (observed) behavior
        if  as(t)~= 0 % Even if opponent's action experienced, subject do not update own's action proba since not choose at time and therefore default action selected
            % Cumulated Log likelihood:  
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta))));
            
            % 1) Model observes the opponent's action and generates an Action Prediction Error over her (observed) behavior only if subject did not miss
            APE =  mod(ao(t),2) - pstar; % [if as(i)~=0 then ao(i)~=0] APE : (Action) Prediction Error experienced by the subject between the opponent?s expected action p* and whether the opponent chose action 1 at time t (P= 1, mod(oppChoice=1,2)), or chose another action (P = 0, mod(oppChoice=2,2)).
            % 2) Estimate proba actions of the other:
            pstar = pstar + (lr1 * APE); % updates opponent's action 1 probability at trial i
            
            % NORMALIZATION
            if pstar < epsilon %if pstar too small, or 0, then superior to 0 
                pstar = epsilon; % or pstar= epsilon ?
            elseif pstar > (1-epsilon)
                pstar = (1-epsilon);
            end
            
            % 3) (factual and counterfactual) Q-values Update 
            % No matter if the subject selected the action at time or not, the subject updates, from the estimated opponent's action proba, the Q-values of both action 1 and 2 independantly of the action selected (if so). This update depends on the expected value of each action, thus the payoffs associated and therefore on its role (since assymetric payoff matrix)
            if pl(t) == 1 
                Q(1,1) = 1*(1-pstar); % expected value of action 1 "work" given opponent's action proba -> 0*(pstar)+0.5*(1-pstar) 
                Q(1,2) = 2*(pstar); % expected value of action 2 "not work" -> 1*(pstar)+0*(1-pstar)
            elseif pl(t) == 2 
                Q(1,1) = 2*(pstar); % expected value action 1 "not inspect" -> 1*(pstar)+0*(1-pstar) 
                Q(1,2) = 1*(1-pstar); % expected value of action 2 "inspect" exp value -> 0*(pstar)+0.5*(1-pstar)
            end
        end


    %% Influence # 1 - "original" (Hampton's SI)
    elseif model == 11
        % Agent observes the opponent's action and generates an Action Prediction Error over her (observed) behavior
        if  as(t)~= 0 % Even if opponent's action experienced, subject do not update own's action proba since not choose at time and therefore default action selected
            
            % Cumulated Log likelihood:  
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta))));
            
            APE =  mod(ao(t),2) - pstar; % [if as(t)~=0 then ao(t)~=0]
            % 2) Estimate proba actions of the other: 
            % a) once the subject observed the opponent's action, she will have to simulate her own action 1 proba as presumably estimated by the opponent through a fictitious (Qstar)
            % b) and then from the opponent's action observed and the opponent's estimated influence of her action she will update the opponent's action 1 proba (pstar) 
            % c) from the other's estimated action proba she will update the expected value associated to the action 1 depending on her role (since assymetric payoff matrix)
            weight = 3*Beta;
            if pl(t) == 1 % subject is the employee (Hampton's nomenclature : p = subject-as-employer's proba action, and p** = subject-as-employer's action proba simulated by the opponent-as-employee)
                % a) estimation of the own's action proba as estimated by the opponent through a fictitious play (inferred probabilities that the opponent has of the subject) : such second order beliefs are inferred by the agent directly from the inferred opponent?s strategy by inverting the inferred probabilities of the opponent?s actions estimated with it's APE update
                Qstar= (1/3)-((1/weight)*log((1-pstar)/pstar)); % [p**(t)] in Hampton's equations
                % NORMALIZATION Qstar
                if Qstar < epsilon %if pstar too small, or 0, then superior to 0 
                    Qstar = epsilon; % or pstar= epsilon ?
                elseif Qstar > (1-epsilon)
                    Qstar = (1-epsilon);
                end
                % b) Opponent had seen an action from the subject (the one that maximize her gain depending on her own move) therefore :
                LT = (lr1* APE); % Learning Term
                pstar= pstar + LT + (lr2*weight*pstar*(1-pstar)*(mod(as(t),2)-Qstar)); % [q*(t)] : update given what the subject's action the opponent has experienced
                % NORMALIZATION pstar
                if pstar < epsilon %if pstar too small, or 0, then superior to 0 
                    pstar = epsilon; % or pstar= epsilon ?
                elseif pstar > (1-epsilon)
                    pstar = (1-epsilon);
                end
                % if subject missed or not the 2 Q-values associated to the 2 actions are updated
                % c) Q-value update from estimated pstar
                Q(1,1) = 1*(1-pstar); % expected value of action 1 "work" given opponent's action proba -> 0*(pstar)+0.5*(1-pstar) 
                Q(1,2) = 2*(pstar); % expected value of action 2 "not work" -> 1*(pstar)+0*(1-pstar)
            elseif pl(t) == 2 % subject is the employer (Hampton's nomenclature : q = subject-as-employer's proba action, and q** = subject-as-employer's action proba simulated by the opponent-as-employer)
                % a)
                Qstar= (1/3)+((1/weight)*log((1-pstar)/pstar)); % [q**(t)] in Hampton's equations
                % NORMALIZATION Qstar
                if Qstar < epsilon %if pstar too small, or 0, then superior to 0 
                    Qstar = epsilon; % or pstar= epsilon ?
                elseif Qstar > (1-epsilon)
                    Qstar = (1-epsilon);
                end
                % b)
                LT = (lr1* APE); % Learning Term
                pstar= pstar + LT - (lr2*weight*pstar*(1-pstar)*(mod(as(t),2)-Qstar)); % [p*(t)]
                % NORMALIZATION pstar
                if pstar < epsilon %if pstar too small, or 0, then superior to 0 
                    pstar = epsilon; % or pstar= epsilon ?
                elseif pstar > (1-epsilon)
                    pstar = (1-epsilon);
                end
                % c)
                Q(1,1) = 2*pstar; % expected value action 1 "not inspect" -> 1*(pstar)+0*(1-pstar) 
                Q(1,2) = 1*(1-pstar); % expected value of action 2 "inspect" exp value -> 0*(pstar)+0.5*(1-pstar)
            end  
        end   

    
    %% Influence # 2 - "1-Inf" (Devaine et al)
    elseif model == 12
        kappa = lr2; % kappa = Beta*lr2 
        % 1) Agent observes the opponent's action and generates an Action Prediction Error over her (observed) behavior
        if  as(t)~= 0 % Even if opponent's action experienced, subject do not update own's action proba since not choose at time and therefore default action selected
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta))));
            APE =  mod(ao(t),2) - pstar; % [if as(t)~=0 then ao(t)~=0]
            % 2) Estimate proba actions of the other: 
            % a) once the subject observed the opponent's action, she will have to simulate her own action 1 proba as presumably estimated by the opponent through a fictitious (Qstar)
            % b) and then from the opponent's action observed and the opponent's estimated influence of her action she will update the opponent's action 1 proba (pstar) 
            % c) from the other's estimated action proba she will update the expected value associated to the action 1 depending on her role (since assymetric payoff matrix)
            weight = 3*Beta;
            if pl(t) == 1 % subject is the employee (Hampton's nomenclature : p = subject-as-employer's proba action, and p** = subject-as-employer's action proba simulated by the opponent-as-employee)
                % a) estimation of the own's action proba as estimated by the opponent through a fictitious play (inferred probabilities that the opponent has of the subject) : such second order beliefs are inferred by the agent directly from the inferred opponent?s strategy by inverting the inferred probabilities of the opponent?s actions estimated with it's APE update
                Qstar= (1/3)-((1/weight)*log((1-pstar)/pstar)); % [p**(t)] in Hampton's equations
                % NORMALIZATION
                if Qstar < epsilon %if pstar too small, or 0, then superior to 0 
                    Qstar = epsilon; % or pstar= epsilon ?
                elseif Qstar > (1-epsilon)
                    Qstar = (1-epsilon);
                end
                % b) Opponent had seen an action from the subject (the one that maximize her gain depending on her own move) therefore :
                LT = (lr1* APE); % Learning Term
                pstar= pstar + LT + (3*kappa*pstar*(1-pstar)*(mod(as(t),2)-Qstar)); % [q*(t)] : update given what the subject's action the opponent has experienced
                % NORMALIZATION
                if pstar < epsilon %if pstar too small, or 0, then superior to 0 
                    pstar = epsilon; % or pstar= epsilon ?
                elseif pstar > (1-epsilon)
                    pstar = (1-epsilon);
                end
                % if subject missed or not the 2 Q-values associated to the 2 actions are updated
                % c) Q-value update from estimated pstar
                Q(1,1) = 1*(1-pstar); % expected value of action 1 "work" given opponent's action proba -> 0*(pstar)+0.5*(1-pstar) 
                Q(1,2) = 2*(pstar); % expected value of action 2 "not work" -> 1*(pstar)+0*(1-pstar)
            elseif pl(t) == 2 % subject is the employer (Hampton's nomenclature : q = subject-as-employer's proba action, and q** = subject-as-employer's action proba simulated by the opponent-as-employer)
                % a)
                Qstar= (1/3)+((1/weight)*log((1-pstar)/pstar)); % [q**(t)] in Hampton's equations
                % NORMALIZATION
                if Qstar < epsilon %if pstar too small, or 0, then superior to 0 
                    Qstar = epsilon; % or pstar= epsilon ?
                elseif Qstar > (1-epsilon)
                    Qstar = (1-epsilon);
                end
                % b)
                LT = (lr1* APE); % Learning Term
                pstar= pstar + LT - (3*kappa*pstar*(1-pstar)*(mod(as(t),2)-Qstar)); % [p*(t)]
                % NORMALIZATION
                if pstar < epsilon %if pstar too small, or 0, then superior to 0 
                    pstar = epsilon; % or pstar= epsilon ?
                elseif pstar > (1-epsilon)
                    pstar = (1-epsilon);
                end
                % c)
                Q(1,1) = 2*pstar; % expected value action 1 "not inspect" -> 1*(pstar)+0*(1-pstar) 
                Q(1,2) = 1*(1-pstar); % expected value of action 2 "inspect" exp value -> 0*(pstar)+0.5*(1-pstar)
            end
        end   


    %% Influence # 3 - "2-Inf" (Devaine et al)
    elseif model == 13
        kappa = lr2;
        omega = lr3;
        % 1) Agent observes the opponent's action and generates an Action Prediction Error over her (observed) behavior
        if  as(t)~= 0 % Even if opponent's action experienced, subject do not update own's action proba since not choose at time and therefore default action selected
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta))));
            APE =  mod(ao(t),2) - pstar; % [if as(t)~=0 then ao(t)~=0]
            % 2) Estimate proba actions of the other: 
            % a) once the subject observed the opponent's action, she will have to simulate her own action 1 proba as presumably estimated by the opponent through a fictitious (Qstar)
            % b) and then from the opponent's action observed and the opponent's estimated influence of her action she will update the opponent's action 1 proba (pstar) 
            % c) from the other's estimated action proba she will update the expected value associated to the action 1 depending on her role (since assymetric payoff matrix)
            weight = 3*Beta;
            if pl(t) == 1 % subject is the employee (Hampton's nomenclature : p = subject-as-employer's proba action, and p** = subject-as-employer's action proba simulated by the opponent-as-employee)
                % a) estimation of the own's action proba as estimated by the opponent through a fictitious play (inferred probabilities that the opponent has of the subject) : such second order beliefs are inferred by the agent directly from the inferred opponent?s strategy by inverting the inferred probabilities of the opponent?s actions estimated with it's APE update
                Qstar= (1/3)-((1/weight)*log((1-pstar)/pstar)); % [p**(t)] in Hampton's equations
                % NORMALIZATION
                if Qstar < epsilon %if pstar too small, or 0, then superior to 0 
                    Qstar = epsilon; % or pstar= epsilon ?
                elseif Qstar > (1-epsilon)
                    Qstar = (1-epsilon);
                end
                % b) Opponent had seen an action from the subject (the one that maximize her gain depending on her own move) therefore :
                LT = (lr1* APE); % Learning Term
                pstar= pstar + LT + (3*kappa*pstar*(1-pstar)*((mod(as(t),2)-Qstar) - (3*omega*Qstar*(1-Qstar)))); % [q*(t)] : update given what the subject's action the opponent has experienced
                % NORMALIZATION
                if pstar < epsilon %if pstar too small, or 0, then superior to 0 
                    pstar = epsilon; % or pstar= epsilon ?
                elseif pstar > (1-epsilon)
                    pstar = (1-epsilon);
                end
                % if subject missed or not the 2 Q-values associated to the 2 actions are updated
                % c) Q-value update from estimated pstar
                Q(1,1) = 1*(1-pstar); % expected value of action 1 "work" given opponent's action proba -> 0*(pstar)+0.5*(1-pstar) 
                Q(1,2) = 2*(pstar); % expected value of action 2 "not work" -> 1*(pstar)+0*(1-pstar)
            elseif pl(t) == 2 % subject is the employer (Hampton's nomenclature : q = subject-as-employer's proba action, and q** = subject-as-employer's action proba simulated by the opponent-as-employer)
                % a)
                Qstar= (1/3)+((1/weight)*log((1-pstar)/pstar)); % [q**(t)] in Hampton's equations
                % NORMALIZATION
                if Qstar < epsilon %if pstar too small, or 0, then superior to 0 
                    Qstar = epsilon; % or pstar= epsilon ?
                elseif Qstar > (1-epsilon)
                    Qstar = (1-epsilon);
                end
                % b)
                LT = (lr1* APE); % Learning Term
                pstar= pstar + LT - (3*kappa*pstar*(1-pstar)*((mod(as(t),2)-Qstar) + (3*omega*Qstar*(1-Qstar)))); % [p*(t)]
                % NORMALIZATION
                if pstar < epsilon %if pstar too small, or 0, then superior to 0 
                    pstar = epsilon; % or pstar= epsilon ?
                elseif pstar > (1-epsilon)
                    pstar = (1-epsilon);
                end
                % c)
                Q(1,1) = 2*pstar; % expected value action 1 "not inspect" -> 1*(pstar)+0*(1-pstar) 
                Q(1,2) = 1*(1-pstar); % expected value of action 2 "inspect" exp value -> 0*(pstar)+0.5*(1-pstar)
            end  
        end   
    end
end 
lik = -lik;   % Inverse LogLikelihood for minimisation in fmincon (gradient descent process)

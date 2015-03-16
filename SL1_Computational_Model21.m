function lik = SL1_Computational_Model21(params, pl, ss, as, rs, so, ao, ro, model) % input vectors: pl= player role of subject, (s= state, a= action, r= reward) of subject (s) and opponent (o), at each trial
% This function estimates the inverse loglikelihood of each model for each parameter value that fmincon tests in the optimization process
% Thibaud Griessinger, Paris, November, 2014
% Adapted from : "Thibaud Griessinger, Los Angeles, February, 2014 - adapted from "Stefano Palminteri, Paris, October, 2013""

% 0, 50, 100 pts converted in 0, 1, 2 units in outcome data and models 

%% Parameters in each model 
Beta = params(1);                  % choice (inverse) temperature (1/T). Note : High temperatures (beta -> 0) cause all actions to be nearly equiprobable, whereas low temperatures (beta -> Inf) cause greedy action selections                                   
lr1  = params(2);                 % factual learning rate
lr2  = params(3);
%if model > 6; lr2=params(3); end; % Influence parameter (K) / simulation of the other's also using a Fictitious Play 
 
%% Initial values 

if model == 1 || model == 3 || model == 5 || model == 7 || model == 9 || model == 11 || model == 13
%pstar = (p*) proba opponent chooses action 1 (ao(t) ==1), since the other action proba is complementary and = 1-pstar
    pstar  = 1/2; % random level oponent's action 1 and 2
else % if model == 3 || model == 5 || model == 7
    pstar = 1/3; % subject might also have initial prior over her opponent's action given the payoff matrix, therefore the opponent's mixed strategy p= 1/3 could be enter as inital action proba (like in this case where opponent's as player2, but also for opponent's as player1 )
end 

% Q table for each pair (state, actions) 
% Q = zeros(ss(t),max(as)); % State in row, action in column 
% BUT here there is only 1 state, and 2 actions. Therefore could be simply put as: 
Q       = zeros(2,2); % Expected values of actions 1 and 2 in state 1. 

% Qvalue initial = initial expected value depending on the inital belief over the opponent's action proba. Therefore depends on the role of the subject 
% if model == 1
%     Q(1,1)  = 0.75;
%     Q(1,2)  = 0.75;
% else % if model == 2 || model == 3 || model == 5 || model == 7 ||
if pl == 1 % employee (Q(1,1) -> Q-value action "work", Q(1,2) -> Q-value action "not work")
    Q(:,1)  = 1*(1-pstar); 
    Q(:,2)  = 2*(pstar); 
else % pl == 2 % employer (Q(1,1) -> Q-value action "inspect", Q(1,2) -> Q-value action "not inspect")
    Q(:,1)  = 2*(pstar); 
    Q(:,2)  = 1*(1-pstar); 
end
% end

lik     = 0; % Initial loglikelihood

% "Normalization" parameter
epsilon = 10^-6;

for t = 1:length(as);

    %% Counterfactual-Self Q-RL (stefano's)
    % a) Model does not consider trials missed by the subject (as(t)== 0) (and those both missed (ao(t)== 0))
    if model == 1 || model == 2
        % Agent experiences a reward for the action selected and generates a Reward Prediction Error
        if as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            
            % Factual Update: 
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(1,as(t)); % (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(1,as(t)) = Q(1,as(t)) + (lr1*RPE);
            
            % Counterfactual Update
            % 0) generates counterfactual reward. What would have been obtained if chose differently 
            if pl(t) == 1 % model as agent is the employee (Hampton's nomenclature : p = subject-as-employer's proba action, and p** = subject-as-employer's action proba simulated by the opponent-as-employee)
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 2;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 1;
                end
            elseif pl(t) == 2 % model as agent is the employee (Hampton's nomenclature : p = subject-as-employer's proba action, and p** = subject-as-employer's action proba simulated by the opponent-as-employee)
                if as(t) == 1 && ao(t) == 1
                    rs_u(t)= 0;
                elseif as(t) == 1 && ao(t) == 2
                    rs_u(t)= 2;
                elseif as(t) == 2 && ao(t) == 1
                    rs_u(t)= 1;
                elseif as(t) == 2 && ao(t) == 2
                    rs_u(t)= 0;    
                end
            end
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE_u =  rs_u(t) - Q(1,(mod(as(t),2)+1)); % (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(1,(mod(as(t),2)+1)) = Q(1,(mod(as(t),2)+1)) + (lr2*RPE_u);
        end


    %% 2-rand-States Q-RL #1 
    elseif model == 3 || model == 4 
        if  as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % what state are we in ? 
            s = RandomChoice([0.5 0.5]); % state selected randomly [1:2]
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            % Factual Update: 
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);
        end

    %% 2-prec-States Q-RL #2 
    elseif model == 5 || model == 6 
        if  as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % what state are we in ? 
            if t > 1 && (ao(t-1) ~= 0) % if as(t)~= 0 then ao(t)~= 0 but ao(t-1) coul be = 0 !
                s = ao(t-1); % same state than previous one (previous other's choice)
            else 
                s= RandomChoice([0.5 0.5]);
            end
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            % Factual Update: 
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);
        end
        
    %% 2-invprec-States Q-RL #3 
    elseif model == 7 || model == 8 
        if  as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % what state are we in ? `
            if t > 1 && (ao(t-1) ~= 0) % if as(t)~= 0 then ao(t)~= 0 but ao(t-1) coul be = 0 !
                s = (mod(ao(t-1),2)+1); % diff state than previous one (previous other's choice)
            else 
                s= RandomChoice([0.5 0.5]);
            end
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            % Factual Update: 
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);
        end


    %% 2-meanadvant-States Q-RL #4 
    elseif model == 9 || model == 10 
        if  as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % what state are we in ? 
            if mean(Q(1,:),2) ~= mean(Q(2,:),2) % if mean q-values different
                s = find(mean(Q(:,:),2)== max(mean(Q(:,:),2))); % diff state than previous one (previous other's choice)
            else 
                s= RandomChoice([0.5 0.5]);
            end
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            % Factual Update: 
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);
        end

    %% 2-advant-States Q-RL #5 
    elseif model == 11 || model == 12
        if  as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % what state are we in ? 
            if max(Q(1,:)) ~= max(Q(2,:)) % if only one q-value max
                mx = mod(find((Q(:,:)) == max(max(Q(:,:)))),2); % diff state than previous one (previous other's choice)
                if mx == 0 % mx is even
                    s = 2;
                else 
                    s= 1;
                end
            else 
                s= RandomChoice([0.5 0.5]);
            end
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            % Factual Update: 
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);
        end


    %% 2-clever-States Q-RL #6
    elseif model == 13 || model == 14
        if  as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % what state are we in ? 
            freq_o = sum(ao(1:t))/t; % count frequency choice opponent from trial 1
            if freq_o < 1.5
                s = 1;
            elseif freq_o > 1.5
                s = 2;
            else
                s= RandomChoice([0.5 0.5]);
            end
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(s,(mod(as(t),2)+1))- Q(s,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            % Factual Update: 
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(ao(t),as(t)); % state = other's action (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(ao(t),as(t)) = Q(ao(t),as(t)) + (lr1*RPE);
        end

    end
end 
lik = -lik;   % Inverse LogLikelihood for minimisation in fmincon

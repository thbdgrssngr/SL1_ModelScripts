%% Model 1-2: the 3 Hampton's Models with different initial values%%
function lik = SL1_Computational_Model12(params, pl, ss, as, rs, so, ao, ro, model) % input vectors: pl= player role of subject, (s= state, a= action, r= reward) of subject (s) and opponent (o), at each trial
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

if model == 1 || model == 2 || model == 4 || model == 6
%pstar = (p*) proba opponent chooses action 1 (ao(t) ==1), since the other action proba is complementary and = 1-pstar
    pstar  = 1/2; % random level oponent's action 1 and 2
else % if model == 3 || model == 5 || model == 7
    pstar = 1/3; % subject might also have initial prior over her opponent's action given the payoff matrix, therefore the opponent's mixed strategy p= 1/3 could be enter as inital action proba (like in this case where opponent's as player2, but also for opponent's as player1 )
end 
Qstar = 0.5 ; % proba own action at random 

% Q table for each pair (state, actions) 
% Q = zeros(ss(t),max(as)); % State in row, action in column 
% BUT here there is only 1 state, and 2 actions. Therefore could be simply put as: 
Q       = zeros(1,2); % Expected values of actions 1 and 2 in state 1. 
% Qvalue initial = initial expected value depending on the inital belief over the opponent's action proba. Therefore depends on the role of the subject 
if model == 1
    Q(1,1)  = 0.75;
    Q(1,2)  = 0.75;
else % if model == 2 || model == 3 || model == 5 || model == 7 ||
    if pl == 1 % employee (Q(1,1) -> Q-value action "work", Q(1,2) -> Q-value action "not work")
        Q(1,1)  = 1*(1-pstar); 
        Q(1,2)  = 2*(pstar); 
    else % pl == 2 % employer (Q(1,1) -> Q-value action "inspect", Q(1,2) -> Q-value action "not inspect")
        Q(1,1)  = 2*(pstar); 
        Q(1,2)  = 1*(1-pstar); 
    end
end

lik     = 0; % Initial loglikelihood

% "Normalization" parameter
epsilon = 10^-6;

for t = 1:length(as);

    %% Q-Learning
    % a) Model does not consider trials missed by the subject (as(t)== 0) (and those both missed (ao(t)== 0))
    if model == 1 || model == 2 || model == 3 
        % Agent experiences a reward for the action selected and generates a Reward Prediction Error
        if as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            % 1) Model generates a Reward Prediction Error from the reward experienced by the subject
            RPE =  rs(t) - Q(1,as(t)); % (Reward Obtained for action selected - Q-Value stored of this action)
            % 2) Model updates estimated Q-value of the action the subject selected: 
            Q(1,as(t)) = Q(1,as(t)) + (lr1*RPE);
        end
        

    %% Fictitious Play 

    % a) Model does not consider trials missed by the subject (as(t)== 0) (and those both missed (ao(t)== 0)),  experienced the other's action
    elseif model == 4 || model == 5 
        % Outcome observation: action value update: 
        if  as(t)~= 0 % Even if opponent's action experienced, subject do not update own's action proba since not choose at time and therefore default action selected
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta))));
            % 1) Agent observes the opponent's action and generates an Action Prediction Error over her (observed) behavior only if she did not miss
            APE =  mod(ao(t),2) - pstar; % [if as(t)~=0 then ao(t)~=0] APE : (Action) Prediction Error experienced by the subject between the opponent?s expected action p* and whether the opponent chose action 1 at time t (P= 1, mod(oppChoice=1,2)), or chose another action (P = 0, mod(oppChoice=2,2)).
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
        
        
    %% Influence

    % a) Model does not consider trials missed by the subject (as(t)== 0) (and those both missed (ao(t)== 0)). even the other experienced the default action selected for her, since she missed she might not think the other experienced it.
    elseif model == 6 || model == 7
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
                pstar= pstar + LT + (lr2*weight.*pstar.*(1-pstar).*(mod(as(t),2)-Qstar)); % [q*(t)] : update given what the subject's action the opponent has experienced
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
                pstar= pstar + LT - (lr2*weight*pstar*(1-pstar)*(mod(as(t),2)-Qstar)); % [p*(t)]
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
lik = -lik;   % Inverse LogLikelihood for minimisation in fmincon

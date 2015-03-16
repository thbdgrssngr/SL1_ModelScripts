%% Roth&Erev + EWA 
function lik = SL1_Computational_Model14(params, pl, ss, as, rs, so, ao, ro, model) % input vectors: pl= player role of subject, (s= state, a= action, r= reward) of subject (s) and opponent (o), at each trial
% This function estimates the inverse loglikelihood of each model for each parameter value that fmincon tests in the optimization process
% Thibaud Griessinger, Paris, February, 2015

% 0, 50, 100 pts converted in 0, 1, 2 units in outcome data and models 

%% Parameters in each model 
Beta = params(1);                  % choice (inverse) temperature (1/T). Note : High temperatures (beta -> 0) cause all actions to be nearly equiprobable, whereas low temperatures (beta -> Inf) cause greedy action selections                                   
lr1  = params(2);                  % factual learning rate
lr2  = params(3);

%% Initial values 

if model == 1 || model == 3
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
    Q(1,1)  = 1*(1-pstar); 
    Q(1,2)  = 2*(pstar); 
else % pl == 2 % employer (Q(1,1) -> Q-value action "inspect", Q(1,2) -> Q-value action "not inspect")
    Q(1,1)  = 2*(pstar); 
    Q(1,2)  = 1*(1-pstar); 
end
% end

lik     = 0; % Initial loglikelihood

% "Normalization" parameter
epsilon = 10^-6;

for t = 1:length(as);

    %% Roth & Erev RL % http://andromeda.rutgers.edu/~jmbarr/EEA2009/chen.pdf
    if model == 1  
        omeg = lr1; 
        epsil = lr2; 
        % Agent experiences a reward for the action selected and generates a Reward Prediction Error
        if as(t)~= 0 % Model updates the estimated Q-value of the action chosen by the subject and generate its choice proba only if she selected the action at time (trial not missed) 
            
            % Cumulated Lok likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            
            % Factual Update: 
            % 1) Model updates estimated Q-value of the action the subject selected: 
            Q(1,as(t)) = (1-omeg)*Q(1,as(t)) + (rs(t)*(1-epsil));
            % Counterfactual Update:
            % 2) Model updates estimated Q-value of the action the subject did not selected: 
            Q(1,(mod(as(t),2)+1)) = (1-omeg)*Q(1,(mod(as(t),2)+1)) + (rs(t)*(epsil));
        end


    % %% EWA (Hamton)
    % elseif model == 2
    %     ?????? equation in Supp Info ??????

    %% EWA (Zhu)
    elseif model == 2 || model == 3
        phi = lr1; % 
        delta = lr2;
        if as(t)~= 0
            % Cumulated Log likelihood of the data given the free parameters : likelihood of the model to select as(t)). Goal -> closest to 0 (log(p(as)=1))
            lik = lik + log(1/ (1+ (exp((Q(1,(mod(as(t),2)+1))- Q(1,as(t)))*Beta)))); % equivalent to Stefano-fromDaw10's : lik = lik + beta1 * Q(s(t),a(t)) - log(sum(exp(beta1 * Q(s(t),:)))) % with s(t) state at trial i, here = 1
            
            % Estimate Counterfactual Reward given subject and the opponent's choices: 
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

            % Action Values update given weight RL/"BB" - 2 values updated each trial = Q and N
            N = 1; % N(t) = (rho * NN(t)) +1; The initial experience N (0) was included in the original EWA model so that Bayesian learning models are nested as a special case—N (0) represents the strength of prior beliefs. We restrict N (0) = 1 here because its influence fades rapidly as an experimentprogresses and mostsubjects come to experiments with weak priors anyway. T
            Q(1,as(t)) = (phi * N * Q(1,as(t)) + rs(t)) /N;
            Q(1,(mod(as(t),2)+1)) = (phi * N * Q(1,(mod(as(t),2)+1)) + (delta*rs_u(t))) /N;

        end
        
    end
end 
lik = -lik;   % Inverse LogLikelihood for minimisation in fmincon

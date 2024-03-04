from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from interface import Agent
import gymnasium as gym
import pickle
from evaluate import evaluate_HIV

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
class ReplayBuffer() :
    def __init__(self,buffer_size,state_size) :
        self.buffer_size=buffer_size
        self.buffer_pos=0
        self.observations=np.zeros((buffer_size,state_size),dtype=np.float32)
        self.next_observations=np.zeros((buffer_size,state_size),dtype=np.float32)
        self.actions=np.zeros(buffer_size,dtype=np.uint8)
        self.rewards=np.zeros(buffer_size,dtype=np.float32)
        self.dones=np.zeros(buffer_size,dtype=np.bool_)

    def sample(self,bath_size) :
        if self.buffer_pos>=self.buffer_size :
            valid_idxs=np.arange(self.buffer_size)
        else :
            valid_idxs=np.arange(self.buffer_pos)

        idxs=np.random.choice(valid_idxs,bath_size)

        return self.observations[idxs],\
               self.actions[idxs],\
               self.rewards[idxs],\
               self.next_observations[idxs],\
               self.dones[idxs]
    
    def store(self,observation,action,reward,next_observation,done) :
        pos=self.buffer_pos%self.buffer_size
        self.observations[pos]=observation
        self.actions[pos]=action
        self.rewards[pos]=reward
        self.next_observations[pos]=next_observation
        self.dones[pos]=done
        self.buffer_pos+=1

    def save(self,fname) :
        with open(fname,'wb') as f :
            pickle.dump({'buffer_pos':self.buffer_pos,\
                         'observations':self.observations,\
                         'actions':self.actions,\
                         'rewards':self.rewards,\
                         'next_observations':self.next_observations,\
                         'dones':self.dones},f)
    def load (self,fname) :
        with open(fname,'rb') as f :
            save=pickle.load(f)

        self.buffer_pos=save['buffer_pos']
        self.observations=save['observations']
        self.actions=save['actions']
        self.rewards=save['rewards']
        self.next_observations=save['next_observations']
        self.dones=save['dones']
        
    def can_sample(self,bath_size) :
        return bath_size<=self.buffer_pos

class DQN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers,
        output_dim,
        dropout
    ):
        super().__init__()
        self.ffn = nn.Sequential()
        self.ffn.append(nn.Linear(input_dim, hidden_layers[0]))
        self.ffn.append(nn.ReLU())
        for k in range(len(hidden_layers)-1) :
            if dropout is not None :
                self.ffn.append(nn.Dropout(p=dropout))
            self.ffn.append(nn.Linear(hidden_layers[k], hidden_layers[k+1]))
            self.ffn.append(nn.ReLU())
        self.ffn.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x: torch.tensor):
        x = self.ffn(x)
        return x
     
class ProjectAgent:
    def __init__(self,
                gamma=0.95, #discount factor for bellman equation
                grad_iter=3, #number of descent grad per train iteraion
                hidden_layers=[256,256,256,256,256], #size of the hidden layers of the DQN
                learning_rate=0.0007, #learning rate of optimizer
                buffer_size=200_000, #buffer size
                batch_size=1024, #batch size during learning
                update_cycle=400, #number of train iteration between target/policy sync
                epsilon1=1.,  #epsilon exploration at the beginning of the learning
                epsilon2=0.02, #epsilon exploration at the end of the learning
                n1=0, #number of episode before epsilon decay
                n2=100, #number of episode at the end of epsilon decay
                dropout=None, #dropout for the DQN
                ) :
        
        self.gamma=gamma
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.action_space.n
        
        self.hidden_layers=hidden_layers
        self.learning_rate=learning_rate
        self.grad_iter=grad_iter

        self.target_network=DQN(input_dim=self.state_size,
                                hidden_layers=hidden_layers,
                                output_dim=self.action_size,
                                dropout=dropout)
        self.target_network.eval()
        self.policy_network=DQN(input_dim=self.state_size,
                                hidden_layers=hidden_layers,
                                output_dim=self.action_size,
                                dropout=dropout)
        
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.target_update()

        self.eval=True

        self.buffer=ReplayBuffer(buffer_size,self.state_size)
        
        self.batch_size=batch_size
        self.update_cycle=update_cycle
        self.n_step_train=0
        
        self.n1=n1
        self.n2=n2
        self.epsilon1=epsilon1
        self.epsilon2=epsilon2

    def act(self, observation, use_random=False):
        if not self.eval and np.random.random()<=self.epsilon_fnct() :
            return np.random.randint(0,self.action_size)
        with torch.no_grad():
            observation=torch.as_tensor(observation,dtype=torch.float32)
            q_values=self.policy_network(observation)
        
        action =q_values.argmax().tolist()
        return action 
    
    def epsilon_fnct(self) :
        """
        Returns the current epsilon for exploration
        """
        if self.n_step_train<=self.n1*200 :
            return self.epsilon1
        elif self.n_step_train<=self.n2*200 :
            return self.epsilon1-(self.epsilon1-self.epsilon2)*(self.n_step_train-self.n1*200)/(self.n2*200-self.n1*200)
        else :
            return self.epsilon2

    def train(self) :
        """
        Performs gradient descent on the DQN
        """
        if self.eval or not self.buffer.can_sample(self.batch_size) :
            return
        for _ in range(self.grad_iter) :
            observations,actions,rewards,next_observations,dones = self.buffer.sample(self.batch_size)
            observations=torch.as_tensor(observations,dtype=torch.float32)
            actions=torch.as_tensor(actions,dtype=torch.int64)
            rewards=torch.as_tensor(rewards,dtype=torch.float32)
            dones=torch.as_tensor(dones,dtype=torch.float32)
            next_observations=torch.as_tensor(next_observations,dtype=torch.float32)
    
            with torch.no_grad():
                next_q_values_policy=self.policy_network(next_observations)
                next_q_values_target=self.target_network(next_observations)
                next_actions=next_q_values_policy.argmax(axis=-1)
    
                next_q_values_target=next_q_values_target.gather(index=next_actions[:,None],axis=1)[:,0]
                
                target=rewards+self.gamma * next_q_values_target * (1.-dones)
    
            q_values=self.policy_network(observations)
            q_values=q_values.gather(index=actions[:,None],axis=1)[:,0]
            
            loss = F.mse_loss(q_values,target)
    
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.n_step_train+=1
        if self.n_step_train%self.update_cycle==0 :
            self.target_update()

    def store(self,observation,action,reward,next_observation,done) :
        """
        Store a transition in the buffer
        """
        self.buffer.store(observation,action,reward,next_observation,done)
    
    def eval_on(self) :
        """
        Activate evaluation
        """
        self.eval=True
        self.policy_network.eval()
    
    def eval_off(self) :
        """
        Activate training
        """
        self.eval=False
        self.policy_network.train()

    def target_update(self) :
        """
        Update the target network with policy network weights
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save(self, path='./model', path_buffer=None):
        """
        Save the networks and the buffer if path_buffer is not None
        """
        torch.save({"target_network":self.target_network.state_dict(),
                    "policy_network":self.policy_network.state_dict(),
                    "n_step_train":self.n_step_train},
                    path)
        if path_buffer is not None :
            self.buffer.save(path_buffer)

    def load(self,path='./model',path_buffer=None):
        """
        Load the networks and the buffer if path_buffer is not None
        """
        model = torch.load(path)
        self.policy_network.load_state_dict(model["policy_network"])
        self.target_network.load_state_dict(model["target_network"])
        self.n_step_train=model["n_step_train"]
        if path_buffer is not None :
            self.buffer.load(path_buffer)
        
def train_agent(agent: Agent, env: gym.Env, nb_episode: int = 200) -> float:
    agent.eval_off()
    for k in range(1,nb_episode+1):
        obs, info = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            action = agent.act(obs)
            new_obs, reward, done, truncated, _ = env.step(action)
            agent.store(obs,action,reward,new_obs,done)
            agent.train()
            obs=new_obs
        if k%10==0 :
            print(f"Evaluation epoch {k}")
            agent.eval_on()
            eval_score = evaluate_HIV(agent=agent, nb_episode=1)
            print(f"{eval_score=}")
            agent.save(f"./model_{k}.pt")

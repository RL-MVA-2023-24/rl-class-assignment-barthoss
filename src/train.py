from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from interface import Agent
import gymnasium as gym
import pickle
import random
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
class DQN(nn.Module):
    """DQN https://www.nature.com/articles/nature14236.pdf"""

    def __init__(
        self,
        input_dim,
        hidden_layers,
        output_dim
    ):
        super().__init__()
        self.ffn = nn.Sequential()
        self.ffn.append(nn.Linear(input_dim, hidden_layers[0]))
        self.ffn.append(nn.ReLU())
        for k in range(len(hidden_layers)-1) :
            self.ffn.append(nn.Linear(hidden_layers[k], hidden_layers[k+1]))
            self.ffn.append(nn.ReLU())
        self.ffn.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x: torch.tensor):
        x = self.ffn(x)
        return x
     
class ProjectAgent:
    def __init__(self,
                gamma=0.95,
                hidden_layers=[32,128,256,128,32],
                learning_rate=0.001,
                buffer_size=50000,
                batch_size=512,
                update_cycle=200,
                n1=50,
                n2=100,
                epsilon1=1.,
                epsilon2=0.01) :
        
        self.gamma=gamma
        self.state_size=env.observation_space.shape[0]
        self.action_size=env.action_space.n
        
        self.hidden_layers=hidden_layers
        self.learning_rate=learning_rate

        self.target_network=DQN(input_dim=self.state_size,
                                hidden_layers=hidden_layers,
                                output_dim=self.action_size)
        self.target_network.eval()
        self.policy_network=DQN(input_dim=self.state_size,
                                hidden_layers=hidden_layers,
                                output_dim=self.action_size)
        
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        self.target_update()

        self.eval=True

        self.buffer_size=buffer_size
        self.buffer_pos=0
        self.buffer={'observations':np.zeros((buffer_size,self.state_size),dtype=np.float32),
                     'next_observations':np.zeros((buffer_size,self.state_size),dtype=np.float32),
                     'actions':np.zeros(buffer_size,dtype=np.uint8),
                     'rewards':np.zeros(buffer_size,dtype=np.float32),
                     'dones':np.zeros(buffer_size,dtype=np.bool_)}
        
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
        if self.n_step_train<=self.n1*200 :
            return self.epsilon1
        elif self.n_step_train<=self.n2*200 :
            return self.epsilon1-(self.epsilon1-self.epsilon2)*(self.n_step_train-self.n1*200)/(self.n2*200-self.n1*200)
        else :
            return self.epsilon2

    def train(self) :
        if self.buffer_pos<self.batch_size :
            return
        
        observations,actions,rewards,next_observations,dones = self.sample()
        observations=torch.as_tensor(observations,dtype=torch.float32)
        actions=torch.as_tensor(actions,dtype=torch.int64)
        rewards=torch.as_tensor(rewards,dtype=torch.float32)
        dones=torch.as_tensor(dones,dtype=torch.float32)
        next_observations=torch.as_tensor(next_observations,dtype=torch.float32)

        with torch.no_grad():
            next_q_values_policy=self.policy_network(next_observations)
            next_q_values_target=self.target_network(next_observations)
            next_actions=next_q_values_target.argmax(axis=-1)

            next_q_values_target=next_q_values_policy.gather(index=next_actions[:,None],axis=1)[:,0]
            
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
        pos=self.buffer_pos%self.buffer_size
        self.buffer['observations'][pos]=observation
        self.buffer['actions'][pos]=action
        self.buffer['rewards'][pos]=reward
        self.buffer['next_observations'][pos]=next_observation
        self.buffer['dones'][pos]=done
        self.buffer_pos+=1

    def sample(self) :
        if self.buffer_pos>=self.buffer_size :
            valid_idxs=np.arange(self.buffer_size)
        else :
            valid_idxs=np.arange(self.buffer_pos)

        idxs=np.random.choice(valid_idxs,self.batch_size)

        return self.buffer['observations'][idxs],\
               self.buffer['actions'][idxs],\
               self.buffer['rewards'][idxs],\
               self.buffer['next_observations'][idxs],\
               self.buffer['dones'][idxs]
    
    def eval_on(self) :
        self.eval=True
        self.policy_network.eval()
    
    def eval_off(self) :
        self.eval=False
        self.policy_network.train()

    def target_update(self) :
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save(self, path):
        torch.save({"target_network":self.target_network.state_dict(),
                    "policy_network":self.policy_network.state_dict(),
                    "buffer":self.buffer,
                    "buffer_pos":self.buffer_pos,
                    "n_step_train":self.n_step_train},
                    path)

    def load(self,path='./model'):
        model = torch.load(path)
        self.policy_network.load_state_dict(model["policy_network"])
        self.target_network.load_state_dict(model["target_network"])    
        self.buffer=model["buffer"]  
        self.buffer_pos=model["buffer_pos"]  
        self.n_step_train=model["n_step_train"]
        
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

def train_agent(agent: Agent, env: gym.Env, nb_episode: int = 200) -> float:
    agent.eval_off()
    for k in range(1,nb_episode+1):
        seed_everything(k)
        obs, info = env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            action = agent.act(obs)
            new_obs, reward, done, truncated, _ = env.step(action)
            agent.store(obs,action,reward,new_obs,done)
            agent.train()
            obs=new_obs

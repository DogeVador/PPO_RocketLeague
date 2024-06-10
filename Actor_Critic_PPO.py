import rlgym
import gym
import matplotlib.pyplot as plt
import math
import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim
import Rewards
from ActionParsers import DiscreteAction
from StateSetters import CustomStateSetter
from ObsBuilder import CustomObsBuilder
from Rewards import CustomReward

class MultiCategoricalDistribution:

    def __init__(self, action_dims):
        super(MultiCategoricalDistribution, self).__init__()
        self.distribution = None
        self.action_dims = action_dims

    def proba_distribution(self, output) -> "MultiCategoricalDistribution":
        self.distribution = [Categorical(chunk) for chunk in output]
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=1))],dim=1
        ).sum(dim=1)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample().unsqueeze(0) for dist in self.distribution], dim=1)

    def mode(self) -> torch.Tensor:
        return torch.stack([torch.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

class FeatureExtractor(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(FeatureExtractor,self).__init__()
        self.Extractor=nn.Linear(input_dim,output_dim)

    def forward(self,observations):
        x=self.Extractor(observations)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_bins):
        super(ActorCritic, self).__init__()
        self.chunk_sizes = [n_bins] * 5 + [2] * 3
        self.extractor = FeatureExtractor(state_dim,128)
        self.LSTM = nn.LSTM(128, 256, num_layers=1)
        self.valuelayer1=nn.Linear(256,64)
        self.valuelayer2=nn.Linear(64,1)
        self.policylayer1 = nn.Linear(256, 128)
        self.policylayer2 = nn.Linear(128, 64)
        self.policylayer3 = nn.Linear(64, sum(self.chunk_sizes))
        self.softmax = nn.Softmax(dim=-1)
        self.call=0

    def forward(self, x, lstm_states):
        self.call+=1
        x = self.extractor(x)
        x=x.unsqueeze(0)

        lstm_out, lstm_states = self.LSTM(x, lstm_states)
        lstm_out = lstm_out.squeeze(0)  # Remove sequence length dimension

        pi = torch.relu(self.policylayer1(lstm_out))
        pi = torch.relu(self.policylayer2(pi))
        pi = self.policylayer3(pi)

        # Chunk sizes should match the output dimensions correctly
        chunks = torch.split(pi, self.chunk_sizes, dim=-1)

        # Apply softmax to the correct dimensions
        softmax_outputs = [self.softmax(chunk) for chunk in chunks]

        vf=torch.relu(self.valuelayer1(lstm_out))
        vf = self.valuelayer2(vf)
        return softmax_outputs,vf,lstm_states


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.terminals = []
        self.lstm_states=[]

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.terminals[:]
        del self.lstm_states[:]

class PPO:
    def __init__(self, state_dim, gamma, K_epochs, eps_clip,n_bins,lr,deterministic):
        self.gamma=gamma
        self.K_epochs=K_epochs
        self.eps_clip=eps_clip
        self.deterministic=deterministic

        self.buffer=RolloutBuffer()
        self.policy = ActorCritic(state_dim,n_bins)
        self.optimizer=optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim,n_bins)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mseLoss=nn.MSELoss()
        self.track=[]

    def get_action(self,state,lstm_states):
        with torch.no_grad():
            state=torch.from_numpy(state).float()
            self.buffer.lstm_states.append(lstm_states)
            pi, vf ,lstm_states= self.policy_old(state,lstm_states)
            self.mask_actions(state, torch.cat(pi))
            distribution = MultiCategoricalDistribution(self.policy_old.chunk_sizes)
            distribution.proba_distribution(pi)
            action=distribution.get_actions(deterministic=False)
            log_probs=distribution.log_prob(action)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(log_probs)
        self.buffer.state_values.append(vf)
        return action,lstm_states

    def mask_actions(self,obs,pi):
        HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32)
        has_boost = obs[23]>0.0
        on_ground = obs[22]
        has_flip = obs[21]

        not_on_ground = torch.logical_not(on_ground)
        mask = torch.ones_like(pi, dtype=torch.bool)

        mask[0] = on_ground  # throttle -1
        # mask[2] = on_ground  # throttle 1

        mask[3] = on_ground # steer -1
        mask[5] = on_ground # steer 1

        mask[6] = not_on_ground # yaw -1
        mask[8] = not_on_ground # yaw 1

        mask[9] = not_on_ground  # pitch -1
        mask[11] = not_on_ground  # pitch 1

        mask[12] = not_on_ground  # roll -1
        mask[14] = not_on_ground  # roll 1

        mask[16] = has_flip  # Jump
        mask[18] = has_boost  # boost

        mask[20] = on_ground  # Handbrake

        pi = torch.where(mask, pi,HUGE_NEG)
        return torch.split(pi,self.policy_old.chunk_sizes, dim=-1)


    def make_batch(self):
        s, a, logprobs, vfs = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach(), torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach(), \
                                              torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach(), torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        h_states = [state[0] for state in self.buffer.lstm_states]
        c_states = [state[1] for state in self.buffer.lstm_states]

        h_states_stacked = torch.stack(h_states, dim=1).detach()
        c_states_stacked = torch.stack(c_states, dim=1).detach()

        batched_lstm_states = (h_states_stacked, c_states_stacked)
        rewards=self.buffer.rewards.copy()

        self.buffer.clear()

        return s, a, logprobs, vfs, batched_lstm_states,rewards

    def train(self):
        s, a, logprobs, vfs, lstm_states, rewards=self.make_batch()
        d_rewards=[]
        discounted_reward=0
        for reward in reversed(rewards):
            discounted_reward= reward + (self.gamma*discounted_reward)
            d_rewards.insert(0,discounted_reward)

        d_rewards = torch.tensor(d_rewards, dtype=torch.float32)
        d_rewards = (d_rewards - d_rewards.mean()) / (d_rewards.std() + 1e-7)

        advantages=d_rewards.detach()-vfs

        for i in range(self.K_epochs):

            pi, vf, _=self.policy(s,lstm_states)
            distribution=MultiCategoricalDistribution(self.policy_old.chunk_sizes)
            distribution.proba_distribution(pi)
            new_logprobs=distribution.log_prob(a)

            ratio=torch.exp(new_logprobs-logprobs)
            surr1=ratio*advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            # print(f'1 : {-torch.min(surr1,surr2).mean()}')
            # print(f'2 : {0.5*self.mseLoss(vf.view(-1),d_rewards).mean()}')
            # print(f'3 : {0.01 * distribution.entropy().mean()}')
            loss= -torch.min(surr1,surr2)+0.5 * self.mseLoss(vf.view(-1), d_rewards) - 0.01 * distribution.entropy()
            loss=loss.mean()
            self.track.append(loss.detach().numpy())

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            print(f"successfully trained during {i} epoch out of {self.K_epochs}")

        self.policy_old.load_state_dict(self.policy.state_dict())


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path))
        self.policy.load_state_dict(torch.load(checkpoint_path))

    def print(self):
        plt.plot(self.track)
        plt.show()

env = rlgym.make(game_speed=100,action_parser=DiscreteAction(),state_setter=CustomStateSetter(),reward_fn=(),obs_builder=CustomObsBuilder())

state_shape = env.observation_space.shape

ppo=PPO(state_shape[0],gamma=0.99, K_epochs=16, eps_clip=0.2,n_bins=3,lr=0.001,deterministic=False)

ppo.load(r'C:\Users\totos\OneDrive\Documents\Reinforcement\ActorCritic_state_dict2.pth')

n_episodes=20
obs=env.reset()

for episode in range(n_episodes):
    lstm_states = (torch.zeros(256,dtype=torch.float32).unsqueeze(0).detach(),torch.zeros(256,dtype=torch.float32).unsqueeze(0).detach())
    for _ in range(512):
        action,lstm_states=ppo.get_action(obs,lstm_states)
        obs, reward, term, _ =env.step(action.numpy())
        ppo.buffer.rewards.append(reward)
        ppo.buffer.terminals.append(term)
    ppo.train()
ppo.save(r'C:\Users\totos\OneDrive\Documents\Reinforcement\ActorCritic_state_dict2.pth')
ppo.print()
env.close()
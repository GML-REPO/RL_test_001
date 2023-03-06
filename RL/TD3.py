import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys, os
if __name__ == '__main__': sys.path.append('.')
from .base_modules import *


### Actor module
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layers = MLP(state_dim, [400,300], action_dim)
        self.max_action = max_action
    def forward(self, state):
        action = self.layers(state)
        action = torch.tanh(action) * self.max_action
        return action
        
### Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layers = MLP(state_dim+action_dim, [400,300], 1)
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = self.layers(state_action)
        return q

### TD3 
class TD3:
    def __init__(self, lr, state_dim, action_dim, device='cpu', n_critics=2):
        
        self.max_action = 1
        self.device = device
        self.n_critics = n_critics

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.train(False)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critics = []
        self.critic_targets = []
        self.critic_optimizers = []
        for i in range(n_critics):
            self.critics += [Critic(state_dim, action_dim).to(self.device)]
            self.critic_targets += [Critic(state_dim, action_dim).to(self.device)]
            self.critic_targets[i].load_state_dict(self.critics[i].state_dict())
            self.critic_optimizers += [optim.Adam(self.critics[i].parameters(), lr=lr)]
            self.critic_targets[i].train(False)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        
        loss_Q_stack = [[]]*self.n_critics
        loss_PI_stack = []
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            # state, action, reward, next_state, done = next(iter(replay_buffer))
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)#.reshape(batch_size,1)
            next_state = next_state.to(self.device)
            done = done.to(self.device)#.reshape(batch_size,1)

            with torch.no_grad():
                # Select next action according to target policy:
                noise = torch.zeros_like(action).data.normal_(0, policy_noise).to(self.device)
                noise = noise.clamp(-noise_clip, noise_clip)
                next_action = (self.actor_target(next_state) + noise)
                next_action = next_action.clamp(-self.max_action, self.max_action)
                # next_action = self.actor_target(next_state)
                
                # Compute target Q-value:
                target_Q = None
                for k in range(self.n_critics):
                    _Q = self.critic_targets[k](next_state, next_action)
                    if target_Q is None: target_Q = _Q
                    else: target_Q = torch.minimum(target_Q, _Q)
                target_Q = reward + ((1-done) * gamma * target_Q)
            
            # Optimize Critics:
            for k in range(self.n_critics):
                self.critics[k].train()
                current_Q = self.critics[k](state, action)
                loss_Q = F.mse_loss(current_Q, target_Q)
                self.critic_optimizers[k].zero_grad()
                loss_Q.backward()
                self.critic_optimizers[k].step()
                loss_Q_stack[k] += [loss_Q.item()]
                self.critics[k].train(False)
                
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                self.actor.train(True)
                actor_loss = -self.critics[0](state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor.train(False)
                loss_PI_stack.append(actor_loss.item())
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for k in range(self.n_critics):
                    for param, target_param in zip(self.critics[k].parameters(), self.critic_targets[k].parameters()):
                        target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
            loss_Q_stack = np.array(loss_Q_stack)
        return [*np.mean(loss_Q_stack, axis=1).tolist(), np.mean(loss_PI_stack)]
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, name))
        
        for i in range(self.n_critics):
            torch.save(self.critics[i].state_dict(), '%s/%s_crtic_%d.pth' % (directory, name,i))
            torch.save(self.critic_targets[i].state_dict(), '%s/%s_critic_%d_target.pth' % (directory, name,i))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
        for i in range(self.n_critics):
            self.critics[i].load_state_dict(torch.load('%s/%s_crtic_%d.pth' % (directory, name,i), map_location=lambda storage, loc: storage))
            self.critic_targets[i].load_state_dict(torch.load('%s/%s_critic_%d_target.pth' % (directory, name,i), map_location=lambda storage, loc: storage))
        
    def load_actor(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, name), map_location=lambda storage, loc: storage))
        
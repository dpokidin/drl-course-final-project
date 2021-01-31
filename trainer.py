from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
from collections import deque

class Trainer:
    def __init__(self, args, env, load_weights=False, seed=0):
        self.seed = 0
        np.random.seed(self.seed)
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.env = env
        self.agents = self._init_agents(load_weights)
        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir 
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self, load_weights):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args, load_weights)
            agents.append(agent)
        return agents

    def run(self, n_episodes):
        returns = []
        scores_deque = deque(maxlen=self.args.rerrot_every)
        for time_step in range(n_episodes):
            # reset the environment
            env_info = self.env.reset()['TennisBrain'] 
            s = env_info.vector_observations
            score = np.zeros((2,1))  
            while True:
                u = []
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)
                env_info = self.env.step(actions)['TennisBrain']  
                s_next = env_info.vector_observations                      # get the next state
                r = np.array(env_info.rewards)[:, None]                   # get the reward
                done = np.array(env_info.local_done )[:, None]                 # see if episode has finished
                score += r
                self.buffer.store_episode(s[:self.args.n_agents], u, r[:self.args.n_agents], s_next[:self.args.n_agents])
                s = s_next
                if self.buffer.current_size >= self.args.batch_size:
                    transitions = self.buffer.sample(self.args.batch_size)
                    for agent in self.agents:
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        agent.learn(transitions, other_agents)
                if np.any(done): break
            scores_deque.append(score.max())
            returns.append(score.max())
            self.noise = max(0.05, self.noise - 0.00000005)
            self.epsilon = max(0.05, self.noise - 0.00000005)
            if time_step % self.args.rerrot_every == 0:
                print(f'After episode {time_step} the average score is {np.mean(scores_deque)}')
                if np.mean(scores_deque) > .6:
                    for agent in self.agents:
                        agent.policy.save_model()
                    return returns
        return returns  
            

    def evaluate(self, times):
        returns = []
        for episode in range(times):
            # reset the environment
            env_info = self.env.reset(train_mode=False)['TennisBrain'] 
            s = env_info.vector_observations
            score = np.zeros((2,1))
            while True:
                actions = []
                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], 0, 0)
                        actions.append(action)

                env_info = self.env.step(actions)['TennisBrain']  
                s_next = env_info.vector_observations                      # get the next state
                r = np.array(env_info.rewards)[:, None]                   # get the reward
                done = np.array(env_info.local_done )[:, None]                 # see if episode has finished
                score += r
                s = s_next
                if np.any(done): break
            returns.append(score.max())
            print('Returns is', score.max())


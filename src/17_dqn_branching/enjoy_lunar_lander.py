from tqdm import tqdm
import torch 
from branching_dqn_lunar_lander import BranchingQNetwork, TensorEnv, ExperienceReplayMemory, AgentConfig, BranchingTensorEnv

ll = 'LunarLander-v2'

bins = 4
env = BranchingTensorEnv(ll, bins)
        
agent = BranchingQNetwork(env.observation_space.shape[0], env.action_space.n, bins)
agent.load_state_dict(torch.load('./runs/{}/model_state_dict'.format(ll)))

print(agent)
for ep in tqdm(range(10)):

    s = env.reset()
    done = False
    ep_reward = 0
    while not done: 

        with torch.no_grad(): 
            out = agent(s).squeeze(0)
        action = torch.argmax(out, dim = 1).numpy().reshape(-1)
        print(action)
        s, r, done, _ = env.step(action)

        env.render()
        ep_reward += r 

    print('Ep reward: {:.3f}'.format(ep_reward))

env.close()
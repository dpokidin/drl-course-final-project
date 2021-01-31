from trainer import Trainer
from common.Config import Config
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt 
env = UnityEnvironment(file_name="...") # file_name should be the path to tennis env (see Tennis.ipynd)


            
cfg = Config(n_agents=2,
             lr_actor=1e-4, 
             lr_critic=1e-3,
             epsilon=0.1,
             noise_rate=0.1,
             gamma=0.99,
             tau=0.01,
             buffer_size=int(5e5),
             batch_size=256,
             save_dir="./model",
             save_rate=2000,
             high_action=1,
             low_action=-1,
             obs_shape=2*[24],
             action_shape=2*[2],
             rerrot_every=100)


if __name__ == '__main__':
    trainer = Trainer(cfg, env)
    returns = trainer.run(4000)
    plt.plot(returns)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    

# DRL-Renegades-Game-Bots

A collection of my implemented RL agents for games like Pacman, Lunarlander, Pong, SpaceInvaders, Frozenlake, Taxi, Pixelcopter, Montezuma, Soccer, VizDoom, Kungfu-Master and a lot more by implementing various DRL algorithms using gym, unity-ml, pygame, sb3, rl-zoo, pandagym and sample factory libraries. A lot of agents implemented as part of this free course on DRL at HuggingFace: [Link](https://huggingface.co/learn/deep-rl-course/unit0/introduction) by [Thomas Simonini](https://x.com/ThomasSimonini)

## Covered DRL Topics
  * Q-learning + DQN
  * Bellman Equation + Monte Carlo
  * Hyperparameter Tuning using Optuna
  * Policy Gradient
  * DDPG & DDPGfD
  * Unity Ml Agents
  * Curiosity Driven RL - Random Network Distillation(RND)
  * Advantage Actor Critic(A2C) for Robotics
  * MultiAgent Reinforcement Learning(MARL)
  * Self Play
  * Proximal Policy Optimization(PPO) with All Variants
  * Clipped Surrogate Objective Function
  * Sample Factory
  * Asynchronous Advantage Actor-Critic (A3C)
  * Intrinsic Curiosity Module (ICM)

## Table of Implemented Agents

| **Environments**                       | **Libraries Used(includes HF)**                                       | **Algos**                    | **Kaggle Notebooks** |
|----------------------------------------|-----------------------------------------------------------------------|------------------------------|----------------------|
| LunarLander-v2                         | gym, stable-baselines3                                                | PPO                          | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-1-lunarlander)                     |
| LunarLander-v2                         | gym                                                                   | DQN(Scratch)                 | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-1-lunarlander)                     |
| Huggy                                  | unity-mlagents                                                        | PPO                          | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-1-bonus-huggydog)                     |
| FrozenLake-v1 (all variants)           | gym                                                                   | Q-Learn (Scratch)            | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-2-frozenlake-v1-taxi-v3)                    |
| Taxi-v3                                | gym                                                                   | Q-Learn (Scratch)            | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-2-frozenlake-v1-taxi-v3)                    |
| SpacesInvadersNoFrameskip-v4           | RL-Baselines3-Zoo, gym, atari                                         | DQN(CNNPolicy)               | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-3-space-invaders)                     |
| CartPole-v1                            | stable-baselines3, sb3-contrib, optuna, gym, atari, RL-Baselines3-Zoo | A2C(MlpPolicy)               | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-3-optuna-cartpole-pong-br-out)                     |
| PongNoFrameskip-v4                     | stable-baselines3, sb3-contrib, optuna, gym, atari, RL-Baselines3-Zoo | PPO(CNNPolicy)               | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-3-optuna-cartpole-pong-br-out)                     |
| BreakoutNoFrameskip-v4                 | stable-baselines3, sb3-contrib, optuna, gym, atari, RL-Baselines3-Zoo | PPO(CNNPolicy)               | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-3-optuna-cartpole-pong-br-out)                     |
| MsPacman-v5                            | gym, atari                                                            | DQN(CNNPolicy) - Scratch     | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-extra-unit-3-mspacmandqn-scratch)                     |
| CartPole-v1                            | pytorch, gym                                                          | Policy Gradient-scratch      | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-4-pg-cartpole-pixelcopter)                     |
| Pixelcopter-PLE-v0                     | pytorch, gym, pygame                                                  | Policy Gradient-scratch      | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-4-pg-cartpole-pixelcopter)                     |
| Pendulum-v1                            | pytorch, gym                                                          | DDPG-Scratch                 | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-extra-personal-unit-4-ddpg-ddpgfd-pendulum-v1)                     |
| Pendulum-v1                            | pytorch, gym                                                          | DDPGfD - Scratch             | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-extra-personal-unit-4-ddpg-ddpgfd-pendulum-v1)                     |
| Snowball-Target                        | unity-mlagents                                                        | PPO                          | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-5-unity-ml-snowball-pyramid)                     |
| Pyramids                               | unity-mlagents                                                        | PPO + RND                    | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-5-unity-ml-snowball-pyramid)                     |
| MontezumaRevengeNoFrameskip-v4         | pytorch, gym, atari                                                   | PPO + RND(CNN) - Scratch     | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-extra-personal-unit-5-rnd-montezuma-mario-bros)                     |
| Super-Mario-Bros                       | pytorch, gym, atari                                                   | PPO + RND(CNN) - Scratch     | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-extra-personal-unit-5-rnd-super-mario-bros)                     |
| PandaReachDense-v3                     | gym, panda-gym, stable-baselines3                                     | A2C(MultiInputPolicy)        | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-6-pandagym-reachdns-pickplace)                     |
| PandaPickAndPlace-v3                   | gym, panda-gym, stable-baselines3                                     | A2C(MultiInputPolicy)        | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-6-pandagym-reachdns-pickplace)                     |
| SoccerTwos                             | unity-mlagents                                                        | MA-POCA                      | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-7-marl-soccer-2vs2)                     |
| LunarLander-v2                         | pytorch, gym                                                          | PPO (All Variations Scratch) | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-8-i-ppo-scratch-lunarlander)                     |
| VizDoom (doom_health_gathering_supreme)| sample_factory                                                        | PPO                          | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-8-ii-smpfc-vizdoom-deathmatch)                     |
| Doom Deathmatch (doom_deathmatch_bots) | sample_factory                                                        | APPO                         | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-huggingface-unit-8-ii-smpfc-vizdoom-deathmatch)                     |
| KungFuMaster-v5                        | pytorch, gym                                                          | A3C-ICM-scratch              | [Link](https://www.kaggle.com/code/syedjarullahhisham/drl-extra-personal-adv-drl-a3c-icm-kungfu-master)

## HuggingFace Models
Find all my traned agents at [hishamcse agents](https://huggingface.co/hishamcse)

## Game Previews
  ![Montezuma](https://www.gymlibrary.dev/_images/montezuma_revenge.gif) ![Breakout](https://www.gymlibrary.dev/_images/breakout.gif) ![MsPacman](https://www.gymlibrary.dev/_images/ms_pacman.gif) ![Pong](https://www.gymlibrary.dev/_images/pong.gif) ![Space-Invaders](https://www.gymlibrary.dev/_images/space_invaders.gif) <img src="https://www.gymlibrary.dev/_images/taxi.gif" width="160" height="160"/> <img src="https://www.gymlibrary.dev/_images/frozen_lake.gif" width="160" height="160"/> <img src="https://www.gymlibrary.dev/_images/cart_pole.gif" width="160" height="160"/> <img src="https://www.gymlibrary.dev/_images/pendulum.gif" width="160" height="160"/>  <img src="https://www.gymlibrary.dev/_images/lunar_lander.gif" width="160" height="160"/> <img src="https://media.tenor.com/pKgBbArPChQAAAAC/mario-super.gif" height="200"/> <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS1LFs0griZFmQBd1Pw_odjoUN1vdxBY1iz3de2HQFxHMHAlqJA9un1zJRQk8F6LuNjFiM&usqp=CAU" height="200" width="200"/> <img src="https://www.gocoder.one/static/unity-ml-agents-soccertwos-b6e8a229df44d79b8d8a559338680026.gif" width="390" height="200"/> <img src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/unit-bonus1/huggy-cover.jpeg" height="250"/> ![PixelCopter](https://pygame-learning-environment.readthedocs.io/en/latest/_images/pixelcopter.gif) ![Snowball-Target](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRBXWwA-sMgbWQ9OrvPFczBEVCyAx9PheTDkGZdF-JxjpwZI510QWHSCF4dUO8KWEGLZGQ&usqp=CAU) <img src="https://unity-technologies.github.io/ml-agents/images/pyramids.png" width="220" height="220"/>  


## Acknowledgements & Resources
   * [Deep RL Course](https://huggingface.co/learn/deep-rl-course/unit0/introduction)
   * [Thomas Simonini](https://x.com/ThomasSimonini)
   * [Deep RL Course Leaderboard - HuggingFace](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard)
   * [Open RL Leaderboard - HuggingFace](https://huggingface.co/spaces/open-rl-leaderboard/leaderboard)
   * [Stable-Baseline3 Agents - HuggingFace](https://huggingface.co/sb3)
   * [Stable-Baseline3](https://stable-baselines3.readthedocs.io/en/master/)
   * [PPO-RND for Montezuma & Mario](https://github.com/alirezakazemipour/PPO-RND/tree/main)
   * [RND - Random Network Distillation](https://medium.com/data-from-the-trenches/curiosity-driven-learning-through-random-network-distillation-488ffd8e5938)
   * [All PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
   * [CleanRL Single File Implementations](https://docs.cleanrl.dev/)
   * [Unity-mlagents](https://github.com/Unity-Technologies/ml-agents)
   * [Coursera Machine Learning Specialization Course 3 by Andrew Ng](https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning?specialization=machine-learning-introduction)
   * [gym](https://www.gymlibrary.dev/index.html)
   * [Sample_Factory](https://www.samplefactory.dev/)
   * [RL-Zoo3](https://stable-baselines3.readthedocs.io/en/master/guide/rl_zoo.html)
   * [Panda-gym](https://panda-gym.readthedocs.io/en/latest/)
   * [Tools for Robotic RL](https://github.com/araffin/tools-for-robotic-rl-icra2022)
   * [Optuna](https://github.com/optuna/optuna)
   * [QRDQN - Quantile Regression DQN](https://advancedoracademy.medium.com/quantile-regression-dqn-pushing-the-boundaries-of-value-distribution-approximation-in-620af75ec5f3)
   * [Advanced DQN Pacman](https://github.com/jakegrigsby/AdvancedPacmanDQNs), [DQN Pacman](https://github.com/StarVeteran/Ms-Pacman-DQN)
   * [All Policy Gradient Algorithms Implementations](https://github.com/MrSyee/pg-is-all-you-need)
   * [DDPGfD - DDPG from Demonstrations](https://wikidocs.net/204469)
   * [Deep RL Paradise](https://github.com/alirezakazemipour/DeepRL-Paradise)
   * [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
   * [Arthur Juliani Simple Reinforcement Learning in Tensorflow Series](https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
   * [Awesome Deep RL](https://github.com/kengz/awesome-deep-rl)
   * [Dennybritz RL](https://github.com/dennybritz/reinforcement-learning)
   * [HuggingFace Hub Python Library](https://huggingface.co/docs/huggingface_hub/index)

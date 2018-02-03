import gym
import NeuralNetwork as nn


# create model
model = nn.Control_Model()


env = gym.make('CartPole-v1')
max_reward = 0
game_count = 0

while (max_reward < 500):
    observation = env.reset()

    game_count += 1
    cumulative_reward = 0
    obs_log = []
    action_log = []

    done = False

    while not done:
        #env.render()

        action = model.predict_move(observation)

        # keep a log of actions and observations
        obs_log += [observation]
        action_log += [action]

        # use action to make a move
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward


    print("Episode {} finished after {} timesteps, max = {}".format(game_count, len(action_log), max_reward))

    if (cumulative_reward > max_reward * 0.85):
        if cumulative_reward > max_reward:
            # reset the goal
            max_reward = cumulative_reward

        # train the dnn
        model.train_game(obs_log, action_log)


print('max_score: {} was acheived in {} games'.format(max_reward, game_count))
model.save_model()

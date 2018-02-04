import gym
import NeuralNetwork as nn
import evaluate_model as em


# create model
model = nn.Control_Model()
env = gym.make('CartPole-v1')
train_count = 0

# train to 500 until the average is above 500
while(em.EvalModel(model, env) < 495):
    max_reward = 0

    # run a segment of 200 'games' and train off of the max score
    for i in range(200):
        observation = env.reset()

        cumulative_reward = 0
        obs_log = []
        action_log = []
        done = False

        while not done:
            action = model.predict_move(observation)

            # keep a log of actions and observations
            obs_log += [observation]
            action_log += [action]

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        if cumulative_reward > max_reward:
            max_reward = cumulative_reward
            max_obs_log = obs_log
            max_action_log = action_log

        print('Episode {} scored {}, max {}'.format(i, cumulative_reward, max_reward))

        if max_reward >= 500:
            # if 500 has already been acheived
            break

    train_count += 1
    # train the dnn
    model.train_game(max_obs_log, max_action_log)


print('{} training episodes'.format(train_count))
# save the model for evaluation
model.save_model()

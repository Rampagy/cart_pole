import gym
import NeuralNetwork as nn


def EvalModel(model, env=gym.make('CartPole-v1')):
    cumulative_reward = 0
    num_of_games = 100

    for i in range(num_of_games):
        observation = env.reset()

        done = False

        while not done:
            if i < 5:
                # render for viewing experience
                env.render()

            action = model.predict_move(observation, train=False)

            # use action to make a move
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward

        print('current average: {} in {} games'.format(cumulative_reward/(i+1), (i+1)))

    print('average score: {}'.format(cumulative_reward/num_of_games))
    return cumulative_reward/num_of_games



if __name__ == "__main__":
    # create model
    model = nn.Control_Model()
    EvalModel(model)

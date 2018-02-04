# cart_polev1

Training/Evaluating the cart-polev1 gym from [open ai](https://gym.openai.com/envs/CartPole-v1/).

## Training

To do the training I decided to run 200 episodes and pick the max score for training.  At the end of the 200 episodes I then evaluate the new model to determine if it is 'solved'.

## Evaluating

Solving is described on the OpenAI site as achieving an average reward of 475.0 over 100 consecutive trials.  I only evaluate over 30 consecutive trials to make the training faster and I didn't see a noticeable difference in the average when evaluating with 30 vs 100 episodes.  

Evaluating the model will render the graphics so that you can visibly see the progress instead of just looking at a number.

## Usage

Simply run [train_cartpolev1.py](train_cartpolev1.py) or [evaluate_model.py](evaluate_model.py) to either train or evaluate the model.

```bash
python3 train_cartpolev1.py
python3 evaluate_model.py
```

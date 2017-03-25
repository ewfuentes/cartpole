import gym
import time
import numpy as np

SOLVED_LIMIT = 5000
NOISE_FACTOR = .001

def runEpisode(env, params):
    obs = env.reset()
    totalReward = 0
    done = False
    while not done:
        env.render()
        dp = np.dot(obs, params)
        action = 0 if dp < 0 else 1
        res = env.step(action)
        obs, reward, done, info = res
        totalReward += reward
        if totalReward > SOLVED_LIMIT:
            break
    return totalReward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    
    maxReward = 0
#    maxParams = np.random.uniform(low=-1, high=1, size = (4,1))
    maxParams = np.array([[0.30888264, 0.16867702, 0.98111015, 0.66310767]]).T
    currParamCount = 0
    while maxReward < SOLVED_LIMIT:
        newParams = maxParams + NOISE_FACTOR * np.random.uniform(low=-1.0, high=1.0, size=(4,1))
        newReward = runEpisode(env, newParams)
        currParamCount += 1
        print currParamCount, newReward, maxReward
        if newReward > maxReward:
            maxParams = newParams
            maxReward = newReward
            currParamCount = 0

        if currParamCount > 50:
            maxParams = np.random.uniform(low=-1, high=1, size=(4,1))
            currParamCount = 0
            maxReward = 0
            

    print maxReward, newParams
#    env.reset()
#    isDone = False
#    while not isDone:
#        env.render()
#        res = env.step(env.action_space.sample()) # take a random action
#        obs, reward, done, info = res
#        isDone = done
#        print obs, reward, info

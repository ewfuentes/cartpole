import gym
import time
import numpy as np
import os
import pickle
import tensorflow.contrib.slim as slim
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

#set random seeds
if os.path.exists('randState.p'):
    with open('randState.p', 'r') as fileIn:
        state = pickle.load(fileIn)
    np.random.set_state(state)
else:
    with open('randState.p', 'w') as fileOut:
        pickle.dump(np.random.get_state(), fileOut)


LEARNING_RATE = 1e-2

class Spinner:
    SPINNER_CHARS='-\|/'
    def __init__(self):
        self.state = 0

    def printSpinner(self):
        print '\b\b' + self.SPINNER_CHARS[self.state],
        sys.stdout.flush()
        self.state = (self.state + 1) & 0x03

class LunarLanderPolicy:
    def __init__(self, sess, obsSize, outputSize, numObs=3 ,hiddenLayerSize=32):
        self.sess = sess
        self.numObs = numObs
        self.obsSize = obsSize
        self.inputPlaceHolder = tf.placeholder(tf.float32,
                                               shape=[None, obsSize * numObs],
                                               name='inputLayer')
        self.fc1 = slim.fully_connected(self.inputPlaceHolder, hiddenLayerSize,
                                        scope='fc1')
        self.fc2 = slim.fully_connected(self.fc1, hiddenLayerSize, scope='fc2')
        self.outputAct = slim.fully_connected(self.fc2, outputSize,
                                           activation_fn=None, scope='outputAct')
        self.output = slim.softmax(self.outputAct, scope='softmax')

        self.selectedActions = tf.placeholder(tf.int32, [None, 2], name="selectedActions")
        self.advantage = tf.placeholder(tf.float32, [None, 1], name='targetAction')
        self.pickedActionProb = tf.gather_nd(self.output, self.selectedActions,
                                             name='pickedActionProb')
        self.loss = tf.reduce_mean(tf.multiply(-tf.log(self.pickedActionProb),
                                               self.advantage, name='lossMult'),
                                   name="meanLoss")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.trainOp = self.optimizer.minimize(self.loss) 

    def forwardPass(self, obs):
        probabilities = self.sess.run(self.output, feed_dict={
            self.inputPlaceHolder:obs
        })
        return probabilities

    def backwardPass(self, obs, target, action):
        action = np.array([[i,v] for i,v in enumerate(action)])
        
        feed_dict = {
                self.selectedActions:action,
                self.advantage:target,
                self.inputPlaceHolder:obs
        }
        _, loss = sess.run([self.trainOp, self.loss], feed_dict)
        return loss

    def createActionPicker(self):
        def actionPicker(obs):
            # First add the observation to the buffer
            if actionPicker.obsBuffer is None:
                # Repeat the observation we saw self.numObs times
                actionPicker.obsBuffer = np.hstack([obs] * self.numObs)
            else:
                actionPicker.obsBuffer = np.hstack([obs, actionPicker.obsBuffer[:-self.obsSize]])
            probs = self.forwardPass(np.expand_dims(actionPicker.obsBuffer, 0))
            probs = np.squeeze(probs)

            r = np.random.rand(1)
            for i in range(len(probs)):
                r -= probs[i]

                if r <= 0:
                    break
            return i, actionPicker.obsBuffer
        actionPicker.obsBuffer = None
        return actionPicker
            
            
def randomActionPicker(outputSpaceSize):
    def actionPicker(obs):
        return np.random.randint(0, outputSpaceSize)
    return actionPicker

def runEpisode(env, actionPicker):
    trace = {
        'obs':[],
        'reward':[],
        'action':[]
    }
    
    obs = env.reset()
    done = False
    s = Spinner()
    while not done:
        s.printSpinner()
        action, inputData = actionPicker(obs)
        trace['obs'].append(inputData)
        res = env.step(action)
        obs, reward, done, info = res
        trace['action'].append(action)
        trace['reward'].append(reward)
    env.close()
    return trace

def rollout(env, actionPicker, numEpisodes=10):
    episodes = []
    for i in range(numEpisodes):
        episodes.append(runEpisode(env, actionPicker))
        print '\b\b.  ',
        sys.stdout.flush()

    return episodes

def discountReward(reward, discountFactor=.85):
    ret = np.zeros_like(reward)
    for i in np.arange(-1, -(len(reward) +1), -1):
        ret[i] = reward[i] + discountFactor * ret[i+1] 
    return ret

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(2017)
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(2017)
    obsSize = env.observation_space.shape[0]
    outputSize = env.action_space.n
    lunarPolicy = LunarLanderPolicy(sess, obsSize, outputSize)
    sess.run(tf.global_variables_initializer())

    avgRewardTrace = []
    lossTrace = []
    fig = plt.figure()
    ax1 = plt.subplot(211)
    avgRewardLine, = plt.plot([])
    plt.title('Average Reward')
    ax2 = plt.subplot(212)
    lossLine, = plt.plot([])
    plt.title('Loss')
    plt.show(block=False)

    for i in range(200):
        print 'rollout', i, '  ',
        sys.stdout.flush()
        # Forward Pass
        episodes = rollout(env, lunarPolicy.createActionPicker(), numEpisodes=20)

        # Backward pass
        rewards = []
        actions = []
        obs = []
        for ep in episodes:
            rewards.extend(discountReward(ep['reward']))
            actions.extend(ep['action'])
            obs.extend(ep['obs'])

        rewards = (np.array(rewards) - np.mean(rewards)) / np.std(rewards)
        rewards = np.expand_dims(rewards,-1)
        loss = lunarPolicy.backwardPass(obs, rewards, actions)
        avgRewardTrace.append(np.mean([np.sum(ep['reward']) for ep in episodes]))
        lossTrace.append(loss)

        avgRewardLine.set_ydata(avgRewardTrace)
        avgRewardLine.set_xdata(range(len(avgRewardTrace)))
        lossLine.set_ydata(lossTrace)
        lossLine.set_xdata(range(len(lossTrace)))

        print avgRewardTrace[-1], lossTrace[-1]


        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw()
        plt.pause(.1)
        
        


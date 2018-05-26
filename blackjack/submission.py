import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return [1]
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state == 0:
            return [(1,0.99,1),(-1,0.01,100)]
        else:
            return []
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE
############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        value_inhand,nextcard_ifpeeked,deckcounts=state
        if deckcounts is None or sum(deckcounts) == 0:
            return []
        elif action=='Quit':
            new_state=(value_inhand,None,None)
            return [(new_state,1,value_inhand)]
        elif nextcard_ifpeeked!=None:
            if action == 'Peek':
                return []
            else:
                deckcounts_list=list(deckcounts)[:]
                deckcounts_list[nextcard_ifpeeked]-=1
                newdeckcounts=tuple(deckcounts_list)
                new_state=(value_inhand+self.cardValues[nextcard_ifpeeked],None,newdeckcounts)
                if new_state[0]>self.threshold:
                    new_state = (value_inhand + self.cardValues[nextcard_ifpeeked],None, None)
                    return [(new_state,1,0)]
                elif sum(newdeckcounts) == 0:
                    new_state = (value_inhand + self.cardValues[nextcard_ifpeeked], None, None)
                    return[(new_state, 1 , value_inhand + self.cardValues[nextcard_ifpeeked])]
                else:
                    return [(new_state,1,0)]
        else:
            transition_list=[]
            total = sum(deckcounts)
            if action=='Peek':
                for i in range(len(deckcounts)):
                    if deckcounts[i] > 0:
                        prob=1.0*deckcounts[i]/total
                        new_state=(value_inhand,i,deckcounts)
                        transition_list.append((new_state,prob,-self.peekCost))
                return transition_list
            else:
                for i in range(len(deckcounts)):
                    if deckcounts[i] > 0:
                        prob=1.0*deckcounts[i]/total
                        deckcounts_list = list(deckcounts)[:]
                        deckcounts_list[i] -= 1
                        newdeckcounts = tuple(deckcounts_list)
                        new_state = (value_inhand + self.cardValues[i], None, newdeckcounts)
                        if new_state[0] > self.threshold:
                            new_state = (value_inhand + self.cardValues[i], None, None)
                            transition_list.append((new_state, prob, 0))
                        elif sum(newdeckcounts)==0:
                            new_state = (value_inhand + self.cardValues[i], None, None)
                            transition_list.append((new_state, prob, value_inhand + self.cardValues[i]))
                        else:
                            transition_list.append((new_state, prob, 0))
                return transition_list
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    peekingMDP=BlackjackMDP([5,21],50,20,1)
    return peekingMDP
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        if newState != None:
            max_nextQ=0
            for nextaction in self.actions(newState):
                nextQ=self.getQ(newState,nextaction)
                if nextQ>max_nextQ:
                    max_nextQ=nextQ
            difference=(reward+self.discount*max_nextQ)-self.getQ(state,action)
            step_size=self.getStepSize()
            for f, v in self.featureExtractor(state, action):
                self.weights[f]+=step_size*difference*v
        # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
largeMDP.computeStates()



############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    features=[]
    key=(total,action)
    features.append((key,1))
    if counts!=None:
        presencelist=[]
        for i in range(len(counts)):
            if counts[i]>0:
                presencelist.append(1)
            else:
                presencelist.append(0)
        key=(tuple(presencelist),action)
        features.append((key,1))
    if counts!=None:
        for i in range(len(counts)):
            key=(i,counts[i],action)
            features.append((key,1))
    return features
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

'''
# Problem 4b:
rl = QLearningAlgorithm(smallMDP.actions, smallMDP.discount(),identityFeatureExtractor)
util.simulate(smallMDP, rl, numTrials=30000,verbose=False,sort=False)
rl.explorationProb=0
vi=ValueIteration()
vi.solve(smallMDP)
count = 0
for k, v in vi.pi.iteritems():
    print k,v,rl.getAction(k)
    if v!=rl.getAction(k):
        count+=1
print (count)

rl = QLearningAlgorithm(largeMDP.actions, largeMDP.discount(),identityFeatureExtractor)
util.simulate(largeMDP, rl, numTrials=30000,verbose=False,sort=False)
rl.explorationProb=0
vi=ValueIteration()
vi.solve(largeMDP)
count = 0
for k, v in vi.pi.iteritems():
    print k,v,rl.getAction(k)
    if v!=rl.getAction(k):
        count+=1
print count
print count*1.0/len(vi.pi)


# Problem 4d:
vi=ValueIteration()
vi.solve(originalMDP)
fixrl=util.FixedRLAlgorithm(vi.pi)
print 'Fixed RL:',util.simulate(newThresholdMDP, fixrl, numTrials=10,verbose=False,sort=False)
rl = QLearningAlgorithm(newThresholdMDP.actions, newThresholdMDP.discount(),identityFeatureExtractor)
util.simulate(newThresholdMDP, rl, numTrials=30000,verbose=False,sort=False)
rl.explorationProb=0
print 'Q Learning:',util.simulate(newThresholdMDP, rl, numTrials=10,verbose=False,sort=False)
'''

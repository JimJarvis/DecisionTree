from learning import *
# train the decision tree
dtLearner = DecisionTreeLearner()
dtLearner.train(zoo)
# generate decision tree code
dtLearner.dt.outputDecisionTree('./decisionTree.py')

print '>>> cat decisionTree.py\n'
for line in open('decisionTree.py'):
    print line.rstrip()
print '============================='

# make prediction
from decisionTree import *
print "example: ['shark',0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0]"
print "output: ", predict(['shark',0,0,1,0,0,1,1,1,1,0,0,1,0,1,0,0])
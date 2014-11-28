# test a classification algorithm
from learning import *
import matplotlib.pyplot as pt

car_train = DataSet(name='../data/car/car_train', attrnames='buying maint doors persons lug_boot safety values', target='values')
car_cv = DataSet(name='../data/car/car_cv', attrnames='buying maint doors persons lug_boot safety values', target='values')
car_test = DataSet(name='../data/car/car_test', attrnames='buying maint doors persons lug_boot safety values', target='values')

train_set = car_train
test_set = car_test
cv_set = car_cv

def run_test(learner, maxDeviation, 
	learning_curve_interval, test_trials, random_shuffle,  
	train_set, cv_set, test_set,
	cv_scores=None, graph=False):
	learner.train(train_set)
	if maxDeviation > 0 :
		learner.prune(cv_set, maxDeviation)
		
	if cv_scores is not None:
		cv_scores.append(test(learner, cv_set))
		return  # don't print any additional info

	print "Macro F1 score on test data: ", "%.5f" % test(learner, test_set)
	res = learningcurve(learner, maxDeviation, train_set, cv_set, test_set, 
		learning_curve_interval, test_trials, random_shuffle=random_shuffle)

	sizes = [x[0] for x in res]
	f1vals = ["%.5f" % x[1][1] for x in res]
	print "train dataset size: ", sizes
	print "average node count on test dataset: ", ["%.2f" % x[1][0] for x in res]
	print "average F1 score on test dataset: ", f1vals
	if graph:
		pt.plot(sizes, f1vals)
		pt.show()
		

learner = DecisionTreeLearner()
interval = 100

# test
print "Before Pruning"
print "========================="
run_test(learner, 0, interval, 1, False, train_set, cv_set, test_set, graph=True)

# Pruning validation: select best maxDev
maxdev = 0
print "Select best maxdev ="
cv_scores = []
sheet = []
maxdevTrials = [0.1 * x for x in range(1, 100)]
for maxdev in maxdevTrials:
    run_test(learner, maxdev, interval, 1, False, train_set, cv_set, test_set, cv_scores)
# get the best maxdev
best_cv_score, maxdev = max(zip(cv_scores, maxdevTrials))

print maxdev
print 'Best CV score =', best_cv_score
print ''

# test
print "After Pruning"
print "========================="
run_test(learner, maxdev, interval, 1, False, train_set, cv_set, test_set, graph=True)


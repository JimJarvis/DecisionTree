# test a classification algorithm
from learning import *
import matplotlib.pyplot as pt

car_train = DataSet(name='../data/car/car_train', attrnames='buying maint doors persons lug_boot safety values', target='values')
car_cv = DataSet(name='../data/car/car_cv', attrnames='buying maint doors persons lug_boot safety values', target='values')
car_test = DataSet(name='../data/car/car_test', attrnames='buying maint doors persons lug_boot safety values', target='values')

train_set = car_train
test_set = car_test
cv_set = car_cv

# table entry: [maxdev, macro CV F1, mean node count]
table = []
f1graph = []
f1sampleMaxdev = range(1, 5)

def run_test(learner, maxDeviation, 
	learning_curve_interval, test_trials, random_shuffle,  
	train_set, cv_set, test_set,
	cv_scores=None, graph=False):
	
	learner.train(train_set)
	if maxDeviation > 0 :
		learner.prune(cv_set, maxDeviation)
		
	verbose = cv_scores is None
	if not verbose:
		cv_scores.append(test(learner, cv_set))

	F1test = test(learner, test_set)
	F1cv = test(learner, cv_set)
	entry = [maxDeviation, F1cv]
	if verbose: 
		print "Macro F1 score on test data: ", "%.5f" % F1test
	
	res = learningcurve(learner, maxDeviation, train_set, cv_set, test_set, 
			learning_curve_interval, test_trials, random_shuffle=random_shuffle)

	global sizes
	sizes = [x[0] for x in res]
	f1vals = [x[1][1] for x in res]
	nodecnts = [x[1][0] for x in res]
	
	if maxDeviation in f1sampleMaxdev:
		f1graph.append(f1vals)

	if verbose:
 		print 'train dataset size: ', sizes
 		print "average node count on test dataset: ", ["{:.2f}".format(x) for x in nodecnts]
 		print "average F1 score on test dataset: ", ["{:.2f}".format(x) for x in f1vals]
	
	if maxDeviation > 0 and table is not None: 
		entry.append(sum(nodecnts)*1.0/len(nodecnts))
		table.append(entry)

# 	if graph:
# 		pt.plot(sizes, f1vals, 'ro-', sizes, [1.2*x for x in f1vals], 'bo-')
# 		pt.show()
	
		

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
maxdevTrials = [0.1 * x for x in range(1, 50)]
for maxdev in maxdevTrials:
    run_test(learner, maxdev, interval, 1, False, train_set, cv_set, test_set, cv_scores)
# get the best maxdev
best_cv_score, maxdev = max(zip(cv_scores, maxdevTrials))

# table generated
print 'maxdev\tF1\t#nodes'
for entry in table:
	print '{:.2f}\t{:.2f}\t{:.1f}'.format(*entry)

# plot 1->2 maxdev VS F1
pt.plot(*(zip(*table)[:2]+['gx-']))
pt.show()
# plot 1->3 maxdev VS treesize
pt.plot(*(zip(*table)[::2]+['rx-']))
pt.show()

print maxdev
print 'Best CV score =', best_cv_score
print ''

# graph the learning curves side by side
pt.plot(*(sum([[sizes, f1val, color + 'o-'] for f1val, color in zip(f1graph, 'brgm')], [])))
pt.show()

# test
print "After Pruning"
print "========================="
run_test(learner, maxdev, interval, 1, False, train_set, cv_set, test_set, graph=True)


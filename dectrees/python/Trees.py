import monkdata as m 
import dtree




monks = [m.monk1, m.monk2, m.monk3]
monk_tests = [m.monk1test, m.monk2test, m.monk3test]
# assign 6
print('**** Assign 6 : pruning ****')

from random import shuffle


def partition(data, fraction):
    ldata = list(data)
    shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def prune(tree, prune_data):
    all_pruned = dtree.allPruned(tree)

    dirty = False

    for pruned in all_pruned:
        if dtree.check(tree, prune_data) < dtree.check(pruned, prune_data):
            dirty = True
            tree = pruned

    if dirty:
        return prune(tree, prune_data)
    else:
        return tree


# split dataset into train set & pruning set
monk_train = []
monk_prune = []

for data in monks:
    train_data, prune_data = partition(data, 0.6)
    monk_train.append(train_data)
    monk_prune.append(prune_data)

# build & prune tree
for i in range(len(monk_train)):
    print('MONK -', i+1, ':', end=' ')

    # build
    t = dtree.buildTree(monk_train[i], m.attributes)

    print('%.8f' % dtree.check(t, monk_tests[i]), end=' ')

    # prune
    t = prune(t, monk_prune[i])

    print('%.8f' % dtree.check(t, monk_tests[i]))

print()

# assign 7
print('**** Assign 7 : pruning with different fractions ****')

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
expr_num = 100

total_result = []

print('Doing tests ...')

for i in range(len(monk_train)):

    monk_result = []

    for f in fractions:

        fraction_result = []

        for __ in range(expr_num):
            # split
            train_data, prune_data = partition(monks[i], f)

            # build
            t = dtree.buildTree(train_data, m.attributes)

            # prune
            t = prune(t, prune_data)

            # save
            fraction_result.append(1 - dtree.check(t, monk_tests[i]))

        monk_result.append(fraction_result)

    total_result.append(monk_result)

print('Making graph ...')

import matplotlib.pyplot as plt
import statistics as stat

for monk_result in total_result:

    avg = []
    stddev = []

    plt.figure(1)

    plt.subplot(211)
    plt.ylabel('Error Rate')
    plt.xlabel('Partition Fraction')

    for x, y in zip(fractions, monk_result):
        plt.scatter([x] * len(y), y, s=2)

    plt.subplot(212)
    plt.ylabel('Average Error Rate')
    plt.xlabel('Partition Fraction')

    for fraction_result in monk_result:
        avg.append(stat.mean(fraction_result))
        stddev.append(stat.stdev(fraction_result))

    plt.errorbar(fractions, avg, stddev, marker='o')
    for x, y, err in zip(fractions, avg, stddev):
        plt.annotate('%.3f' % y, xy=(x, y))
        plt.annotate('%.3f' % err, xy=(x, y-err), color='blue', size=8)
    plt.show()




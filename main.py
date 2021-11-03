
# coding: utf-8

# In[1]:

import os
import subprocess

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from itertools import groupby
import math
import random
import glob, os
from condition import Condition
from manual_cases import manual_cases
from settings import max_value, min_value, y_threshold, intersection_threshold, min_conditions, results_count, expressiveness_rate
import pygal
from pygal.style import Style
import shutil
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
from collections import Counter


# In[2]:


def get_survey_data():
    if os.path.exists("ps_india_prepared.csv"):
        print("-- ps_india_prepared.csv found")
        return pd.read_csv("ps_india_prepared.csv", sep=';')
    return df

def visualize_tree(tree, tree_id, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    print("visualizing tree " + str(tree_id))
    class_names = [str(class_name) for class_name in tree.classes_]
    with open("images/dt" + str(tree_id) + ".dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names, class_names=class_names)

    command = ["dot", "-Tpng", "images/dt" + str(tree_id) + ".dot", "-o", "images/dt" + str(tree_id) + ".png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

def visualize_data(data, condition_x, condition_y, feature_category, filename):
    style = Style(colors=('#00cc00', 'red', 'blue', 'blue'))
    xy_chart = pygal.XY(stroke=False, style=style)
    high_points = [(row[condition_x._feature], row[condition_y._feature]) for i, row in data.iterrows() if row[feature_category] == True]
    low_points = [(row[condition_x._feature], row[condition_y._feature]) for i, row in data.iterrows() if row[feature_category] == False]
    xy_chart.x_title = str(condition_x)
    xy_chart.y_title = str(condition_y)
    high_points_counter = Counter(high_points)
    low_points_counter = Counter(low_points)

    xy_chart.add('High', high_points,
                 formatter=lambda x: '[%.2f,%.2f] (%s)' % (x[0], x[1], high_points_counter.get(tuple(x))))
    xy_chart.add('Low', low_points, dots_size=1,
                 formatter=lambda x: '[%.2f,%.2f] (%s)' % (x[0], x[1], low_points_counter.get(tuple(x))))
    xy_chart.add(condition_x._feature + ' = ' + str(condition_x._threshold),
                 [(condition_x._threshold, min_value), (condition_x._threshold, max_value)],
                 stroke=True)
    xy_chart.add(condition_y._feature + ' = ' + str(condition_y._threshold),
                 [(min_value, condition_y._threshold), (max_value, condition_y._threshold)],
                 stroke=True)
    xy_chart.render_to_file('images/' + filename + '.svg')

def visualize_3data(data, condition_x, condition_y, condition_z, feature_category, filename):

    high_perf = go.Scatter3d(
        x=[row[condition_x._feature] for i, row in data.iterrows() if row[feature_category] == True],
        y=[row[condition_y._feature] for i, row in data.iterrows() if row[feature_category] == True],
        z=[row[condition_z._feature] for i, row in data.iterrows() if row[feature_category] == True],
        mode='markers',
        marker=dict(
            size=12,
            color='rgb(208, 81, 81)',  # set color to an array/list of desired values
            opacity=0.9
        )
    )

    low_perf = go.Scatter3d(
        x=[row[condition_x._feature] for i, row in data.iterrows() if row[feature_category] == False],
        y=[row[condition_y._feature] for i, row in data.iterrows() if row[feature_category] == False],
        z=[row[condition_z._feature] for i, row in data.iterrows() if row[feature_category] == False],
        mode='markers',
        marker=dict(
            size=12,
            color='rgb(81, 81, 208)',  # set color to an array/list of desired values
            opacity=0.8
        )
    )

    data = [high_perf, low_perf]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig,filename='images/3d' + filename + '.html')
    #py.iplot(fig, filename='images/3d' + filename + '.html')

def rate(coverage, precision, parsimony):
    return coverage * precision * pow(parsimony, 1 / expressiveness_rate)

def tree_to_cases(tree, tree_id, feature_names, coverage):
    """Produce decision tree as table with features as rows and resulting cases as columns

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    n_node_samples = tree.tree_.n_node_samples
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    impurity = tree.tree_.impurity
    classes = tree.classes_

    def recurse(node, conditions):
        node_precision = 1 - impurity[node]
        node_parsimony = np.prod([condition.expressiveness() for condition in conditions])

        if (threshold[node] != -2):
            left_result = recurse(left[node], conditions + [Condition(features[node], "<", threshold[node])])
            left_case = left_result[0]
            right_result = recurse(right[node], conditions + [Condition(features[node], ">", threshold[node])])
            right_case = right_result[0]

            samples = [left_case['samples'][0] + right_case['samples'][0], left_case['samples'][1] + right_case['samples'][1]]
            category = samples[1] > samples[0]
            samples_total = samples[0] + samples[1]

            if left_case['category'] == category:
                exception = left_case['exception']
            else:
                exception = left_case

            if right_case['category'] == category:
                exception = right_case['exception'] if right_case['exception']['own_rate'] > exception['own_rate'] else exception
            else:
                exception = right_case if right_case['own_rate'] > exception['own_rate'] else exception

            own_samples = samples[category]
            own_coverage = coverage(own_samples)
            own_rate = rate(own_coverage, node_precision, node_parsimony)

            if (exception['own_rate'] >= 0):
                exception_samples = exception['samples'][not category]
                exception_parsimony = exception['own_parsimony']
                exception_precision = exception['own_precision']

                pair_coverage = coverage(own_samples + exception_samples)
                pair_precision = (exception_precision + node_precision) / 2
                pair_parsimony = exception_parsimony * node_parsimony

                pair_rate = rate(pair_coverage, pair_precision, pair_parsimony)
            else:
                pair_rate = own_rate

            case = {
                'label': ('High' if category else 'Low'),
                'category': category,
                'count': samples_total,
                'conditions': conditions,
                'tree_id': tree_id,
                'samples': samples,
                'own_rate': own_rate,
                'own_parsimony': node_parsimony,
                'own_precision': node_precision,
                'pair_rate': pair_rate,
                'exception': exception
            }
            return [case] + left_result + right_result
        else:
            target = value[node][0]
            best_index = np.argmax(target)
            best_category = classes[best_index]
            node_samples = n_node_samples[node]
            node_coverage = coverage(node_samples)
            case = {
                'label': ('High' if best_category else 'Low'),
                'category': best_category,
                'count': node_samples,
                'conditions': conditions,
                'tree_id': tree_id,
                'samples': [0.0, node_samples] if best_category else [node_samples, 0.0],
                'own_rate': rate(node_coverage, node_precision, node_parsimony),
                'own_parsimony': node_parsimony,
                'own_precision': node_precision,
                'pair_rate': rate(node_coverage, node_precision, node_parsimony),
                'exception': {
                    'own_rate': -1.0,
                    'pair_rate': -1.0,
                    'samples': [0.0, 0.0]
                }
            }
            return [case]

    return recurse(0, [])


# In[6]:


def cases_to_table(cases, feature_names):
    def cell(feature, case):
        feature_conditions = [str(condition) for condition in case['conditions'] if condition._feature == feature]
        return ' && '.join(feature_conditions) if len(feature_conditions) > 0 else '-'

    features_used = [feature for feature in feature_names if any(cell(feature, case) != '-' for case in cases)]

    lines = ['feature;' + ';'.join(('T' + str(case['tree_id']) + ':' + case['label'] + ' (' + str(case['count']) + ')' for case in cases))]

    for feature in features_used:
        lines.append(feature + ';' + ';'.join(cell(feature, case) for case in cases))

    return lines

def case_to_text(case):
    exception = case['exception']
    conditions = case['significant_conditions'] if 'significant_conditions' in case else case['conditions']
    text = 'From tree ' + str(case['tree_id']) + \
           ': If ' + ' and '.join(str(condition) for condition in conditions) + ' then value is ' + case['label']

    if exception['own_rate'] >= 0:
        text += '\nexcept for cases ' + \
                ' and '.join(str(condition) for condition in exception['conditions'][len(case['conditions']):]) + \
                ' where value is ' + exception['label']
    return text

def filter_conditions(conditions):
    significant = []
    for condition in conditions:
        if any([other._feature == condition._feature \
                and other._comparison == condition._comparison \
                and other.residual_points() < condition.residual_points() for other in conditions]):
            continue
        significant.append(condition)

    return significant


def feature_intersection(conditions_a, conditions_b, feature):
    lower_bound = min_value
    upper_bound = max_value
    for condition in (conditions_a + conditions_b):
        if condition._feature == feature:
            if (condition._comparison == '>'):
                lower_bound = max(lower_bound, condition._threshold)
            else:
                upper_bound = min(upper_bound, condition._threshold)

    return max(upper_bound - lower_bound, 0)

def volume_intersection(conditions_a, conditions_b, features):
    return np.prod([feature_intersection(conditions_a, conditions_b, feature) for feature in features])

def volume_rate(conditions_a, conditions_b, features):
    volume_a = volume_intersection(conditions_a, [], features)
    volume_b = volume_intersection(conditions_b, [], features)
    intersection = 1.0 * volume_intersection(conditions_a, conditions_b, features)
    return max(intersection / volume_a, intersection / volume_b)

def filter_cases(cases, features, max_count):
    for case in cases:
        if case['own_rate'] > case['pair_rate']:
            case['pair_rate'] = case['own_rate']
            case['exception'] = {
                'own_rate': -1.0,
                'pair_rate': -1.0,
                'samples': [0.0, 0.0]
            }

    cases.sort(key=lambda case: -case['pair_rate'])

    result = []

    for case in cases:
        if len(result) >= max_count:
            break
        case['conditions'] = filter_conditions(case['conditions'])
        if case['exception']['own_rate'] >= 0:
            case['exception']['conditions'] = filter_conditions(case['exception']['conditions'])

        valid_length = len(case['conditions']) >= min_conditions
        max_intersect = max([volume_rate(existing['conditions'], case['conditions'], features)
                             for existing in result]) if len(result) > 0 else 0
        unique = max_intersect < intersection_threshold
        if valid_length and unique:
            result.append(case)

    return result


def predict_by_cases(cases, sample, default = lambda: random.choice([True, False])):
    def check_case(case):
        conditions = case['significant_conditions'] if 'significant_conditions' in case else case['conditions']
        return all(condition.apply(sample) for condition in conditions)

    for case in cases:
        exception = case['exception']
        if exception['pair_rate'] >= 0 and check_case(exception):
            return exception['category']
        if check_case(case):
            return case['category']

    return default()


def evaluate_case(case, X, Y):
    cases_predictions = [predict_by_cases([case], row, lambda: None) for index, row in X.iterrows()]
    results = [Y[index] == predicted_item for index, predicted_item in enumerate(cases_predictions) if predicted_item is not None]
    matching_results = [result for result in results if result]
    coverage = round(1.0 *  len(results) / len(X), 2)
    precision = round(1.0 * len(matching_results) / len(results), 2) if len(results) > 0 else 0.0
    return coverage, precision
#
# def compare(actual, predicted):
#     if predicted is None:
#         return None
#     if actual:
#         return 'tp' if actual == predicted else 'fn'
#     else:
#         return 'tn' if actual == predicted else 'fp'

# def f1(actual, predicted):
#     results = [compare(actual[index], predicted_item) for index, predicted_item in enumerate(predicted)]
#     classes = {key: len(list(group)) for key, group in groupby(sorted(results))}
#     precision = classes.get('tp', 0) / (classes.get('tp', 0) + classes.get('fp', 0) + 0.0000001)
#     recall = classes.get('tp', 0) / (classes.get('tp', 0) + classes.get('fn', 0) + 0.000001)
#     f1 = 2  * (precision * recall) / (precision + recall + 0.000001)
#     return round(f1, 3)
#
# for f in glob.glob("dt*.png") + glob.glob("dt*.dot"):
#     os.remove(f)
shutil.rmtree('images/', True)
os.makedirs('images/')

df = get_survey_data()


print("* df.head()", df.head(), sep="\n", end="\n\n")
print("* df.tail()", df.tail(), sep="\n", end="\n\n")


features = list(df.columns[:4])
features.extend(df.columns[5:8])
print("* features:", features, sep="\n")

y_feature = "ORG_GROWTH"

train, test = train_test_split(df, test_size = 0.2)
print("Train set: " + str(len(train)) + " samples")
print("Test set: " + str(len(test)) + " samples")

Y_train = train[y_feature] > y_threshold
X_train = train[features]
Y_test = test[y_feature] > y_threshold
X_test = test[features]

train_count = len(train)
train = pd.concat([X_train, Y_train], axis=1)

forest = RandomForestClassifier(n_estimators=100, min_samples_split=20, criterion='entropy')
forest.fit(X_train, Y_train)

Y_train = Y_train.values.tolist()
Y_test = Y_test.values.tolist()


def coverage(samples):
    return 100.0 * samples * math.log(samples) / (train_count * math.log(train_count))

cases = [case for tree_id, dt in enumerate(forest.estimators_) for case in tree_to_cases(dt, tree_id, features, coverage)]
cases = filter_cases(cases, features, results_count)
trees_to_visualize = {case['tree_id'] for case in cases}
for tree_id, dt in enumerate(forest.estimators_):
    if tree_id in trees_to_visualize:
        visualize_tree(dt, tree_id, features)

print('\n')
# # assess performance of train set
# cases_predictions = [predict_by_cases(cases, row) for index, row in X_train.iterrows()]
# forest_predictions = forest.predict(X_train)
# print("cases F1 on train set: " + str(f1(Y_train, cases_predictions)))
# print("forest F1 on train set: " + str(f1(Y_train, forest_predictions)))
#
# # assess performance of test set
# cases_predictions = [predict_by_cases(cases, row) for index, row in X_test.iterrows()]
# forest_predictions = forest.predict(X_test)
# print("cases F1 on test set: " + str(f1(Y_test, cases_predictions)))
# print("forest F1 on test set: " + str(f1(Y_test, forest_predictions)))
# manual_cases_predictions = [predict_by_cases(manual_cases, row) for index, row in X_test.iterrows()]
# print("manual cases F1 on test set: " + str(f1(Y_test, manual_cases_predictions)))
# print('\n')

#write cases as CSV file
# lines = cases_to_table(cases, features)
# with open("dt.csv", 'w') as f:
#     for line in lines:
#         print(line, file=f)

for case in cases:
    text = case_to_text(case)

    if (len(case['conditions']) == 2):
        visualize_data(train, case['conditions'][0], case['conditions'][1], y_feature, text.split('\n')[0])
    if (len(case['conditions']) > 2):
        visualize_3data(train, case['conditions'][0], case['conditions'][1], case['conditions'][2], y_feature, text.split('\n')[0])

    train_matching, train_precision = evaluate_case(case, X_train, Y_train)
    test_matching, test_precision = evaluate_case(case, X_test, Y_test)
    text += '\n'
    text += '      | {0: ^16} | {1: ^16} | {2: ^16}\n'.format('precision', 'coverage', 'rate')
    text += 'Train | {0: ^16} | {1: ^16} | {2: ^16}\n'.format(train_precision, train_matching, round(case['pair_rate'], 2))
    text += 'Test  | {0: ^16} | {1: ^16} |\n'.format(test_precision, test_matching)
    print(text + '\n')


# In[ ]:





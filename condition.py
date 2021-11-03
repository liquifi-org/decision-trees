import math
from settings import max_value, min_value, parsimony_rate, expressiveness_rate

class Condition(object):
    def __init__(self, feature, comparison, threshold):
        self._comparison = comparison
        self._threshold = threshold
        self._feature = feature

    def __str__(self):
        category = 'high' if self._comparison == '>' else 'low'
        value = '(' + self._comparison + " " + str(round(self._threshold, 2)) + ')'
        return self._feature + " is " + category + value

    def apply(self, sample):
        value = sample[self._feature]
        return value > self._threshold if self._comparison == '>' else value <= self._threshold

    def residual_points(self):
        return max_value - self._threshold if self._comparison == '>' else self._threshold - min_value

    def expressiveness(self):
        return 1 - math.log10(self.residual_points() * parsimony_rate * expressiveness_rate + 1)


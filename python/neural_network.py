#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Sean Kirmani <sean@kirmani.io>
#
# Distributed under terms of the MIT license.
"""TODO(Sean Kirmani): DO NOT SUBMIT without one-line documentation for test

TODO(Sean Kirmani): DO NOT SUBMIT without a detailed description of test.
"""
class NeuralNetwork(object):
  pass

class Goal(object):
  def __init__(self):
    self.count = 0

  def GoalMet(self):
    return self.count == 50

  def UpdateGoalStatus(self):
    self.count += 1

class IterationGoal(object):
  def __init__(self, iterations):
    self._count = 0
    self._iterations = iterations

  def GoalMet(self):
    return self._count >= self._iterations

  def UpdateGoalStatus(self, variables):
    self._count += 1

class ErrorReachedGoal(object):
  def __init__(self, error_to_reach):
    self._error_to_reach = error_to_reach
    self._error = float('inf')
    self._count = 0

  def GoalMet(self):
    goal_met = self._error <= self._error_to_reach
    if goal_met:
      print("Goal met: current error %s is less than target error %s after %s "
          "iterations" % (self._error, self._error_to_reach, self._count))
    return goal_met

  def UpdateGoalStatus(self, variables):
    self._error = variables['error']
    self._count += 1

class DeltaFromTargetGoal(object):
  def __init__(self, delta, targets):
    self._delta = delta
    self._targets = targets
    self._count = 0
    self._outputs = [1000] * len(self._targets)

  def GoalMet(self):
    goal_met = True
    for index in range(len(self._targets)):
      if abs(self._targets[index] - self._outputs[index]) > self._delta:
        goal_met = False
    if goal_met:
      print("Goal met: current output %s agrees with target output %s within a "
          "degree of uncertainty %s after %s iterations"
          % (self._outputs, self._targets, self._delta, self._count))
    return goal_met

  def UpdateGoalStatus(self, variables):
    self._outputs = variables['outputs']
    self._count += 1

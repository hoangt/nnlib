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
import argparse
import feedfoward_neural_network as ffnn
import neural_network
import os
import random
import re
import sys
import time
import traceback

class FullyRecurrentNeuralNetwork(neural_network.NeuralNetwork):
  def __init__(self, ni, nh, no):
    self._ni = ni # +1 for bias node
    self._nh = nh
    self._no = no
    self._ffnn = ffnn.FeedfowardNeuralNetwork(ni, nh, no)
    self._inputs = None

  def test(self, inputs, targets):
    print(self._inputs, '->', self._ffnn.update(self._inputs), '->', targets)

  def train(self, inputs, targets, goal, N=0.4, M=0.1):
    variables = {}
    self._inputs = inputs
    variables['outputs'] = self._ffnn.update(inputs)
    variables['error'] = 0.0
    variables['error'] = variables['error'] + self._ffnn.backPropagate(targets, N, M)
    self._ffnn = ffnn.FeedfowardNeuralNetwork(self._ni + self._no, self._nh, self._no)
    goal.UpdateGoalStatus(variables)
    while not goal.GoalMet():
      new_inputs = inputs + variables['outputs']
      self._inputs = new_inputs
      variables['outputs'] = self._ffnn.update(new_inputs)
      variables['error'] = self._ffnn.backPropagate(targets, N, M)
      goal.UpdateGoalStatus(variables)

def main():
  global args
  random.seed()
  inputs = [random.random()]
  targets = [0.5]
  pat = [[inputs, targets]]
  ni = 1
  nh = [2, 2]
  no = 1

  num_tests = 1000

  ffnn_wins = 0
  frnn_wins = 0
  for i in range(num_tests):
    print("-----MLP-----")
    goal = neural_network.DeltaFromTargetGoal(0.01, targets)
    n = ffnn.FeedfowardNeuralNetwork(ni, nh, no)
    n.train(pat, goal)
    n.test(pat)
    ffnn_iters = goal._count

    print("-----RECURRENT-----")
    goal = neural_network.DeltaFromTargetGoal(0.01, targets)
    n = FullyRecurrentNeuralNetwork(ni, nh, no)
    n.train(inputs, targets, goal)
    n.test(inputs, targets)
    frnn_iters = goal._count

    if (ffnn_iters < frnn_iters):
      ffnn_wins += 1
    else:
      frnn_wins += 1
    print("FFNN: %s, FRNN: %s" % (ffnn_wins, frnn_wins))

if __name__ == '__main__':
  try:
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbose', action='store_true', default=False, \
        help='verbose output')
    parser.add_argument('-d','--debug', action='store_true', default=False, \
        help='debug output')
    args = parser.parse_args()
    # if len(args) < 1:
    #   parser.error('missing argument')
    if args.verbose: print(time.asctime())
    main()
    if args.verbose: print(time.asctime())
    if args.verbose: print('TOTAL TIME IN MINUTES:')
    if args.verbose: print(time.time() - start_time) / 60.0
    sys.exit(0)
  except KeyboardInterrupt, e: # Ctrl-C
    raise e
  except SystemExit, e: # sys.exit()
    raise e
  except Exception, e:
    print('ERROR, UNEXPECTED EXCEPTION')
    print(str(e))
    traceback.print_exc()
    os._exit(1)

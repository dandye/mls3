#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nn
import numpy as np


def test_logistic():
    x = np.arange(-2, 2, 0.5)
    print(x)
    expected = [0.12, 0.18, 0.27, 0.38, 0.5, 0.62, 0.73, 0.82]
    out = nn.logistic(x)
    print(out, expected)
    close = np.isclose(out, expected, atol=0.01)  # very approximate
    print(close)
    assert np.all(close)


def test_relu():
    x = np.arange(-2, 2, 0.5)
    print(x)
    expected = [0, 0, 0, 0, 0, 0.5, 1, 1.5]

    out = nn.relu(x)
    print(out, expected)
    assert np.all(out == expected)


def test_combination():
    w = np.array([0, 1, 0, 1])
    x = np.arange(20).reshape((5, 4))
    c = nn.Combiner(w)
    # output will just add second and forth columns
    expected = np.array([4, 12, 20, 28, 36])
    out = c(x)
    print(out, expected)

    assert np.all(out == expected)


def test_backprop():
    # backprop of relu and combinations are simple
    # sigmoid is harder, so we'll do that
    # for these inputs, take a look at
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    x = np.array([1.1059])
    out = nn.logistic(x)
    assert np.all(np.isclose(out, [0.75], atol=0.01))
    # normally we would use the output to calculate error
    # let's just assume it came out as below
    # this trick is accomplished by making an object
    # logistic and a __call__ function on it
    # note: neural networks must store their
    # inputs to perform backpropagation
    reverse = nn.logistic.backprop(0)
    expected = np.array([0.187])

    print(reverse, expected)
    assert np.all(np.isclose(expected, reverse, atol=0.01))

    # now, if we had a combiner, we'd pass these inputs
    # back to the combiner to adjust its weights
    # for now we can test that it gave us the right value

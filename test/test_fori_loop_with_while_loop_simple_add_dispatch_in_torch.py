import os
import unittest
from typing import Callable, Dict, List

import torch
import torch_xla
# We need to import the underlying implementation function to register with the dispatcher
import torch_xla.experimental.fori_loop
from torch._higher_order_ops.while_loop import while_loop
import torch_xla.core.xla_model as xm
import torch_xla.core.xla_builder as xb


def _fake_while_loop(cond_fn, body_fn, operands):
  while cond_fn(operands):
    operands = body_fn(operands)
  return operands


class WhileLoopTest(unittest.TestCase):

  def test_while_loop_tpu(self):

    device = xm.xla_device()

    def cond_fn(x):
      limit_value = torch.ones(1, dtype=torch.int32, device=device)
      return x[0] >= limit_value[0]

    def body_fn(x):
      return (torch.sub(x[0], 1),)

    xi = torch.tensor([5], dtype=torch.int32, device=device)
    res = while_loop(cond_fn, body_fn, (xi,))
    expected = _fake_while_loop(cond_fn, body_fn, xi)
    self.assertEqual(expected, res)

  def test_while_loop_tpu_addition(self):

    device = xm.xla_device()

    def cond_fn(x):
      limit_value = torch.ones(1, dtype=torch.int32, device=device)
      limit_value[0] = 10
      return x[0] <= limit_value[0]

    def body_fn(x):
      return (torch.add(x[0], 1),)

    xi = torch.tensor([0], dtype=torch.int32, device=device)
    res = while_loop(cond_fn, body_fn, (xi,))
    expected = _fake_while_loop(cond_fn, body_fn, xi)
    self.assertEqual(expected, res)

if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)

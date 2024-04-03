import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_builder as xb
import torch_xla.core.xla_model as xm
import torch_xla.utils.utils as xu
import torch_xla.core.xla_op_registry as xor

from torch._C import DispatchKey
from torch._ops import HigherOrderOperator
import torch._higher_order_ops.while_loop
from torch._higher_order_ops.while_loop import while_loop_op


# TODO(@manfei): delete one_value?
def fori_loop(lower, upper, body_fun, one_value, init_val, *input_value):

  device = xm.xla_device()

  def cond_fn(upper, lower, x, input_value):
    return lower[0] < upper[0]

  # one_value, init_val, l_in_i
  def body_fn(upper, lower, x, *input_value):
    # one_value = torch.ones(1, dtype=torch.int32, device=device)
    
    # ---
    result = body_fun(one_value, x, *input_value)
    if type(result) is tuple:
      return_list = list(body_fun(one_value, x, *input_value))
      # [body_fun_result]
      return_list.insert(0, lower) # lower
      # [lower, body_fun_result]
      return_list.insert(0, torch.sub(upper, one_value)) # upper
      # [upper, lower, body_fun_result]
      weight = torch.ones([20, 10], dtype=torch.float32, device=device) # f32[20,10]
      # return_list.append(weight)
      return_list.insert(-1, weight)
      # [upper, lower, body_fun_result, weight]
      # final_one = torch.tensor(1, dtype=torch.int64, device=device) # s64[]
      # return_list.append(final_one)
      # [upper, lower, body_fun_result, weight, final_one]
    else:
      return torch.sub(upper, one_value), lower, body_fun(one_value, x, input_value)
    # ---

    # return torch.sub(upper, one_value), lower, body_fun(one_value, x, input_value)
    return return_list

  res = while_loop(cond_fn, body_fn, (lower, upper, init_val, *input_value))
  return res


@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs=None):
  # TODO(@manfei): PyTorch require carried_inputs to be list/tuple, PyTorch/XLA _xla_while_loop only accept *operands, *operands would tuple items again: (a, '')
  # cond_fn&body_fn: callable
  # carried_inputs: (Tuple of possibly nested dict/list/tuple of tensors)
  if additional_inputs is None:
    additional_inputs = tuple()
  return _xla_while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs=additional_inputs)


def _xla_while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs):
  # untuple carried_inputs from while_loop
  carried_inputs = carried_inputs[0]
  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    #TODO(@manfei) type = carried_input.type
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=torch.int32).to(device))
  fake_carried_inputs = tuple(fake_carried_inputs)

  # trans fake_carried_inputs from list(tensor) to list(xla::op)
  kwargs = {}
  if type(fake_carried_inputs) is tuple:
    shapes = xb.tensor_shape(fake_carried_inputs)
  else:
    shapes = xb.tensor_shape((fake_carried_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)

  # generate cond_fn xlacomputation
  cond_result = cond_fn(*fake_carried_inputs)
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")
  cond_ctx.buildforiloop([cond_result], list(fake_carried_inputs[2:]))
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  # generate body_fn xlacomputation
  body_result = body_fn(*fake_carried_inputs)
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  body_ctx.buildforiloop(list(body_result), [])
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # generate while xlacomputation
  input_tuple = xb.Op.tuple(tuple(params))
  w = xb.mkop(
      'While', (input_tuple.op,),
      condition_computation=cond_computation,
      body_computation=body_computation)
  name = 'fori_loop_ed_torch_func'
  computation = w.build(name)
  hlo_print = xb.get_computation_hlo(computation)
  print("while computation: !!!!!!!!!")
  print(hlo_print)

  # gain final result with generated while xlacomputation
  result = torch_xla._XLAC._xla_user_computation('xla::_op_test_while',
                                                 (carried_inputs),
                                                 computation)

  return result

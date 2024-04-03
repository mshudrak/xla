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

  # a = torch.tensor(1, dtype=torch.int32, device=device)
  # b = torch.tensor(1, dtype=torch.int32, device=device)
  # c = torch.tensor(1, dtype=torch.int32, device=device)

  def cond_fn(upper, lower, x, *input_value, a, b, c):
    return lower[0] < upper[0]

  # one_value, init_val, l_in_i
  def body_fn(upper, lower, x, *input_value, a, b, c):
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
      return_list.append(weight)
      final_one = torch.ones([10], dtype=torch.int32, device=device)
      return_list.append(final_one)
      # [upper, lower, body_fun_result, weight]
      # final_one = torch.tensor(1, dtype=torch.int64, device=device) # s64[]
      # return_list.append(final_one)
      # [upper, lower, body_fun_result, weight, final_one]
    else:
      return torch.sub(upper, one_value), lower, body_fun(one_value, x, input_value)
    # ---
    # return torch.sub(upper, one_value), lower, body_fun(one_value, x, input_value)
    return return_list

  # s32[1], f32[20], /*index=5*/f32[20,10],
  a = torch.ones(1, dtype=torch.int32, device=device) # s32[1]
  b = torch.ones(20, dtype=torch.float32, device=device) # f32[20]
  c = torch.ones([20, 10], dtype=torch.float32, device=device) # f32[20,10]
  res = while_loop(cond_fn, body_fn, (lower, upper, init_val, *input_value, a, b, c)) # , additional_inputs=(a, b, c))
  return res


@while_loop_op.py_impl(DispatchKey.XLA)
def while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs=None): # a=None, b=None, c=None, 
  # TODO(@manfei): PyTorch require carried_inputs to be list/tuple, PyTorch/XLA _xla_while_loop only accept *operands, *operands would tuple items again: (a, '')
  # cond_fn&body_fn: callable
  # carried_inputs: (Tuple of possibly nested dict/list/tuple of tensors)
  if additional_inputs is None:
    additional_inputs = tuple()
  return _xla_while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs=additional_inputs) #  a=a, b=b, c=c,


def _xla_while_loop(cond_fn, body_fn, *carried_inputs, additional_inputs): # a, b, c, 
  print("carried_inputs: ", carried_inputs)
  # untuple carried_inputs from while_loop
  carried_inputs = carried_inputs[0]
  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    #TODO(@manfei) type = carried_input.type
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.type()).to(device))
  fake_carried_inputs = tuple(fake_carried_inputs)
  print("fake_carried_inputs: ", fake_carried_inputs)
  print("additional_inputs: ", additional_inputs)

  # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
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
  # tmp_params = params
  # (s32[1], s32[1], s32[1], s32[10], s32[1], /*index=5*/s32[20], s32[20,10]) 
  # params = params[:-4]+
  import pdb; pdb.set_trace()
  params += [params.pop(-1)]
  print("params: ", params)
  # add additional_inputs for init of xla::while
  # if type(additional_inputs) is tuple:
  #   additional_shapes = xb.tensor_shape(additional_inputs)
  # else:
  #   additional_shapes = xb.tensor_shape((additional_inputs))
  
  # for additional_shape in additional_shapes:
  #   p = xb.mkparam(builder, len(params), additional_shape)
  #   params.append(p)

  # keep the last item is s32[10]
  # init     (s32[1], s32[1], s32[1], s32[10], s32[1], /*index=5*/f32[20], f32[20,10])
  # expected (s32[1], s32[1], s32[1], s32[1], f32[20], /*index=5*/f32[20,10], s32[10])


  # generate cond_fn xlacomputation
  # TODO(@manfei): specify which element is for which argument like a,b,c
  cond_result = cond_fn(*fake_carried_inputs[:-3], a=fake_carried_inputs[-3], b=fake_carried_inputs[-2], c=fake_carried_inputs[-1]) # , a=additional_inputs[0], b=additional_inputs[1], c=additional_inputs[2])
  cond_ctx = torch_xla._XLAC.lowering.LoweringContext()
  cond_ctx.set_name_string("condctx")
  additional_inputs_list = list(fake_carried_inputs[2:])
  for i in range(len(additional_inputs)):
    additional_inputs_list.append(additional_inputs[0])
  cond_ctx.buildforiloop([cond_result], additional_inputs_list) # list(fake_carried_inputs[2:]), )
  cond_hlo = cond_ctx.hlo()
  cond_computation = xb.computation_from_module_proto("condcomputation",
                                                      cond_hlo)
  cond_hlo_print = xb.get_computation_hlo(cond_computation)
  print("cond computation: !!!!!!!!!")
  print(cond_hlo_print)

  # generate body_fn xlacomputation
  # body_result = body_fn(*fake_carried_inputs) # , a=additional_inputs[0], b=additional_inputs[1], c=additional_inputs[2])
  body_result = body_fn(*fake_carried_inputs[:-3], a=fake_carried_inputs[-3], b=fake_carried_inputs[-2], c=fake_carried_inputs[-1])
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
                                                 (carried_inputs), # , a, b, c
                                                 computation)

  return result

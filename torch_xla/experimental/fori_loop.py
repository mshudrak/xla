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

  def cond_fn(one_value, upper, lower, x, *input_value, b, c, output_value):
    return lower[0] <= upper[0]

  # one_value, init_val, l_in_i
  # a = torch.ones(1, dtype=torch.int32, device=device) # s32[1]
  # b = torch.ones(20, dtype=torch.float32, device=device) # f32[20]
  # c = torch.ones([20, 10], dtype=torch.float32, device=device) # f32[20,10]
  # output_value = torch.ones([20], dtype=torch.float32, device=device) # f32[20]
  #           s32[1]
  #   s32[1], s32[1], s32[1], s32[1], f32[20], f32[20,10], f32[10], f32[20])) 
  # def body_fn(upper, lower, x, *input_value, a, b, c, output_value):
  def body_fn(one_value, upper, lower, x, *input_value, b, c, output_value):
    # one_value, upper, lower, x_i, bias, weight, l_in_i
    # init_one_value = torch.ones(1, dtype=torch.int32, device=device)
    
    # ---
    result = () # body_fun(one_value, x, *input_value)
    if type(result) is tuple:
      # one_value, torch.add(one_value, init_val), l_out
      # return_list = list(body_fun(one_value, x, *input_value))
      # one_value_new, torch_add_res, l_out = body_fun(one_value, x, *input_value)
      # one_value, torch.add(one_value, init_val), l_out
      # one_value_new = one_value
      # l_out = body_fun(*input_value)
      # torch_add_res = torch.add(one_value, x)

      return_list = []
      one_value_new = one_value
      return_list.append(one_value_new)
      # hard-code change body xlacomputation to meet requirement
      # [body_fun_result]
      # return_list.insert(1, lower) # lower
      return_list.append(lower) # lower
      # [lower, body_fun_result]
      # return_list.insert(1, torch.sub(upper, one_value)) # upper
      return_list.append(torch.sub(upper, one_value_new)) # one_value)) # upper
      torch_add_res = torch.add(one_value, x)
      return_list.append(torch_add_res)
      # [upper, lower, body_fun_result]
      # TODO(@manfei): should initialize bias with torch.nn.linear's real bias, currently we use placeholder for all bias are 1
      bias = torch.ones([20], dtype=torch.float32, device=device) # f32[10] #???
      # return_list.insert(-1, bias)
      return_list.append(bias)
      # TODO(@manfei): should initialize weight with torch.nn.linear's real weight, currently we use placeholder for all weight are 1
      weight = torch.ones([20, 10], dtype=torch.float32, device=device) # f32[20,10] # ???
      # return_list.append(weight)
      # return_list.insert(-1, weight)
      return_list.append(weight)
      # return_list.append(l_out) # f32[20]
      l_in_i_plus_1 = torch.ones([10], dtype=torch.float32, device=device) # f32[10]
      # return_list.insert(-1, l_in_i_plus_1)
      return_list.append(l_in_i_plus_1) # f32[10] # input_value replace l_in_i_plus_1?
      l_out = body_fun(*input_value)
      return_list.append(l_out) # f32[20]
# [one_value_new, lower, torch.sub(upper, one_value_new), torch_add_res, bias, weight, l_in_0, l_out]
      # return_list.append(l_out) # f32[20]
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
  a = torch.ones(1, dtype=torch.int32, device=device) # s32[1] # one_value? # ???
  b = torch.ones(20, dtype=torch.float32, device=device) # f32[20] # bias?
  c = torch.ones([20, 10], dtype=torch.float32, device=device) # f32[20,10] # weight?
  output_value = torch.ones([20], dtype=torch.float32, device=device) # f32[20]
  res = while_loop(cond_fn, body_fn, (one_value, lower, upper, init_val, *input_value, b, c, output_value)) # , additional_inputs=(a, b, c))
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
  # print("carried_inputs: ", carried_inputs)
  # untuple carried_inputs from while_loop
  carried_inputs = carried_inputs[0]
  # fake carried_inputs to split formal code
  fake_carried_inputs = []
  for carried_input in carried_inputs:
    device = carried_input.device
    # print("type carried_input: ", carried_input.dtype)
    # print("is torch.int32: ", carried_input.dtype==torch.int32)
    #TODO(@manfei) type = carried_input.type
    fake_carried_inputs.append(
        torch.randint(10, carried_input.size(),
                      dtype=carried_input.dtype).to(device))
  fake_carried_inputs = tuple(fake_carried_inputs)
  # print("fake_carried_inputs: ", fake_carried_inputs)
  # print("additional_inputs: ", additional_inputs)

  # # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
  # kwargs = {}
  # if type(carried_inputs) is tuple:
  #   shapes = xb.tensor_shape(carried_inputs)
  # else:
  #   shapes = xb.tensor_shape((carried_inputs))
  # builder = xb.create_builder('test_while')
  # params = []
  # for shape in shapes:
  #   p = xb.mkparam(builder, len(params), shape)
  #   params.append(p)

  # tmp_params = params
  # (s32[1], s32[1], s32[1], s32[10], s32[1], /*index=5*/s32[20], s32[20,10]) 
  # params = params[:-4]+
  # import pdb; pdb.set_trace()

  # params += [params.pop(-1)]
  # print("params: ", params)

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
  cond_result = cond_fn(one_value=fake_carried_inputs[0], *fake_carried_inputs[1:-3], b=fake_carried_inputs[-3], c=fake_carried_inputs[-2], output_value=fake_carried_inputs[-1]) # , a=additional_inputs[0], b=additional_inputs[1], c=additional_inputs[2])
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
  body_result = body_fn(one_value=fake_carried_inputs[0], *fake_carried_inputs[1:-3], b=fake_carried_inputs[-3], c=fake_carried_inputs[-2], output_value=fake_carried_inputs[-1])
  body_ctx = torch_xla._XLAC.lowering.LoweringContext()
  body_ctx.set_name_string("bodyctx")
  body_ctx.buildforiloop(list(body_result), [])
  body_hlo = body_ctx.hlo()
  body_computation = xb.computation_from_module_proto("bodycomputation",
                                                      body_hlo)
  body_hlo_print = xb.get_computation_hlo(body_computation)
  print("body computation: !!!!!!!!!")
  print(body_hlo_print)

  # reorder carried_inputs to meet generated xlacomputation
  tmp_carried_inputs_list = list(carried_inputs)
  # copy_tmp_carried_inputs_list = tmp_carried_inputs_list
  # print("carried_inputs: ", carried_inputs)
  # print("tmp_carried_inputs: ", tmp_carried_inputs)
  # print("tmp_carried_inputs[:3]: ", tmp_carried_inputs[:3])
  # print("type tmp_carried_inputs[:3]: ", type(tmp_carried_inputs[:3]))
  # carried_inputs = tmp_carried_inputs[:3]
  # carried_inputs.append(tmp_carried_inputs[-3:])
  tmp_save = tmp_carried_inputs_list[3]
  tmp_save2 = tmp_carried_inputs_list[-1]
  del tmp_carried_inputs_list[3]
  del tmp_carried_inputs_list[-1]
  tmp_carried_inputs_list.append(tmp_save)
  tmp_carried_inputs_list.append(tmp_save2)
  carried_inputs = tuple(tmp_carried_inputs_list)

  # trans fake_carried_inputs from list(tensor) to list(xla::op), which part could change init of xla::while
  kwargs = {}
  if type(carried_inputs) is tuple:
    shapes = xb.tensor_shape(carried_inputs)
  else:
    shapes = xb.tensor_shape((carried_inputs))
  builder = xb.create_builder('test_while')
  params = []
  for shape in shapes:
    p = xb.mkparam(builder, len(params), shape)
    params.append(p)
  # import pdb; pdb.set_trace()

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

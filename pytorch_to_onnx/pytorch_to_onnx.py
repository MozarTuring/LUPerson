# coding: utf-8
"""
本代码介绍如何将pytorch训练好的模型转换为onnx格式。主要分三步：
1、导入pytorch模型并加载训练好的参数.pth
2、导出onnx
3、优化onnx
下面以MobileNetV2Agender为例（实际使用时需要换成你自己的模型）
"""
import os

import torch
#import onnx
# from onnxsim import simplify  # pip install onnx-simplifier
import sys
cur_dir = __file__.rsplit(os.sep, 1)[0]
print(cur_dir)
sys.path.append(cur_dir)
from mobilenet_v2_agender import MobileNetV2Agender


def load_pth(net, pth_path):
    if not os.path.exists(pth_path):
        raise ValueError(f'cannot find pth file | {pth_path}')

    state_dict = torch.load(pth_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(state_dict, strict=True)
    return net


def net_to_onnx(net, onnx_path, input_shapes, input_names, output_names, dynamic_batch_size=False):
    inputs = tuple([torch.rand(x, device="cuda") for x in input_shapes])
    dynamic_axes = {k: {0: 'batch_size'} for k in input_names + output_names} if dynamic_batch_size else None
    os.makedirs(os.path.dirname(os.path.realpath(onnx_path)), exist_ok=True)
    torch.onnx.export(net, inputs, onnx_path, verbose=True, input_names=input_names, output_names=output_names,
                      opset_version=9, dynamic_axes=dynamic_axes)  # opset_version默认9，建议明确指定，以免新版本pytorch有更改

    print(f'pytorch to onnx successful | save | {onnx_path}')


def simplify_onnx(onnx_path, input_shapes, input_names, dynamic_batch_size=False):
    """
    （可能出现）低版本tensorrt转换torch.onnx.export导出的onnx报错，建议先优化onnx（例如conv和bn的融合等）
    simplify()会优化并check优化后的onnx结果是否与优化前一致
    无论优化前的onnx是否dynamic_batch_size，优化后的onnx保持一致
    如果dynamic_batch_size，check的时候需要明确指定输入尺寸
    """
    inputs = {name: shape for name, shape in zip(input_names, input_shapes)} if dynamic_batch_size else None
    print(f'\nsimplify | {onnx_path}')
    model_simp, check_ok = simplify(onnx_path, check_n=5, input_shapes=inputs, dynamic_input_shape=dynamic_batch_size)
    if check_ok:
        print('successful!')
        onnx_simp_path = os.path.splitext(onnx_path)[0] + '_simp.onnx'  # 默认将优化后的onnx保存在同一个目录
        onnx.save(model_simp, onnx_simp_path)
        print(f'save | {onnx_simp_path}')
    else:
        print('failed!')


if __name__ == '__main__':
    # 1、导入pytorch模型并加载训练好的参数.pth
    net = MobileNetV2Agender()  # 实际使用时需要换成你自己的模型
    pth_path = 'mobilenet_v2_agender_20210802.pth'  # 实际使用时需要换成你自己训练好的模型参数
    net = load_pth(net, pth_path)

    # 2、导出onnx
    onnx_path = os.path.splitext(pth_path)[0] + '.onnx'  # 指定onnx文件保存路径
    input_shapes = [(1, 3, 112, 96), ]  # 模型输入tensor尺寸，例如NCHW，支持多个输入
    input_names = ['nchw_rgb', ]  # 模型输入tensor的名字，顺序与input_shapes对应。建议起容易理解的名字，例如nchw_rgb, nchw_bgr
    output_names = ['gender', 'age']  # 模型输出的名字，建议起容易理解的名字
    dynamic_batch_size = True  # 如果True，input_shapes中batch_size不起作用，方便之后转成不同max_batch_size的tensorrt engine
    net_to_onnx(net, onnx_path, input_shapes, input_names, output_names, dynamic_batch_size)

    # 3、优化onnx
    # simplify_onnx(onnx_path, input_shapes, input_names, dynamic_batch_size)

    print('\nfinished!')

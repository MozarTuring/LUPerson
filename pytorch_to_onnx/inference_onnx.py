# coding: utf-8
"""
本代码介绍如何使用转换好的onnx模型做inference
"""
import os
import time

import numpy as np
import cv2
# 接口一致，根据机器是否有gpu自动选择
# cpu版本：pip install onnxruntime
# gpu版本：pip install onnxruntime-gpu
import onnxruntime as ort


def read_image(image_path, input_h=112, input_w=96, mode='bgr'):
    """
    保持长宽比resize到（input_h, input_w）,长度不够的边两边补黑
    输出resize及补黑之后的图片，scale，和上下左右补黑的像素个数（以便将输出的点映射到原图）
    """
    if mode not in ['rgb', 'bgr']:
        raise ValueError(f'wrong mode={mode} | only support "rgb" or "bgr"')

    image = cv2.imread(image_path)  # BGR, HWC
    if mode == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Albumentations uses RGB format

    h, w = image.shape[:2]
    # 保持长宽比resize，不够的补黑
    scale = 1.0
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    if h != input_h or w != input_w:
        scale = min(input_h / h, input_w / w)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h_resize, w_resize = image.shape[:2]
        if w_resize < input_w:
            pad_left = (input_w - w_resize) // 2
            pad_right = input_w - w_resize - pad_left
        if h_resize < input_h:
            pad_top = (input_h - h_resize) // 2
            pad_bottom = input_h - h_resize - pad_top
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, 0)

    return image, scale, pad_top, pad_bottom, pad_left, pad_right


def inference(onnx_path, image_path_list, max_batch_size=1, input_h=112, input_w=96):
    """
    如果你的onnx模型使用固定的batch_size(非dynamic)，需要将max_batch_size设置成onnx记录的值，否则会报错
    所有batch使用固定尺寸max_batch_size，尽管最后一个batch_size可能小于max_batch_size
    因为（已验证）如果inputs尺寸有变化，时间会很慢，怀疑是构建了新的ort_session并load到gpu
    """
    if not os.path.exists(onnx_path):
        raise ValueError(f'cannot find onnx file | {onnx_path}')

    ort_session = ort.InferenceSession(onnx_path)  # ort_session只需要构建一次

    # 打印一些信息
    print(f'\ngenerate onnxruntime session from onnx successful | {onnx_path}')
    print(f'providers={ort_session.get_providers()}')  # 优先使用第一个
    for i, node in enumerate(ort_session.get_inputs()):
        print(f'input {i} | {node}')
        batch_size = node.shape[0]
        if isinstance(batch_size, int) and batch_size != max_batch_size:
            raise ValueError(f'\nonnx uses static_batch_size={batch_size} != max_batch_size={max_batch_size}. '
                             f'Please set max_batch_size to {batch_size} and try again.')
    for i, node in enumerate(ort_session.get_outputs()):
        print(f'output {i} | {node}')
        batch_size = node.shape[0]
        if isinstance(batch_size, int) and batch_size != max_batch_size:
            raise ValueError(f'\nonnx uses static_batch_size={batch_size} != max_batch_size={max_batch_size}. '
                             f'Please set max_batch_size to {batch_size} and try again.')

    # 所有batch使用固定尺寸max_batch_size，尽管最后一个batch_size可能小于max_batch_size
    # 因为（已验证）如果inputs尺寸有变化，时间会很慢，怀疑是构建了新的ort_session并load到gpu
    inputs = np.zeros(shape=(max_batch_size, 3, input_h, input_w), dtype=np.float32)

    # ort_session第一次load到gpu会很慢，跑一次假的，方便之后真实图片计时
    ort_outputs = ort_session.run(output_names=None, input_feed={'nchw_rgb': inputs})

    # 将图片按max_batch_size划分成若干个batch，最后一个batch_size可能小于max_batch_size
    image_num = len(image_path_list)
    image_batch_list = [image_path_list[i:i + max_batch_size] for i in range(0, image_num, max_batch_size)]
    batch_num = len(image_batch_list)
    print(f'\nimage_num={image_num} | batch_num={batch_num} | max_batch_size={max_batch_size}')

    for i_batch, image_batch in enumerate(image_batch_list):
        batch_size = len(image_batch)  # 最后一个batch_size可能小于max_batch_size
        if batch_size > max_batch_size:
            raise ValueError(f'wrong batch_size={batch_size} > max_batch_size={max_batch_size}')

        for i_image, image_path in enumerate(image_batch):
            # MobileNetV2Agender使用rgb格式的输入
            image, scale, pad_top, pad_bottom, pad_left, pad_right = read_image(image_path, input_h, input_w, 'rgb')
            inputs[i_image] = image.transpose(2, 0, 1).astype(np.float32) / 255  # HWC to CHW

        t_start = time.time()
        ort_outputs = ort_session.run(output_names=None, input_feed={'nchw_rgb': inputs})  # list
        t_end = time.time()
        print(f'\nbatch {i_batch + 1}/{batch_num} | batch_size={batch_size}'
              f' | time={(t_end - t_start) * 1000:.2f}ms')

        # 输出尺寸是(max_batch_size, ...)，已经自动reshape过了
        gender = ort_outputs[0].argmax(axis=1)
        age = ort_outputs[1]

        # 如果batch_size < max_batch_size，取前batch_size个值就好
        for i_image, image_path in enumerate(image_batch):
            i_gender = 'male' if gender[i_image] == 0 else 'female'
            i_age = age[i_image]
            print(f'{image_path} | gender={i_gender} | age={i_age:.1f}')


if __name__ == '__main__':
    onnx_path = 'mobilenet_v2_agender_20210802_simp.onnx'

    test_image_dir = os.path.join(os.path.realpath(__file__).rsplit(os.sep, 2)[0], 'test_images')
    image_path_list = [os.path.join(test_image_dir, x) for x in os.listdir(test_image_dir) if x.endswith('.png')]
    image_path_list = sorted(image_path_list)

    # 如果onnx模型使用static_batch_size(非dynamic)，需要将max_batch_size设置成onnx记录的值，否则会报错
    inference(onnx_path, image_path_list, max_batch_size=10, input_h=112, input_w=96)

    print('\nfinished!')

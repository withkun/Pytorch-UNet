import argparse
import logging

import torch
from unet import UNet
import os.path as path
import onnx
import math


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints/checkpoint_epoch2200.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--img-size', nargs='+', type=int, default=[512,512], help='image size')  # height, width
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--channels', type=int, default=3, help='Number of image channels')

    return parser.parse_args()


def convert_to_onnx(model, img_size, onnx_path):
    im = torch.zeros(1, model.n_channels, *img_size).to('cpu')  # image size(1, 3, 512, 512) BCHW
    input_layer_names = ["images"]
    output_layer_names = ["output"]

    # Export the model
    print(f'Starting export with onnx {onnx.__version__}.')
    torch.onnx.export(model,
                      im,
                      f=onnx_path,
                      verbose=False,
                      opset_version=12,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=input_layer_names,
                      output_names=output_layer_names,
                      dynamic_axes=None)

    # Checks
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify onnx
    import onnxsim
    print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        dynamic_input_shape=False,
        input_shapes=None)
    assert check, 'assert check failed'
    onnx.save(model_onnx, onnx_path)

    print('Onnx model save as {}'.format(onnx_path))


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    file_name, file_extension = path.splitext(args.model)

    net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)

    device = 'cpu'
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    convert_to_onnx(net, args.img_size, file_name+'.onnx')

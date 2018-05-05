import os
import sys
import argparse
import json
import logging
logging.basicConfig(level=logging.DEBUG)

# need to load success
import mxnet as mx
import caffe
from caffe import layers as CL

# test
from test_network.resnet import get_symbol
from utils.save_sym_model import save_symbol_model_for_test
from utils.convert_symbol2proto import convert_symbol2proto
from utils.convert_params2model import convert_params2model

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="convert mxnet model to caffe model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mxnet_symbol_name', type=str, help='mxnet symbol name ')
    parser.add_argument('--load_epoch', type=int, help='mxnet symbol epoch')
    parser.add_argument('--caffe_net_name', type=str, help='output caffe name(output {name}.prototxt and {name}.caffemodel')

    args = parser.parse_args()

    logging.info('arguments [%s]', args)

    # generate network for test
    # network = get_symbol(1000, 18, "3,224,224")
    # save_symbol_model_for_test('resnet', 0, network, {'data': (1, 3, 128, 128), 'softmax_label': (1,)})

    symbol, arg_params, aux_params = mx.model.load_checkpoint('model/src/resnet_v5', 8)

    caffe_prototxt = str(convert_symbol2proto(symbol))
    # with open('model/dst/resnet_v5.prototxt', 'w') as prototxt_file:
    #     prototxt_file.write(caffe_prototxt)
    convert_params2model('model/dst/resnet_v5.prototxt', 'model/dst/resnet_v5.caffemodel', symbol, arg_params, aux_params)
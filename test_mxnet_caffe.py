import logging
import os
import time
import cv2
import numpy as np
import math
import caffe
import mxnet as mx

if __name__ == '__main__':

    mxnet_model_prefix = './resnet'
    mxnet_load_epoch = 0
    symbol, arg_params, aux_params = mx.model.load_checkpoint(mxnet_model_prefix, mxnet_load_epoch)

    caffe_prototxt = './resnet.prototxt'
    caffe_caffemodel = './resnet.caffemodel'
    caffe_net = caffe.Net(caffe_prototxt, caffe_caffemodel, caffe.TEST)

    input_blob = np.ones((1, 3, 128, 128)) * 128

    # mxnet forward
    symbol = symbol.get_internals()['fc1_output']
    # a = mx.viz.plot_network(symbol, shape={'data': (1, 3, 128, 128)})
    # a.view()
    model = mx.module.Module(context=mx.cpu(), symbol=symbol, label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (1, 3, 128, 128))])
    model.set_params(arg_params=arg_params, aux_params=aux_params)
    model.forward(mx.io.DataBatch([mx.nd.array(input_blob)]))
    mxnet_outputs = model.get_outputs()

    # caffe forward
    caffe_net.blobs['data'].reshape(*(input_blob.shape))
    caffe_net.reshape()
    forward_kwargs = {'data': input_blob}

    caffe_outputs = caffe_net.forward(**forward_kwargs)

    print caffe_outputs

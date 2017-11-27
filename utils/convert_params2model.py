import mxnet as mx
import caffe
import logging
logging.basicConfig(level=logging.DEBUG)

def convert_params2model(prototxt_file, dst_caffemodel_file, symbol, arg_params, aux_params):
    attr_dict = symbol.attr_dict()
    caffe_net = caffe.Net(prototxt_file, caffe.TRAIN)
    mxnet_all_keys = arg_params.keys() + aux_params.keys()
    for idx, key_mxnet in enumerate(mxnet_all_keys):
        if key_mxnet == 'data' or key_mxnet.endswith('_label'):
            logging.info('%s pass' % key_mxnet)
        elif key_mxnet.endswith('_weight'):
            key_caffe = key_mxnet.replace('_weight', '')
            caffe_net.params[key_caffe][0].data[...] = arg_params[key_mxnet].asnumpy()
        elif key_mxnet.endswith('_bias'):
            key_caffe = key_mxnet.replace('_bias', '')
            caffe_net.params[key_caffe][1].data[...] = arg_params[key_mxnet].asnumpy()
        elif key_mxnet.endswith('_gamma'):
            key_caffe = key_mxnet.replace('_gamma', '_scale')
            if 'fix_gamma' in attr_dict[key_mxnet.replace('_gamma', '')] and eval(attr_dict[key_mxnet.replace('_gamma', '')]['fix_gamma']) is True:
                caffe_net.params[key_caffe][0].data[...] = 1
            else:
                caffe_net.params[key_caffe][0].data[...] = arg_params[key_mxnet].asnumpy()
        elif key_mxnet.endswith('_beta'):
            key_caffe = key_mxnet.replace('_beta', '_scale')
            caffe_net.params[key_caffe][1].data[...] = arg_params[key_mxnet].asnumpy()
        elif key_mxnet.endswith('_moving_mean'):
            key_caffe = key_mxnet.replace('_moving_mean', '')
            caffe_net.params[key_caffe][0].data[...] = aux_params[key_mxnet].asnumpy()
            caffe_net.params[key_caffe][2].data[...] = 1
        elif key_mxnet.endswith('_moving_var'):
            key_caffe = key_mxnet.replace('_moving_var', '')
            caffe_net.params[key_caffe][1].data[...] = aux_params[key_mxnet].asnumpy()
            caffe_net.params[key_caffe][2].data[...] = 1
        else:
            logging.warn('unknown %s pass' % key_mxnet)
    caffe_net.save(dst_caffemodel_file)

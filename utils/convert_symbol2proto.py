import mxnet as mx
import caffe
from caffe import layers as CL
import json
import logging
logging.basicConfig(level=logging.DEBUG)
import math

NO_INPLACE = False

def convert_symbol2proto(symbol):
    def looks_like_weight(name):
        """Internal helper to figure out if node should be hidden with `hide_weights`.
        """
        if name.endswith("_weight"):
            return True
        if name.endswith("_bias"):
            return True
        if name.endswith("_beta") or name.endswith("_gamma") or name.endswith("_moving_var") or name.endswith(
                "_moving_mean"):
            return True
        return False

    json_symbol = json.loads(symbol.tojson())
    all_nodes = json_symbol['nodes']
    no_weight_nodes = []
    for node in all_nodes:
        op = node['op']
        name = node['name']
        if op == 'null':
            if looks_like_weight(name):
                continue
        no_weight_nodes.append(node)

    # build next node dict
    next_node = dict()
    for node in no_weight_nodes:
        node_name = node['name']
        for input in node['inputs']:
            last_node_name = all_nodes[input[0]]['name']
            if last_node_name in next_node:
                next_node[last_node_name].append(node_name)
            else:
                next_node[last_node_name] = [node_name]

    supported_op_type = ['null', 'BatchNorm', 'Convolution', 'Activation', 'Pooling', 'elemwise_add', 'SliceChannel',
                         'FullyConnected', 'SoftmaxOutput', '_maximum', 'add_n', 'Concat', '_mul_scalar', 'Deconvolution', 'UpSampling']
    top_dict = dict()
    caffe_net = caffe.NetSpec()
    for node in no_weight_nodes:
        if node['op'] == 'null':
            input_param = dict()
            if node['name'] == 'data':
                input_param['shape'] = dict(dim=[1, 3, 160, 160])
            else:
                input_param['shape'] = dict(dim=[1])
            top_data = CL.Input(ntop=1, input_param=input_param)
            top_dict[node['name']] = [top_data]
            setattr(caffe_net, node['name'], top_data)
        elif node['op'].endswith('_copy'):
            pass
        elif node['op'] == 'BatchNorm':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            attr = node['attrs']
            in_place = False
            if len(next_node[bottom_node_name]) == 1:
                in_place = True
            if 'momentum' in attr:
                momentum = float(attr['momentum'])
            else:
                momentum = 0.9
            if 'eps' in attr:
                eps = float(attr['eps'])
            else:
                eps = 0.001
            if NO_INPLACE:
                in_place = False
            bn_top = CL.BatchNorm(top_dict[bottom_node_name][input[1]], ntop=1,
                                  batch_norm_param=dict(use_global_stats=True,
                                                        moving_average_fraction=momentum,
                                                        eps=eps), in_place=in_place)
            setattr(caffe_net, node['name'], bn_top)
            scale_top = CL.Scale(bn_top, ntop=1, scale_param=dict(bias_term=True), in_place=not NO_INPLACE)
            top_dict[node['name']] = [scale_top]
            setattr(caffe_net, node['name'] + '_scale', scale_top)
        elif node['op'] == 'Convolution':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            attr = node['attrs']
            convolution_param = dict()
            if 'kernel' in attr:
                kernel_size = eval(attr['kernel'])
                assert kernel_size[0] == kernel_size[1]
                convolution_param['kernel_size'] = kernel_size[0]
            else:
                convolution_param['kernel_size'] = 1
            if 'no_bias' in attr:
                convolution_param['bias_term'] = not eval(attr['no_bias'])
            if 'num_group' in attr:
                convolution_param['group'] = int(attr['num_group'])
            convolution_param['num_output'] = int(attr['num_filter'])
            if 'pad' in attr:
                pad_size = eval(attr['pad'])
                assert pad_size[0] == pad_size[1]
                convolution_param['pad'] = pad_size[0]
            if 'stride' in attr:
                stride_size = eval(attr['stride'])
                assert stride_size[0] == stride_size[1]
                convolution_param['stride'] = stride_size[0]
            conv_top = CL.Convolution(top_dict[bottom_node_name][input[1]], ntop=1, convolution_param=convolution_param)
            top_dict[node['name']] = [conv_top]
            setattr(caffe_net, node['name'], conv_top)
        elif node['op'] == 'Deconvolution':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            attr = node['attrs']
            convolution_param = dict()
            if 'kernel' in attr:
                kernel_size = eval(attr['kernel'])
                assert kernel_size[0] == kernel_size[1]
                convolution_param['kernel_size'] = kernel_size[0]
            else:
                convolution_param['kernel_size'] = 1
            if 'no_bias' in attr:
                convolution_param['bias_term'] = not eval(attr['no_bias'])
            else:
                convolution_param['bias_term'] = False
            if 'num_group' in attr:
                convolution_param['group'] = int(attr['num_group'])
            convolution_param['num_output'] = int(attr['num_filter'])
            if 'pad' in attr:
                pad_size = eval(attr['pad'])
                assert pad_size[0] == pad_size[1]
                convolution_param['pad'] = pad_size[0]
            if 'stride' in attr:
                stride_size = eval(attr['stride'])
                assert stride_size[0] == stride_size[1]
                convolution_param['stride'] = stride_size[0]
            conv_top = CL.Deconvolution(top_dict[bottom_node_name][input[1]], ntop=1, convolution_param=convolution_param)
            top_dict[node['name']] = [conv_top]
            setattr(caffe_net, node['name'], conv_top)
        elif node['op'] == 'UpSampling':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            attr = node['attrs']
            convolution_param = dict()
            if 'scale' in attr:
                kernel_size = 2 * eval(attr['scale']) - eval(attr['scale']) % 2
                convolution_param['kernel_size'] = kernel_size
            else:
                convolution_param['kernel_size'] = 1
            convolution_param['bias_term'] = False
            convolution_param['num_output'] = int(attr['num_filter'])
            convolution_param['group'] = int(attr['num_filter'])
            convolution_param['pad'] = int(math.ceil((eval(attr['scale']) - 1) / 2.))
            convolution_param['stride'] = eval(attr['scale'])
            conv_top = CL.Deconvolution(top_dict[bottom_node_name][input[1]], ntop=1,
                                        convolution_param=convolution_param)
            top_dict[node['name']] = [conv_top]
            setattr(caffe_net, node['name'], conv_top)
        elif node['op'] == 'Activation':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            attr = node['attrs']
            in_place = False
            if len(next_node[bottom_node_name]) == 1:
                in_place = True
            if NO_INPLACE:
                in_place = False
            if attr['act_type'] == 'relu':
                ac_top = CL.ReLU(top_dict[bottom_node_name][input[1]], ntop=1, in_place=in_place)
            elif attr['act_type'] == 'sigmoid':
                ac_top = CL.Sigmoid(top_dict[bottom_node_name][input[1]], ntop=1, in_place=in_place)
            elif attr['act_type'] == 'tanh':
                ac_top = CL.TanH(top_dict[bottom_node_name][input[1]], ntop=1, in_place=in_place)
            top_dict[node['name']] = [ac_top]
            setattr(caffe_net, node['name'], ac_top)
        elif node['op'] == 'Pooling':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            attr = node['attrs']
            pooling_param = dict()
            if attr['pool_type'] == 'avg':
                pooling_param['pool'] = 1
            elif attr['pool_type'] == 'max':
                pooling_param['pool'] = 0
            else:
                assert False, attr['pool_type']
            if 'global_pool' in attr and eval(attr['global_pool']) is True:
                pooling_param['global_pooling'] = True
            else:
                if 'kernel' in attr:
                    kernel_size = eval(attr['kernel'])
                    assert kernel_size[0] == kernel_size[1]
                    pooling_param['kernel_size'] = kernel_size[0]
                if 'pad' in attr:
                    pad_size = eval(attr['pad'])
                    assert pad_size[0] == pad_size[1]
                    pooling_param['pad'] = pad_size[0]
                if 'stride' in attr:
                    stride_size = eval(attr['stride'])
                    assert stride_size[0] == stride_size[1]
                    pooling_param['stride'] = stride_size[0]
            pool_top = CL.Pooling(top_dict[bottom_node_name][input[1]], ntop=1, pooling_param=pooling_param)
            top_dict[node['name']] = [pool_top]
            setattr(caffe_net, node['name'], pool_top)
        elif node['op'] == 'elemwise_add' or node['op'] == 'add_n':
            input_a = node['inputs'][0]
            while True:
                if all_nodes[input_a[0]]['op'] not in supported_op_type:
                    input_a = all_nodes[input_a[0]]['inputs'][0]
                else:
                    break
            input_b = node['inputs'][1]
            while True:
                if all_nodes[input_b[0]]['op'] not in supported_op_type:
                    input_b = all_nodes[input_b[0]]['inputs'][0]
                else:
                    break
            bottom_node_name_a = all_nodes[input_a[0]]['name']
            bottom_node_name_b = all_nodes[input_b[0]]['name']
            eltwise_param = dict()
            eltwise_param['operation'] = 1
            ele_add_top = CL.Eltwise(top_dict[bottom_node_name_a][input_a[1]], top_dict[bottom_node_name_b][input_b[1]],
                                     ntop=1, eltwise_param=eltwise_param)
            top_dict[node['name']] = [ele_add_top]
            setattr(caffe_net, node['name'], ele_add_top)
        elif node['op'] == '_maximum':
            input_a = node['inputs'][0]
            while True:
                if all_nodes[input_a[0]]['op'] not in supported_op_type:
                    input_a = all_nodes[input_a[0]]['inputs'][0]
                else:
                    break
            input_b = node['inputs'][1]
            while True:
                if all_nodes[input_b[0]]['op'] not in supported_op_type:
                    input_b = all_nodes[input_b[0]]['inputs'][0]
                else:
                    break
            bottom_node_name_a = all_nodes[input_a[0]]['name']
            bottom_node_name_b = all_nodes[input_b[0]]['name']
            eltwise_param = dict()
            eltwise_param['operation'] = 2
            ele_add_top = CL.Eltwise(top_dict[bottom_node_name_a][input_a[1]], top_dict[bottom_node_name_b][input_b[1]],
                                     ntop=1, eltwise_param=eltwise_param)
            top_dict[node['name']] = [ele_add_top]
            setattr(caffe_net, node['name'], ele_add_top)
        elif node['op'] == '_mul_scalar':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            attr = node['attrs']
            in_place = False
            if len(next_node[bottom_node_name]) == 1:
                in_place = True
            if NO_INPLACE:
                in_place = False

            scale_top = CL.Scale(top_dict[bottom_node_name][input[1]], ntop=1, scale_param=dict(bias_term=False, filler=dict(value=-1)), in_place=in_place)
            # scale_top = CL.Power(top_dict[bottom_node_name][input[1]], power=1.0, scale=float(attr['scalar']), shift=0, in_place=in_place)

            top_dict[node['name']] = [scale_top]
            setattr(caffe_net, node['name'], scale_top)
        elif node['op'] == 'SliceChannel':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            slice_param = dict()
            slice_param['slice_dim'] = 1
            slice_num = 2
            slice_outputs = CL.Slice(top_dict[bottom_node_name][input[1]], ntop=slice_num, slice_param=slice_param)
            top_dict[node['name']] = slice_outputs
            for idx, output in enumerate(slice_outputs):
                setattr(caffe_net, node['name'] + '_' + str(idx), output)
        elif node['op'] == 'FullyConnected':
            input = node['inputs'][0]
            while True:
                if all_nodes[input[0]]['op'] not in supported_op_type:
                    input = all_nodes[input[0]]['inputs'][0]
                else:
                    break
            bottom_node_name = all_nodes[input[0]]['name']
            attr = node['attrs']
            inner_product_param = dict()
            inner_product_param['num_output'] = int(attr['num_hidden'])
            fc_top = CL.InnerProduct(top_dict[bottom_node_name][input[1]], ntop=1,
                                     inner_product_param=inner_product_param)
            top_dict[node['name']] = [fc_top]
            setattr(caffe_net, node['name'], fc_top)
        elif node['op'] == 'SoftmaxOutput':
            input_a = node['inputs'][0]
            while True:
                if all_nodes[input_a[0]]['op'] not in supported_op_type:
                    input_a = all_nodes[input_a[0]]['inputs'][0]
                else:
                    break
            input_b = node['inputs'][1]
            while True:
                if all_nodes[input_b[0]]['op'] not in supported_op_type:
                    input_b = all_nodes[input_b[0]]['inputs'][0]
                else:
                    break
            bottom_node_name_a = all_nodes[input_a[0]]['name']
            bottom_node_name_b = all_nodes[input_b[0]]['name']
            softmax_loss = CL.SoftmaxWithLoss(top_dict[bottom_node_name_a][input_a[1]],
                                              top_dict[bottom_node_name_b][input_b[1]], ntop=1)
            top_dict[node['name']] = [softmax_loss]
            setattr(caffe_net, node['name'], softmax_loss)
        elif node['op'] == 'Concat':
            if len(node['inputs']) == 2:
                input_a = node['inputs'][0]
                while True:
                    if all_nodes[input_a[0]]['op'] not in supported_op_type:
                        input_a = all_nodes[input_a[0]]['inputs'][0]
                    else:
                        break
                input_b = node['inputs'][1]
                while True:
                    if all_nodes[input_b[0]]['op'] not in supported_op_type:
                        input_b = all_nodes[input_b[0]]['inputs'][0]
                    else:
                        break
                bottom_node_name_a = all_nodes[input_a[0]]['name']
                bottom_node_name_b = all_nodes[input_b[0]]['name']
                concat_top = CL.Concat(top_dict[bottom_node_name_a][input_a[1]], top_dict[bottom_node_name_b][input_b[1]], ntop=1)
                top_dict[node['name']] = [concat_top]
                setattr(caffe_net, node['name'], concat_top)
            elif len(node['inputs']) == 3:
                input_a = node['inputs'][0]
                while True:
                    if all_nodes[input_a[0]]['op'] not in supported_op_type:
                        input_a = all_nodes[input_a[0]]['inputs'][0]
                    else:
                        break
                input_b = node['inputs'][1]
                while True:
                    if all_nodes[input_b[0]]['op'] not in supported_op_type:
                        input_b = all_nodes[input_b[0]]['inputs'][0]
                    else:
                        break
                input_c = node['inputs'][2]
                while True:
                    if all_nodes[input_c[0]]['op'] not in supported_op_type:
                        input_c = all_nodes[input_c[0]]['inputs'][0]
                    else:
                        break
                bottom_node_name_a = all_nodes[input_a[0]]['name']
                bottom_node_name_b = all_nodes[input_b[0]]['name']
                bottom_node_name_c = all_nodes[input_c[0]]['name']
                concat_top = CL.Concat(top_dict[bottom_node_name_a][input_a[1]],
                                       top_dict[bottom_node_name_b][input_b[1]],
                                       top_dict[bottom_node_name_c][input_c[1]], ntop=1)
                top_dict[node['name']] = [concat_top]
                setattr(caffe_net, node['name'], concat_top)
        else:
            logging.warn('unknown op type = %s' % node['op'])

    return caffe_net.to_proto()

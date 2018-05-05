import caffe
import math
import numpy as np

prototxt = r'model/dst/person_seg_v10_deploy.prototxt'
caffemodel = r'model/dst/person_seg_v10_finetune_iter_80000.caffemodel'

dst_prototxt = r'model/dst/person_seg_v10_deploy_without_bn.prototxt'
dst_caffemodel = r'model/dst/person_seg_v10_deploy_without_bn.caffemodel'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
net_dst = caffe.Net(dst_prototxt, caffe.TEST)

for k in net_dst.params:
    if k in net.params:
        for i in range(len(net.params[k])):
            net_dst.params[k][i].data[...] = net.params[k][i].data[...]
            print 'copy from', k, net.params[k][i].data.shape

for i in range(len(net.layers)):
    if net.layers[i].type == 'Convolution':
        print net._layer_names[i], net.layers[i].type
        conv_name = net._layer_names[i]

        # start_remove = False
        # if 'CPM' not in conv_name or 'relu1_' in conv_name or 'relu2_' in conv_name or 'relu3_' in conv_name or 'relu4_CPM_L1_conv2d_dw' in conv_name or 'relu4_CPM_L1_conv2d_pw' in conv_name:
        #     start_remove = True

        start_remove = True

        j = i + 1
        print 'next type', net.layers[j].type
        if net.layers[j].type == 'BatchNorm' and start_remove:
            print ' ', net._layer_names[j], net.layers[j].type
            print ' ', net._layer_names[j + 1], net.layers[j + 1].type
            bn_name = net._layer_names[j]
            scale_name = net._layer_names[j + 1]

            bn_mean = net.params[bn_name][0].data
            bn_variance = net.params[bn_name][1].data
            bn_scale = net.params[bn_name][2].data
            scale_weight = net.params[scale_name][0].data
            scale_bias = net.params[scale_name][1].data

            # print '  ', bn_name, bn_mean, bn_variance, bn_scale
            # print '  ', scale_name, scale_weight, scale_bias

            dst_conv_weight = net.params[conv_name][0].data
            if len(net.params[conv_name]) > 1:
                dst_conv_bias = net.params[conv_name][1].data
            else:
                dst_conv_bias = 0

            if np.count_nonzero(bn_variance) != bn_variance.size:
                assert False
            alpha = scale_weight / np.sqrt(bn_variance / bn_scale + 0.001) #remember reading eps

            print 'len(dst_conv_weight)', len(dst_conv_weight), 'len(alpha)', len(alpha)
            assert len(dst_conv_weight) == len(alpha)
            for k in range(len(alpha)):
                dst_conv_weight[k] = dst_conv_weight[k] * alpha[k]

            dst_conv_bias = dst_conv_bias * alpha + (scale_bias - (bn_mean / bn_scale) * alpha)
            net_dst.params[conv_name][0].data[...] = dst_conv_weight

            # print '  ', dst_conv_weight
            # print '  ', dst_conv_bias

            if len(net_dst.params[conv_name]) > 1:
                net_dst.params[conv_name][1].data[...] = dst_conv_bias
    if net.layers[i].type == 'InnerProduct' and start_remove:
        print net._layer_names[i], net.layers[i].type
        ip_name = net._layer_names[i]
        j = i + 1
        if net.layers[j].type == 'BatchNorm':
            print ' ', net._layer_names[j], net.layers[j].type
            print ' ', net._layer_names[j + 1], net.layers[j + 1].type
            bn_name = net._layer_names[j]
            scale_name = net._layer_names[j + 1]

            bn_mean = net.params[bn_name][0].data
            bn_variance = net.params[bn_name][1].data
            bn_scale = net.params[bn_name][2].data

            scale_weight = net.params[scale_name][0].data
            scale_bias = net.params[scale_name][1].data

            # print bn_name
            # print bn_mean, bn_variance, bn_scale
            # print scale_name
            # print scale_weight, scale_bias

            dst_inner_weight = net.params[ip_name][0].data
            if np.count_nonzero(bn_variance) != bn_variance.size:
                assert False

            alpha = scale_weight / np.sqrt(bn_variance / bn_scale)
            if len(net.params[ip_name]) > 1:
                dst_inner_bias = net.params[ip_name][1].data
            else:
                dst_inner_bias = 0
            print 'len(dst_inner_weight)', len(dst_inner_weight), 'len(alpha)', len(alpha)
            assert len(dst_inner_weight) == len(alpha)
            for k in range(len(alpha)):
                dst_inner_weight[k] = dst_inner_weight[k] * alpha[k]

            dst_inner_bias = dst_inner_bias * alpha + (scale_bias - (bn_mean / bn_scale) * alpha)
            net_dst.params[ip_name][0].data[...] = dst_inner_weight

            if len(net_dst.params[ip_name]) > 1:
                net_dst.params[ip_name][1].data[...] = dst_inner_bias

net_dst.save(dst_caffemodel)
print 'FINISH ##############################'

# net_dst.save(dst_caffemodel)
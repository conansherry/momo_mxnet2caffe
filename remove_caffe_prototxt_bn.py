import caffe
import math
import numpy as np
import json
from caffe.proto import caffe_pb2
from google.protobuf import text_format

prototxt = r'model/dst/hand_net_320x192_stride16.prototxt'

dst_prototxt = r'model/dst/hand_net_320x192_stride16_without_bn.prototxt'

def readProtoFile(filepath, parser_object):
    file = open(filepath, "r")
    text_format.Merge(str(file.read()), parser_object)
    file.close()
    return parser_object

def readProtoSolverFile(filepath):
    solver_config = caffe.proto.caffe_pb2.NetParameter()
    return readProtoFile(filepath, solver_config)

net_params = readProtoSolverFile(prototxt)

outfile = open(dst_prototxt, 'w')
outfile.write('name: \"' + net_params.name + '\"\n')
outfile.write('\n')
print net_params.name
index = 0
start_remove = False
for layer in net_params.layer:
    print layer.name
    index = index + 1
    if (layer.type == 'Convolution' or layer.type == 'InnerProduct') and index < len(net_params.layer) and net_params.layer[index].type == 'BatchNorm':
        layer.top[0] = net_params.layer[index + 1].top[0]

        # if 'CPM' not in layer.name or 'relu1_' in layer.name or 'relu2_' in layer.name or 'relu3_' in layer.name or 'relu4_CPM_L1_conv2d_dw' in layer.name or 'relu4_CPM_L1_conv2d_pw' in layer.name:
        #     start_remove = True

        start_remove = True

    if layer.type == 'BatchNorm' and start_remove:
        continue
    if layer.type == 'Scale' and start_remove:
        start_remove = False
        continue
    if layer.type == 'Convolution' and net_params.layer[index].type == 'BatchNorm' and start_remove:
        layer.convolution_param.bias_term = True
    if layer.type == 'InnerProduct' and (index < len(net_params.layer) and net_params.layer[index].type == 'BatchNorm') and start_remove:
        layer.inner_product_param.bias_term = True
    outfile.write('layer {\n')
    outfile.write('  '.join(('\n' + str(layer)).splitlines(True)))
    outfile.write('\n}\n\n')
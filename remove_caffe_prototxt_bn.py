import caffe
import math
import numpy as np
import json
from caffe.proto import caffe_pb2
from google.protobuf import text_format

prototxt = r'resnet_v4_stage.prototxt'

dst_prototxt = r'resnet_v4_stage_without_bn.prototxt'

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
    index = index + 1
    if (layer.type == 'Convolution' or layer.type == 'InnerProduct') and net_params.layer[index].type == 'BatchNorm':
        if net_params.layer[index + 2].type == 'ReLU' or net_params.layer[index + 2].type == 'Sigmoid' or net_params.layer[index + 2].type == 'TanH':
            layer.top[0] = net_params.layer[index + 2].top[0]
            start_remove = True
    elif (layer.type == 'Convolution' or layer.type == 'InnerProduct') and (net_params.layer[index].type == 'ReLU' or net_params.layer[index].type == 'Sigmoid' or net_params.layer[index].type == 'TanH'):
        layer.top[0] = net_params.layer[index].top[0]
    if layer.type == 'BatchNorm' and start_remove:
        continue
    if layer.type == 'Scale' and start_remove:
        continue
    if layer.type == 'Convolution' and net_params.layer[index].type == 'BatchNorm':
        layer.convolution_param.bias_term = True
    if layer.type == 'InnerProduct' and (index < len(net_params.layer) and net_params.layer[index].type == 'BatchNorm'):
        layer.inner_product_param.bias_term = True
    outfile.write('layer {\n')
    if layer.type == 'ReLU' or layer.type == 'Sigmoid' or layer.type == 'TanH':
        layer.bottom[0] = layer.top[0]
    outfile.write('  '.join(('\n' + str(layer)).splitlines(True)))
    outfile.write('\n}\n\n')
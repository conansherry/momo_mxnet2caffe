import mxnet as mx
import logging
logging.basicConfig(level=logging.DEBUG)

def save_params(prefix, epoch, arg_params, aux_params):
    """Checkpoint the model data into file.
    :param prefix: Prefix of model name.
    :param epoch: The epoch number of the model.
    :param arg_params: dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    :param aux_params: dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    :return: None
    prefix-epoch.params will be saved for parameters.
    """
    save_dict = {('arg:%s' % k) : v for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v for k, v in aux_params.items()})
    param_name = '%s-%04d.params' % (prefix, epoch)
    mx.nd.save(param_name, save_dict)

def save_symbol_model_for_test(prefix, epoch, symbol, shape):

    arg_shape, _, aux_shape = symbol.infer_shape(**shape)
    arg_shape_dict = dict(zip(symbol.list_arguments(), arg_shape))
    aux_shape_dict = dict(zip(symbol.list_auxiliary_states(), aux_shape))

    arg_params = dict()
    aux_params = dict()
    for k in symbol.list_arguments():
        arg_params[k] = mx.random.normal(0, 0.1, shape=arg_shape_dict[k])

    for k in symbol.list_auxiliary_states():
        aux_params[k] = mx.random.normal(0, 0.1, shape=aux_shape_dict[k])

    json_name = '%s-symbol.json' % prefix
    symbol.save(json_name)
    save_params(prefix, epoch, arg_params, aux_params)


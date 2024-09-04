import os
import itertools
import torch
import scipy.io as scio

from .CroSP import Completer
from .util import get_logger, build_adj
from .datasets import load_data


def main(save_path, config):
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    logger = get_logger()
    logger.info('Dataset:' + str(config.data_name))

    # Load data
    X_list = load_data(config, data_path=save_path)
    x1_train_raw = X_list[0]
    x2_train_raw = X_list[2]
    x1_train_raw_edge = X_list[1]
    x2_train_raw_edge = X_list[3]
    metric_graph = X_list[4]

    x1_train = torch.from_numpy(x1_train_raw).to(device)
    x2_train = torch.from_numpy(x2_train_raw).to(device)
    x1_edge = build_adj(x1_train_raw_edge)
    x2_edge = build_adj(x2_train_raw_edge)

    input_size = x1_train.size(1)
    # Build the model
    COMPLETER = Completer(config, input_size)
    optimizer = torch.optim.Adam(
        itertools.chain(COMPLETER.spa_encoder.parameters(), COMPLETER.img_encoder.parameters(),
                        COMPLETER.fusion_model.parameters()),
        lr=config.lr, weight_decay=config.weight_decay)
    COMPLETER.to_device(device)

    logger.info(COMPLETER.spa_encoder)
    logger.info(COMPLETER.img_encoder)
    logger.info(COMPLETER.fusion_model)
    logger.info(optimizer)

    # Training
    z_f, x_hat1, x_hat2, z_f1 = COMPLETER.train(config, logger, x1_train, x2_train, x1_edge, x2_edge, metric_graph, optimizer)
    latent_fusion = z_f.cpu().detach().numpy()
    x_hat1 = x_hat1.cpu().detach().numpy()
    x_hat2 = x_hat2.cpu().detach().numpy()
    z_f1 = z_f1.cpu().detach().numpy()

    task_type = config.task_type
    data_type = config.data_type
    data_name = config.data_name

    if task_type == 'Identify' and data_type == '10X':
        output_dir = os.path.join(save_path, 'Data', config.dataset_name, data_name)
        os.makedirs(output_dir, exist_ok=True)
        scio.savemat(os.path.join(output_dir, f'{data_name}_model_output.mat'),
                     {'data': z_f1})
    elif task_type == 'Identify':
        output_dir = os.path.join(save_path, 'Data', data_type, data_name)
        os.makedirs(output_dir, exist_ok=True)
        scio.savemat(os.path.join(output_dir, f'{data_name}_model_output.mat'),
                     {'data': latent_fusion})
    else:
        output_dir = os.path.join(save_path, 'Data', task_type, data_name)
        os.makedirs(output_dir, exist_ok=True)
        scio.savemat(os.path.join(output_dir, f'{data_name}_model_output.mat'),
                     {'data': latent_fusion})

    logger.info('--------------------Training over--------------------')
    return z_f1


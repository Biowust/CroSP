import os, sys
import scipy.io as scio

from sklearn.preprocessing import MinMaxScaler
from .util import tranfer

def load_data(config, data_path):
    """Load data """
    data_name = config.data_name
    data_type = config.data_type
    task_type = config.task_type
    main_dir = sys.path[0]
    Data_path = os.path.join(main_dir, data_path, "Data")
    X_list = []

    if task_type == 'Identify' and data_type == '10X':
        spa_data, A1, img_data, A2, metric_graph = get_data(root_dir=os.path.join(Data_path, config.dataset_name, data_name),
                                                            data_name=data_name)
        X_list.append(spa_data.astype('float32'))
        X_list.append(A1.astype('float32'))
        X_list.append(img_data.astype('float32'))
        X_list.append(A2.astype('float32'))
        X_list.append(metric_graph.astype('float32'))

    elif task_type == 'Identify':
        spa_data, A1, img_data, A2, metric_graph = get_data(root_dir=os.path.join(Data_path, data_type, config.data_name),
                                                            data_name=data_name)
        X_list.append(spa_data.astype('float32'))
        X_list.append(A1.astype('float32'))
        X_list.append(img_data.astype('float32'))
        X_list.append(A2.astype('float32'))
        X_list.append(metric_graph.astype('float32'))

    else:
        spa_data, A1, img_data, A2, metric_graph = get_data(root_dir=os.path.join(Data_path, task_type, data_name),
                                                            data_name=data_name)
        X_list.append(spa_data.astype('float32'))
        X_list.append(A1.astype('float32'))
        X_list.append(img_data.astype('float32'))
        X_list.append(A2.astype('float32'))
        X_list.append(metric_graph.astype('float32'))

    return X_list

def get_data(root_dir,data_name):
    data = scio.loadmat(os.path.join(root_dir, f"data_{data_name}.mat"))
    spa_data = data['X1']
    image_data = data['X2']
    scaler = MinMaxScaler()
    img_data = scaler.fit_transform(image_data)
    size = spa_data.shape[0]
    A1 = tranfer(data['A1'], size)
    A2 = tranfer(data['A2'], size)
    metric_graph = data['metric_graph']
    return spa_data, A1, img_data, A2, metric_graph
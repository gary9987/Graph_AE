from torch_geometric.data import DataLoader
from utils.CustomDataSet import SelectGraph
from utils.nb201_dataset_pg_ae import Dataset
from utils.train_utils import load_model_result, train_rg
from classification.Classifier import Regressor
import torch
import argparse


def main(arg):
    device = torch.device(arg.device)

    num_epoch = arg.e
    batch_size = arg.batch
    lc_length = arg.lc_length

    data_set = Dataset(hp='200', nb201_seed=777)
    if lc_length == 1:
        for i in range(len(data_set.train_data)):
            data_set.train_data[i].valid_acc = data_set.train_data[i].valid_acc[-1]
        for i in range(len(data_set.test_data)):
            data_set.test_data[i].valid_acc = data_set.test_data[i].valid_acc[-1]

    input_size = data_set.num_features
    shapes = list(map(int, arg.shapes.split(",")))
    train_set = DataLoader(data_set.train_data[:arg.n_train], batch_size=arg.n_train, shuffle=True)
    test_set = DataLoader(data_set.test_data[:arg.n_test], batch_size=arg.n_test, shuffle=False)

    if arg.m == "MIAGAE":
        from classification.Graph_AE import Net
        model = Net(input_size, arg.k, arg.depth, [arg.c_rate] * arg.depth, shapes, device).to(device)
    elif arg.m == "UNet":
        from classification.UNet import Net
        model = Net(input_size, arg.depth, arg.c_rate, shapes, device).to(device)
    elif arg.m == "Gpool":
        from classification.Gpool_model import Net
        model = Net(input_size, arg.depth, arg.c_rate, shapes, device).to(device)
    elif arg.m == "SAGpool":
        from classification.SAG_model import Net
        model = Net(input_size, arg.depth, [arg.c_rate] * arg.depth, shapes, device).to(device)
    else:
        print("model not found")
        return

    model.load_state_dict(torch.load(arg.model_dir + arg.m + ".ckpt"), strict=True)
    latent_train, latent_test = load_model_result(model, train_set, test_set, device)
    input_size2 = latent_train[0].shape[1]
    r_model = Regressor(input_size2, lc_length, arg.dropout).to(device)
    optimizer = torch.optim.Adam(r_model.parameters(), lr=arg.lr)
    train_rg(r_model, optimizer, device, train_set, test_set, num_epoch, latent_train, latent_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Global_Dict generator')

    # for compression model, same as train_compression.py
    parser.add_argument('--lc_length', type=int, default=1, help="regress learning curve length")
    parser.add_argument('--m', type=str, default='MIAGAE', help="model name")
    parser.add_argument('--device', type=str, default='cuda', help="cuda / cpu")
    parser.add_argument('--model_dir', type=str, default="data/model/", help="path to save model")
    parser.add_argument('--k', type=int, default=2, help="number of kernels")
    parser.add_argument('--depth', type=int, default=3, help="depth of encoder and decoder")
    parser.add_argument('--c_rate', type=float, default=0.8, help="compression ratio for each layer of encoder")
    parser.add_argument('--shapes', type=str, default="64,64,64", help="shape of each layer in encoder")

    # for classifier
    parser.add_argument('--n_skip', type=int, default=2000, help="skip some number of samples")
    parser.add_argument('--n_train', type=int, default=1000, help="number of samples for train set")
    parser.add_argument('--n_test', type=int, default=1563, help="number of samples for test set")
    parser.add_argument('--batch', type=int, default=32, help="batch size")
    parser.add_argument('--e', type=int, default=100, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    #parser.add_argument('--hidden', type=int, default=256, help="shape of each layer in encoder")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout rate")
    args = parser.parse_args()
    main(args)

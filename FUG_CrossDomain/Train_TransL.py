import torch as pt
import torch.nn as nn
import torch_geometric as pyg
from Dataset_Load import load_dataset
import GCL
from GCL.eval import get_split, LREvaluator
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import warnings
import torch
import torch_geometric
import time
import argparse

from FUGNN import DimensionNN_V2, GCN_encoder, FUG
from Utils import dimensional_sample_random, DAD_edge_index, freeze_test, get_embedding

def run(args):
    with open(args.log_dir, 'a') as f:
        f.write('\n\n\n')
        f.write(str(args))
    free_gpu_id = args.GPU_ID
    torch.cuda.set_device(int(free_gpu_id))
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    data_dir = args.datadir
    nb_epochs = args.nb_epochs
    lr = args.lr
    wd = args.wd
    hid_units = args.hid_units
    num_hop = args.num_hop
    activator = nn.PReLU if args.activator == 'PReLU' else nn.ReLU
    torch_geometric.seed.seed_everything(args.seed)
    seed = args.seed
    sample_size = args.sample_size
    feature_signal_dim = args.feature_signal_dim
    losslam_ssl = args.losslam_ssl
    losslam_sig_cross = args.losslam_sig_cross
    losslam_ssl_pos = args.losslam_ssl_pos
    if_rand = True if args.if_rand == 'True' else False
    

    train_dataset = train_dataset.split(',')
    dataset_len = len(train_dataset)
    train_data = []
    for d in train_dataset:
        train_data.append( load_dataset(d, data_dir)[0] )
        
    dnn = DimensionNN_V2(sample_size, feature_signal_dim*2, feature_signal_dim, activator)
    gnn = GCN_encoder(feature_signal_dim, hid_units, activator)
    model = FUG(D_NN=dnn, G_NN=gnn, S_mtd=dimensional_sample_random, sample_size=sample_size)
        
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
    if torch.cuda.is_available():
        # data = data.cuda()
        model = model.cuda()

    
    with tqdm(total=nb_epochs, desc='(T)') as pbar:
        for epoch in range(nb_epochs):
            l_ssl = 0
            l_ssl_pos = 0
            l_sig_cross = 0
            for data in train_data:
                data = data.cuda() if torch.cuda.is_available() else data
                model.update_sample(data.x, data.edge_index, if_rand=if_rand)
                model.train()
                optimiser.zero_grad()
                z = model(data.x, data.edge_index)
                loss_ssl = model.ssl_loss_fn_infoNCE(z)
                loss_ssl_pos = model.ssl_loss_fn_pos(z, data.edge_index)
                loss_sig_cross = model.dim_loss_fn()
                loss = losslam_ssl * loss_ssl + losslam_sig_cross * loss_sig_cross + losslam_ssl_pos * loss_ssl_pos
                loss.backward()
                optimiser.step()
                l_ssl = l_ssl + loss_ssl.item()
                l_ssl_pos = l_ssl_pos + loss_ssl_pos.item()
                l_sig_cross = l_sig_cross + loss_sig_cross.item()
            
            pbar.set_postfix({'loss_ssl': l_ssl, 
                                'loss_ssl_pos': l_ssl_pos,
                                'loss_sig_cross': l_sig_cross
                                }
                                )
            pbar.update()

    tag = './pt/' + args.train_dataset+'_'+str(time.time()) + '.pt'
    torch.save(model,tag)
    test_dataset = test_dataset.split(',')
    num_hops = num_hop.split(',') 
    
    for t in range(len(test_dataset)):
        test_data = test_dataset[t]
        num_hop = int(num_hops[t])
        model.eval()
        data = load_dataset(test_data, data_dir)[0]
        data = data.cuda() if torch.cuda.is_available()  else data
        z = get_embedding(data.x, data.edge_index, model, num_hop, if_rand=if_rand)
        m, r  = freeze_test(z, data.y, train_ratio=0.1, test_ratio=0.8, test_num=20)
        
        with open(args.log_dir, 'a') as f:
            f.write('\n')
            f.write(test_data+': ')
            f.write(' mean: '+str(m)+' std: '+ str(r))


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #setting arguments
    parser = argparse.ArgumentParser('FUGNN')
    parser.add_argument('--train_dataset', type=str, default='Cora,CiteSeer', help='Dataset name (splited by ,): Cora, Citeseer, PubMed, CS, Photo, Computers')
    parser.add_argument('--test_dataset', type=str, default='Cora,CiteSeer', help='Dataset name: Cora, Citeseer, PubMed, CS, Photo, Computers')
    parser.add_argument('--datadir', type=str, default='../../../datasets/', help='./data/dir/')
    parser.add_argument('--log_dir', type=str, default='./log/logCora.txt', help='./log/dir/')
    parser.add_argument('--GPU_ID', type=int, default=0, help='The GPU ID')
    parser.add_argument('--seed', type=int, default=66666, help='seed')

    parser.add_argument('--nb_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--activator', type=str, default='PReLU', help='Activator name: PReLU, ReLU')
    parser.add_argument('--if_rand', type=str, default='False', help='feature sample if_rand: True, False')
    
    parser.add_argument('--hid_units', type=int, default=512, help='representation size')
    parser.add_argument('--sample_size', type=int, default=1024, help='feature sample batch size')
    parser.add_argument('--feature_signal_dim', type=int, default=1024, help='feature signal dim')
    
    parser.add_argument('--losslam_ssl', type=float, default=1, help='hyper-parameter of ssl loss')
    parser.add_argument('--losslam_sig_cross', type=float, default=400, help='hyper-parameter of sig_cross loss')
    parser.add_argument('--losslam_ssl_pos', type=float, default=1, help='hyper-parameter of sig_self loss')

    parser.add_argument('--num_hop', type=str, default='5,3', help='graph view hop num (splited by ,), the len of it should be same with test_dataset')
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    run(args)
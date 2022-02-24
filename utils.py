import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs




def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    '''
    Load weight matrix function.
    :param file_path: str, the path of saved weight matrix file.
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return


def getXSYS(data, mode):
    TRAIN_NUM = Time_slot*(n_train+ n_val)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS


def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix for Diffusion GCN.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()


def spatial_difficulty(model,ADJPATH,ks)
    model.eval()
    A = torch.tensor(pd.read_csv(ADJPATH).values).to(device)
    YS_spatial = dict((i,[]) for i in range(A.shape[0]))
    A_q = calculate_random_walk_matrix(A).T.float()
    A_h = calculate_random_walk_matrix(A.T).T.float()
    edge_index=torch.vstack(torch.where(A!=0)).cpu()
    g1 = dgl.graph((edge_index[0],edge_index[1]))

    with torch.no_grad():
        for i in tqdm(range(A.shape[0])):
            node_idx, sub_edges_index, _, _ = k_hop_subgraph(i,ks,edge_index)
            retain_index=torch.tensor(list(set(node_idx.tolist())-{i}))
            missing_A=torch.zeros_like(A).float()
            missing_A[sub_edges_index[0],sub_edges_index[1]]=1
            for x, y, idx in train_iter:
                missing_index=torch.zeros_like(x)
                if retain_index.shape[0]!=0:
                    missing_index[:,:,retain_index]=1
                YS_batch = model(x*missing_index,A_q*missing_A,A_h*missing_A)
                YS_batch = YS_batch.cpu().numpy()
                YS_spatial[i].append(YS_batch[:,:,i:i+1])
            YS_spatial[i] = np.vstack(YS_spatial[i])
    return YS_spatial[i]

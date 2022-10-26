import argparse
from Config import Config
from RDGLite import *
from Attr_Model import *
import warnings
warnings.filterwarnings("ignore")


'''
Follow the code style of RDGCN:
https://github.com/StephanieWyt/RDGCN

@inproceedings{ijcai2019-733,
  title={Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs},
  author={Wu, Yuting and Liu, Xiao and Feng, Yansong and Wang, Zheng and Yan, Rui and Zhao, Dongyan},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},            
  pages={5278--5284},
  year={2019},
}
'''

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/DBP15K/")
    parser.add_argument("--lang", default="zh_en")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path
    lang = args.lang
    path = data_path + lang + '/'
    e1 = path + 'ent_ids_1'
    e2 = path + 'ent_ids_2'
    ill = path + 'ref_ent_ids'
    kg1 = path + 'triples_1'
    kg2 = path + 'triples_2'

    # RDGLite
    e = len(set(loadfile(e1, 1)) | set(loadfile(e2, 1)))
    ILL = loadfile(ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]
    KG1 = loadfile(kg1, 3)
    KG2 = loadfile(kg2, 3)
    KG = KG1 + KG2

    print('initial entity embedding loading ...')
    load_path = path + lang[0:2] + '_vectorList.json'
    e_input = load_json(load_path)
    print('load finish')

    e_epochs = Config.e_epochs
    a_epochs = Config.a_epochs
    e_dim = Config.e_dim
    a_dim = Config.a_dim
    r_dim = e_dim * 2
    dim = [e_dim, r_dim, a_dim]
    act_func = Config.act_func
    para = [Config.alpha, Config.gamma]
    k = Config.k
    rate = Config.rate

    print('RDGLite embedding')
    e_output, loss = r_build(e_input, dim, act_func, para, k, e, train, KG)
    s_vec, J = r_training(e_output, loss, rate, e_epochs, train, e, Config.k, test)

    print('RDGLite Result:')
    get_hits(s_vec, test)

    #print('save RDGLite embedding')
    #savepath = path + 'trained_vector.json'
    #save_json(s_vec, savepath, 1)

    print('Attr embedding')
    print('initial attribute embedding loading ...')
    load_path = path + lang[0:2] + '_attr_vector.json'
    a_input = load_json(load_path)
    print('load finish')
    print('entity-attribute adjacency matrix loading ...')
    load_path = path + lang[0:2] + '_ae_adj_sparse.json'
    adj_input = load_json(load_path)
    print('load finish')
    ae_mat = np.zeros([e, len(a_input)])
    for i in range(len(adj_input)):
        x = adj_input[i][0]
        y = adj_input[i][1]
        ae_mat[x][y] = adj_input[i][2]

    a_output, a_loss = a_build(dim, act_func, Config.gamma, k, train, a_input, ae_mat, e, KG)
    a_vec, J = a_training(a_output, a_loss, rate, a_epochs, train, k, test)

    print('Attribute Embedding Result:')
    get_hits(a_vec, test)

    print('final alignment Result:')
    theta = Config.theta
    e_embedding = np.array(s_vec)
    a_embedding = np.array(a_vec)
    vec = np.concatenate([e_embedding * theta, a_embedding * (1.0 - theta)], axis=1)
    get_hits(vec, test)

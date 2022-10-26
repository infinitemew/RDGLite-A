import math
from Layers import *


def rfunc(KG, e):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])

    r_num = len(head)
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    r_mat_ind = []
    r_mat_val = []
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1
        r_mat_ind.append([tri[0], tri[2]])
        r_mat_val.append(tri[1])

    r_mat = tf.SparseTensor(indices=r_mat_ind, values=r_mat_val, dense_shape=[e, e])

    return head, tail, head_r, tail_r, r_mat


def compute_r(inlayer, head_r, tail_r):
    head_l = tf.transpose(tf.constant(head_r, dtype=tf.float32))
    tail_l = tf.transpose(tf.constant(tail_r, dtype=tf.float32))
    L = tf.matmul(head_l, inlayer) / tf.expand_dims(tf.reduce_sum(head_l, axis=-1), -1)
    R = tf.matmul(tail_l, inlayer) / tf.expand_dims(tf.reduce_sum(tail_l, axis=-1), -1)
    r_embeddings = tf.concat([L, R], axis=-1)

    return r_embeddings


def get_dual_input(inlayer, head, tail, head_r, tail_r):
    r_embeddings = compute_r(inlayer, head_r, tail_r)
    count_r = len(head)
    adj_r = np.zeros((count_r, count_r))
    for i in range(count_r):
        for j in range(count_r):
            a_h = len(head[i] & head[j]) / len(head[i] | head[j])
            a_t = len(tail[i] & tail[j]) / len(tail[i] | tail[j])
            adj_r[i][j] = a_h + a_t

    return r_embeddings, adj_r


def get_loss(outlayer, ILL, gamma, k):
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))

    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


def r_build(e_input, dim, act_func, para, k, e, train, KG):
    alpha = para[0]
    gamma = para[1]
    e_dim = dim[0]
    r_dim = dim[1]

    tf.reset_default_graph()

    print('input layer')
    e_0 = get_input_layer(e_input)

    M, M_arr = get_sparse_tensor(e, KG)
    head, tail, head_r, tail_r, r_mat = rfunc(KG, e)
    r_0, adj_r_0 = get_dual_input(e_0, head, tail, head_r, tail_r)

    print('GAT layer')
    r_1 = add_self_att_layer(r_0, adj_r_0, tf.nn.relu, r_dim)
    e_1 = add_sparse_att_layer(e_0, r_1, r_mat, tf.nn.relu, e)
    e_2 = e_0 + alpha * e_1

    print('GCN layers')
    e_3 = gcn_layer(e_2, e_dim, M, act_func, dropout=0.0)
    e_4 = highway(e_2, e_3, e_dim)
    e_5 = gcn_layer(e_4, e_dim, M, act_func, dropout=0.0)
    output = highway(e_4, e_5, e_dim)
    loss = get_loss(output, train, gamma, k)
    return output, loss


def r_training(output_layer, loss, learning_rate, epochs, ILL, e, k, test):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    print('initializing...')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('running...')
    J = []
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    for i in range(epochs):
        if i % 50 == 0:
            out = sess.run(output_layer)
            neg2_left = get_neg(ILL[:, 1], out, k)
            neg_right = get_neg(ILL[:, 0], out, k)
            feeddict = {"neg_left:0": neg_left,
                        "neg_right:0": neg_right,
                        "neg2_left:0": neg2_left,
                        "neg2_right:0": neg2_right}

        _, th = sess.run([train_step, loss], feed_dict=feeddict)
        if i % 50 == 0:
            th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
            J.append(th)
            get_hits(outvec, test)

        print('%d/%d' % (i + 1, epochs), 'epochs...', th)
    outvec = sess.run(output_layer)
    sess.close()

    return outvec, J
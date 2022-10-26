from Layers import *


def get_sparse_adj(adj):
    one_arr = np.ones(len(adj[0]))
    du = np.dot(adj, one_arr.T)
    ind = []
    val = []
    for i in range(len(adj)):
        for j in range(len(adj[0])):
            if adj[i][j] != 0:
                ind.append((i, j))
                val.append(adj[i][j] / du[i])
    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[len(adj), len(adj[0])])

    return M


def get_a_loss(outlayer, ILL, gamma, k):
    print('getting loss...')
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left_1")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right_1")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left_1")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right_1")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))

    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


def a_build(dim, act_func, para, k, ILL, a_input, ae_adj, e, KG):
    dim = dim[2]
    gamma = para

    tf.reset_default_graph()

    print('input layer')

    a_0 = get_input_layer(a_input)

    adj_s = get_sparse_adj(ae_adj)

    print('GCN layers')

    a_1 = gcn_layer(a_0, dim, adj_s, act_func, dropout=0.0)

    adj, adj_arr = get_sparse_tensor(e, KG)

    a_2 = gcn_layer(a_1, dim, adj, act_func, dropout=0.0)

    a_h = highway(a_1, a_2, dim)

    loss = get_a_loss(a_h, ILL, gamma, k)

    return a_h, loss


def a_training(a_output, a_loss, learning_rate, epochs, ILL, k, test):
    train_step_2 = tf.train.AdamOptimizer(learning_rate).minimize(a_loss)
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
        if i % 100 == 0:
            out2 = sess.run(a_output)
            neg2_left = get_neg(ILL[:, 1], out2, k)
            neg_right = get_neg(ILL[:, 0], out2, k)
            feeddict_2 = {"neg_left_1:0": neg_left,
                          "neg_right_1:0": neg_right,
                          "neg2_left_1:0": neg2_left,
                          "neg2_right_1:0": neg2_right}

        _, th2 = sess.run([train_step_2, a_loss], feed_dict=feeddict_2)
        if i % 100 == 0:
            th2, a_outvec = sess.run([a_loss, a_output], feed_dict=feeddict_2)
            J.append(th2)
            print('AE')
            get_hits(a_outvec, test)

        print('%d/%d' % (i + 1, epochs), 'epochs...', th2)
    outvec = sess.run(a_output)
    sess.close()
    return outvec, J

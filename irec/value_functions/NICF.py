"""
Code taken mainly from: https://github.com/zoulixin93/NICF
Author: zoulixin93
"""
import numpy as np
from tqdm import tqdm
from .ExperimentalValueFunction import ExperimentalValueFunction
from collections import defaultdict
import tensorflow as tf
import copy
from collections import defaultdict
from typing import Any

MEMORYSIZE = 50000
BATCHSIZE = 128
THRESHOLD = 300

start = 0
end = 3000


def decay_function1(x):
    x = 50 + x
    return max(2.0 / (1 + np.power(x, 0.2)), 0.001)


START = decay_function1(start)
END = decay_function1(end)


def decay_function(x):
    x = max(min(end, x), start)
    return (decay_function1(x) - END) / (START - END + 0.0000001)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class basic_model(object):
    GRAPHS: Any = {}
    SESS: Any = {}
    SAVER: Any = {}
    CLUSTER: Any = None
    SERVER: Any = None

    def c_opt(self, learning_rate, name):
        if str(name).__contains__("adam"):
            print("adam")
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif str(name).__contains__("adagrad"):
            print("adagrad")
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif str(name).__contains__("sgd"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif str(name).__contains__("rms"):
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif str(name).__contains__("moment"):
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.1)
        return optimizer

    @classmethod
    def create_model(
        cls,
        config,
        variable_scope="target",
        trainable=True,
        graph_name="DEFAULT",
        task_index=0,
    ):
        jobs = config.jobs
        job = list(jobs.keys())[0]
        cls.CLUSTER = tf.train.ClusterSpec(jobs)
        cls.SERVER = tf.train.Server(
            cls.CLUSTER,
            job_name=job,
            task_index=task_index,
            config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
            ),
        )
        if not graph_name in cls.GRAPHS:
            cls.GRAPHS[graph_name] = tf.Graph()
        with cls.GRAPHS[graph_name].as_default():
            model = cls(config, variable_scope=variable_scope, trainable=trainable)
            if not graph_name in cls.SESS:
                cls.SESS[graph_name] = tf.compat.v1.Session(cls.SERVER.target)
                cls.SAVER[graph_name] = tf.compat.v1.train.Saver(max_to_keep=50)
            cls.SESS[graph_name].run(model.init)
        return {
            "graph": cls.GRAPHS[graph_name],
            "sess": cls.SESS[graph_name],
            "saver": cls.SAVER[graph_name],
            "model": model,
            "cluster": cls.CLUSTER,
            "server": cls.SERVER,
        }

    @classmethod
    def create_model_without_distributed(
        cls, config, variable_scope="target", trainable=True, graph_name="DEFAULT"
    ):
        # if not graph_name in cls.GRAPHS:
        cls.GRAPHS[graph_name] = tf.Graph()
        with cls.GRAPHS[graph_name].as_default():
            model = cls(config, variable_scope=variable_scope, trainable=trainable)
            # if not graph_name in cls.SESS:
            cls.SESS[graph_name] = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(
                    gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)
                )
            )
            cls.SAVER[graph_name] = tf.compat.v1.train.Saver(max_to_keep=50)
            cls.SESS[graph_name].run(model.init)
        return {
            "graph": cls.GRAPHS[graph_name],
            "sess": cls.SESS[graph_name],
            "saver": cls.SAVER[graph_name],
            "model": model,
        }

    def _update_placehoders(self):
        self.placeholders: Any = {"none": {}}
        raise NotImplemented

    def _get_feed_dict(self, task, data_dicts):
        place_holders = self.placeholders[task]
        res = {}
        for key, value in place_holders.items():
            res[value] = data_dicts[key]
        return res

    def __init__(self, args, variable_scope="target", trainable=True):
        print(self.__class__)
        self.args = args
        self.variable_scope = variable_scope
        self.trainable = trainable
        self.placeholders = {}
        self._build_model()

    def _build_model(self):
        with tf.compat.v1.variable_scope(self.variable_scope):
            self._create_placeholders()
            self._create_global_step()
            self._update_placehoders()
            self._create_inference()
            if self.trainable:
                self._create_optimizer()
            self._create_intializer()

    def _create_global_step(self):
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name="global_step"
        )

    def _create_intializer(self):
        with tf.name_scope("initlializer"):
            self.init = tf.compat.v1.global_variables_initializer()

    def _create_placeholders(self):
        raise NotImplementedError

    def _create_inference(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def chose_action(self, state, sess):
        raise NotImplementedError
        pass


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    """Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.compat.v1.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(
    inputs,
    vocab_size,
    num_units,
    zero_pad=True,
    scale=True,
    l2_reg=0.0,
    scope="embedding",
    with_t=False,
    reuse=None,
):
    """Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]
    ```
    """
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable(
            "lookup_table",
            dtype=tf.float32,
            shape=[vocab_size, num_units],
            # initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
        )
        if zero_pad:
            lookup_table = tf.concat(
                (tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0
            )
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
    if with_t:
        return outputs, lookup_table
    else:
        return outputs


def multihead_attention(
    queries,
    keys,
    num_units=None,
    num_heads=8,
    dropout_rate=0,
    is_training=True,
    causality=False,
    scope="multihead_attention",
    reuse=None,
    with_qk=False,
):
    """Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.compat.v1.layers.dense(
            queries, num_units, activation=None
        )  # (N, T_q, C)
        K = tf.compat.v1.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        V = tf.compat.v1.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.compat.v1.concat(
            tf.split(Q, num_heads, axis=2), axis=0
        )  # (h*N, T_q, C/h)
        K_ = tf.compat.v1.concat(
            tf.split(K, num_heads, axis=2), axis=0
        )  # (h*N, T_k, C/h)
        V_ = tf.compat.v1.concat(
            tf.split(V, num_heads, axis=2), axis=0
        )  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(
            tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]
        )  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-(2 ** 32) + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(
                diag_vals
            ).to_dense()  # (T_q, T_k)
            masks = tf.tile(
                tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]
            )  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-(2 ** 32) + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(
            tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]
        )  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.compat.v1.layers.dropout(
            outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training)
        )

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        # outputs = normalize(outputs) # (N, T_q, C)

    if with_qk:
        return Q, K
    else:
        return outputs


def feedforward(
    inputs,
    num_units=[2048, 512],
    scope="multihead_attention",
    dropout_rate=0.2,
    is_training=True,
    reuse=None,
):
    """Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {
            "inputs": inputs,
            "filters": num_units[0],
            "kernel_size": 1,
            "activation": tf.nn.relu,
            "use_bias": True,
        }
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(
            outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training)
        )
        # Readout layer
        params = {
            "inputs": outputs,
            "filters": num_units[1],
            "kernel_size": 1,
            "activation": None,
            "use_bias": True,
        }
        outputs = tf.compat.v1.layers.conv1d(**params)
        outputs = tf.compat.v1.layers.dropout(
            outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training)
        )

        # Residual connection
        outputs += inputs
    return outputs


class FA(basic_model):
    def _create_placeholders(self):
        self.utype = tf.compat.v1.placeholder(tf.int32, (None,), name="uid")
        self.p_rec = [
            tf.compat.v1.placeholder(
                tf.int32,
                (
                    None,
                    None,
                ),
                name="p" + str(i) + "_rec",
            )
            for i in range(6)
        ]
        self.pt = [
            tf.compat.v1.placeholder(tf.int32, (None, 2), "p" + str(i) + "t")
            for i in range(6)
        ]
        self.rec = tf.compat.v1.placeholder(tf.int32, (None,), name="iid")
        self.target = tf.compat.v1.placeholder(tf.float32, (None,), name="target")

    def _update_placehoders(self):
        self.placeholders["all"] = {
            "uid": self.utype,
            "iid": self.rec,
            "goal": self.target,
        }
        for i in range(6):
            self.placeholders["all"]["p" + str(i) + "_rec"] = self.p_rec[i]
            self.placeholders["all"]["p" + str(i) + "t"] = self.pt[i]
        self.placeholders["predict"] = {
            item: self.placeholders["all"][item]
            for item in ["uid"]
            + ["p" + str(i) + "_rec" for i in range(6)]
            + ["p" + str(i) + "t" for i in range(6)]
        }
        self.placeholders["optimize"] = self.placeholders["all"]

    def _create_inference(self):
        p_f = [
            tf.Variable(
                np.random.uniform(
                    -0.01, 0.01, (self.args.item_num, self.args.latent_factor)
                ),
                dtype=tf.float32,
                trainable=True,
                name="item" + str(i) + "_feature",
            )
            for i in range(6)
        ]
        u_f = tf.Variable(
            np.random.uniform(
                -0.01, 0.01, (self.args.utype_num, self.args.latent_factor)
            ),
            dtype=tf.float32,
            trainable=True,
            name="user_feature",
        )
        u_emb = tf.nn.embedding_lookup(u_f, self.utype)
        self.p_rec = [tf.transpose(item, [1, 0]) for item in self.p_rec]
        i_p_mask = [
            tf.expand_dims(tf.compat.v1.to_float(tf.not_equal(item, 0)), -1)
            for item in self.p_rec
        ]

        self.p_seq = [tf.nn.embedding_lookup(p_f[i], self.p_rec[i]) for i in range(6)]
        for iii, item in enumerate(self.p_seq):
            for i in range(self.args.num_blocks):
                with tf.compat.v1.variable_scope(
                    "rate_" + str(iii) + "_num_blocks_" + str(i)
                ):
                    item = multihead_attention(
                        queries=normalize(item),
                        keys=item,
                        num_units=self.args.latent_factor,
                        num_heads=self.args.num_heads,
                        dropout_rate=self.args.dropout_rate,
                        is_training=True,
                        causality=True,
                        scope="self_attention_pos_" + str(i),
                    )

                    item = feedforward(
                        normalize(item),
                        num_units=[self.args.latent_factor, self.args.latent_factor],
                        dropout_rate=self.args.dropout_rate,
                        is_training=True,
                        scope="feed_forward_pos_" + str(i),
                    )
                    item *= i_p_mask[iii]
        self.p_seq = [normalize(item) for item in self.p_seq]

        p_out = [
            tf.gather_nd(tf.transpose(self.p_seq[i], [1, 0, 2]), self.pt[i])
            for i in range(6)
        ]
        context = tf.concat(p_out, 1)
        hidden = tf.compat.v1.layers.dense(
            context, self.args.latent_factor, activation=tf.nn.relu
        )
        self.pi = tf.compat.v1.layers.dense(hidden, self.args.item_num, trainable=True)

    def _build_actor(self, context, name, trainable):
        with tf.compat.v1.variable_scope(name):
            a_prob = tf.layers.dense(context, self.args.item_num, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def _create_optimizer(self):
        a_indices = tf.stack(
            [tf.range(tf.shape(self.rec)[0], dtype=tf.int32), self.rec], axis=1
        )
        self.npi = tf.gather_nd(params=self.pi, indices=a_indices)
        self.loss = tf.losses.mean_squared_error(self.npi, self.target)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            self.args.learning_rate
        ).minimize(self.loss)

    def optimize_model(self, sess, data):
        feed_dicts = self._get_feed_dict("optimize", data)
        return sess.run([self.loss, self.npi, self.optimizer], feed_dicts)[:2]

    def predict(self, sess, data):
        feed_dicts = self._get_feed_dict("predict", data)
        return sess.run(self.pi, feed_dicts)


def convert_item_seq2matrix(item_seq):
    max_length = max([len(item) for item in item_seq])
    matrix = np.zeros((max_length, len(item_seq)), dtype=np.int32)
    for x, xx in enumerate(item_seq):
        for y, yy in enumerate(xx):
            matrix[y, x] = yy
    target_index = list(zip([len(i) - 1 for i in item_seq], range(len(item_seq))))
    return matrix, target_index


class NICF(ExperimentalValueFunction):
    """NICF.

    It is an interactive method based on a combination of neural networks and
    collaborative filtering that also performs a meta-learning of the user’s preferences [1]_.

    References
    ----------
    .. [1] Zhao, Xiaoxue, Weinan Zhang, and Jun Wang. "Interactive collaborative filtering."
       Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.
    """

    def __init__(
        self,
        time_step,
        latent_factor,
        learning_rate,
        training_epoch,
        rnn_layer,
        inner_epoch,
        batch,
        gamma,
        clip_param,
        restore_model,
        num_blocks,
        num_heads,
        dropout_rate,
        *args,
        **kwargs
    ):
        """__init__.

        Args:
            args:
            kwargs:
            time_step:
            latent_factor:
            learning_rate:
            training_epoch:
            rnn_layer:
            inner_epoch:
            batch:
            gamma:
            clip_param:
            restore_model:
            num_blocks:
            num_heads:
            dropout_rate:
        """

        super().__init__(*args, **kwargs)
        self.time_step: Any = time_step
        self.latent_factor: Any = latent_factor
        self.learning_rate: Any = learning_rate
        self.training_epoch: Any = training_epoch
        self.rnn_layer: Any = rnn_layer
        self.inner_epoch: Any = inner_epoch
        self.batch: Any = batch
        self.gamma: Any = gamma
        self.clip_param: Any = clip_param
        self.restore_model: Any = restore_model
        self.num_blocks: Any = num_blocks
        self.num_heads: Any = num_heads
        self.dropout_rate: Any = dropout_rate

    def reset_with_users(self, uid):
        self.state = [(uid, 1), []]
        self.short = {}
        return self.state

    def step(self, action):
        if action in self.rates[self.state[0][0]] and (not action in self.short):
            rate = self.rates[self.state[0][0]][action]
            if rate >= 4:
                reward = 1
            else:
                reward = 0
        else:
            rate = 0
            reward = 0

        if len(self.state[1]) < self.time_step - 1:
            done = False
        else:
            done = True
        self.short[action] = 1
        t = self.state[1] + [[action, reward, done]]
        info = rate
        self.state[1].append([action, reward, done, info])
        return self.state, reward, done, info

    def train_epoch(self, epoch):
        selected_users = np.random.choice(self.train_dataset.uids, (self.inner_epoch,))
        for uuid in selected_users:
            actions = {}
            done = False
            state = self.reset_with_users(uuid)
            while not done:
                data = {"uid": [state[0][1]]}
                for i in range(6):
                    p_r, pnt = convert_item_seq2matrix(
                        [[0] + [item[0] for item in state[1] if item[3] == i]]
                    )
                    data["p" + str(i) + "_rec"] = p_r
                    data["p" + str(i) + "t"] = pnt
                policy = self.fa["model"].predict(self.fa["sess"], data)[0]
                if np.random.random() < 5 * THRESHOLD / (THRESHOLD + self.tau):
                    policy = np.random.uniform(0, 1, (self.args.item_num,))
                for item in actions:
                    policy[item] = -np.inf
                action = np.argmax(policy[1:]) + 1
                s_pre = copy.deepcopy(state)
                state_next, rwd, done, info = self.step(action)
                self.memory.append(
                    [s_pre, action, rwd, done, copy.deepcopy(state_next)]
                )
                actions[action] = 1
                state = state_next

        if len(self.memory) >= BATCHSIZE:
            self.memory = self.memory[-MEMORYSIZE:]
            batch = [
                self.memory[item]
                for item in np.random.choice(range(len(self.memory)), (BATCHSIZE,))
            ]
            data = self.convert_batch2dict(batch, epoch)
            loss, _ = self.fa["model"].optimize_model(self.fa["sess"], data)
            self.tau += 5

    def reset(self, observation):
        """reset.

        Args:
            observation:
        """
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = copy.copy(train_dataset)
        self.train_dataset.data[:, 2]
        self.train_dataset.data[:, 2] = np.ceil(self.train_dataset.data[:, 2])
        self.train_dataset.data[:, 0] += 1
        self.train_dataset.data[:, 1] += 1
        self.train_dataset.data = self.train_dataset.data.astype(int)
        self.train_dataset.update_from_data()
        # self.train_consumption_matrix = scipy.sparse.csr_matrix((self.train_dataset.data[:,2],(self.train_dataset.data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.num_total_users,self.train_dataset.num_total_items))
        self.tau = 0

        args = Namespace(
            **{i: getattr(self, i) for i in dir(self)},
            item_num=self.train_dataset.num_total_items + 1,
            utype_num=self.train_dataset.num_total_users + 1
        )
        self.args = args
        self.fa = FA.create_model_without_distributed(args)
        self.memory = []
        self.rates: Any = defaultdict(dict)
        for i in range(len(self.train_dataset.data)):
            uid = int(self.train_dataset.data[i, 0])
            item = int(self.train_dataset.data[i, 1])
            reward = self.train_dataset.data[i, 2]
            self.rates[uid][item] = reward

        self.rates = dict(self.rates)

        for epoch in tqdm(range(self.training_epoch)):
            self.train_epoch(epoch)

        self.test_users_states = dict()

    def _update(self, uid, item, reward):
        pass

    def action_estimates(self, candidate_actions):
        """action_estimates.

        Args:
            candidate_actions: (user id, candidate_items)

        Returns:
            numpy.ndarray:
        """
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        uid += 1
        if uid not in self.test_users_states:
            self.test_users_states[uid] = [(uid, 1), []]
        state = self.test_users_states[uid]
        data = {"uid": [state[0][1]]}
        for i in range(6):
            p_r, pnt = convert_item_seq2matrix(
                [[0] + [item[0] for item in state[1] if item[3] == i]]
            )
            data["p" + str(i) + "_rec"] = p_r
            data["p" + str(i) + "t"] = pnt

        policy = self.fa["model"].predict(self.fa["sess"], data)[0]
        items_score = policy[1:][candidate_items]
        # for item in candidate_items:
        # policy[item] = -np.inf
        # items_score=np.random.rand(len(candidate_items))
        return items_score, None

    def update(self, observation, action, reward, info):
        """update.

        Args:
            observation:
            action: (user id, item)
            reward (float): reward
            info:
        """
        uid = action[0]
        item = action[1]
        additional_data = info
        uid += 1
        item += 1
        if uid not in self.test_users_states:
            self.test_users_states[uid] = [(uid, 1), []]
        state = self.test_users_states[uid]
        self.state = state

        action = item
        done = False
        rate = reward
        if rate >= 4:
            reward = 1
        else:
            reward = 0

        t = self.state[1] + [[action, reward, done]]
        info = rate
        self.state[1].append([action, reward, done, info])

    def convert_batch2dict(self, batch, epoch):
        uids = []
        pos_recs = {i: [] for i in range(6)}
        next_pos = {i: [] for i in range(6)}
        iids = []
        goals = []
        dones = []
        for item in batch:
            uids.append(item[0][0][1])
            ep = item[0][1]
            for xxx in range(6):
                pos_recs[xxx].append([0] + [j[0] for j in ep if j[3] == xxx])
            iids.append(item[1])
            goals.append(item[2])
            if item[3]:
                dones.append(0.0)
            else:
                dones.append(1.0)
            ep = item[4][1]
            for xxx in range(6):
                next_pos[xxx].append([0] + [j[0] for j in ep if j[3] == xxx])
        data = {"uid": uids}
        for xxx in range(6):
            p_r, pnt = convert_item_seq2matrix(next_pos[xxx])
            data["p" + str(xxx) + "_rec"] = p_r
            data["p" + str(xxx) + "t"] = pnt
        value = self.fa["model"].predict(self.fa["sess"], data)
        value[:, 0] = -500
        goals = (
            np.max(value, axis=-1)
            * np.asarray(dones)
            * min(self.args.gamma, decay_function(max(end - epoch, 0) + 1))
            + goals
        )
        data = {"uid": uids, "iid": iids, "goal": goals}
        for i in range(6):
            p_r, pnt = convert_item_seq2matrix(pos_recs[i])
            data["p" + str(i) + "_rec"] = p_r
            data["p" + str(i) + "t"] = pnt
        return data

    def precision(self, episode):
        return sum([i[1] for i in episode])

    def recall(self, episode, uid):
        return sum([i[1] for i in episode]) / len(self.rates[uid])

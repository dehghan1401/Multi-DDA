

import numpy as np
from utils import *
import scipy.io
import numpy as np
import scipy.sparse as sp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import random
from utils import *
from model import GCNModel
import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



drug_sim = np.loadtxt('data/drug_sim.csv', delimiter=',')
dis_sim = np.loadtxt('data/dis_sim.csv', delimiter=',')
drug_dis_matrix = np.loadtxt('data/drug_dis.csv', delimiter=',')
adj = constructHNet(drug_dis_matrix, drug_sim, dis_sim)
X = constructNet(drug_dis_matrix)
pathway = scipy.io.loadmat('data/Dataset_drug_features.mat')['pathway_feature_matrix']
epoch = 4000
emb_dim = 64
lr = 0.01
adjdp = 0.6
dp = 0.4
simw = 6


features = X
num_features = features.shape[1]
features_nonzero = features.shape[0]
adj_orig = drug_dis_matrix.copy()
import scipy.io
pathway = scipy.io.loadmat('data/Dataset_drug_features.mat')['pathway_feature_matrix']
enzyme = scipy.io.loadmat('data/Dataset_drug_features.mat')['enzyme_feature_matrix']
target = scipy.io.loadmat('data/Dataset_drug_features.mat')['target_feature_matrix']
structure = scipy.io.loadmat('data/Dataset_drug_features.mat')['structure_feature_matrix']

adj_norm = adj#preprocess_graph(adj)
adj_ = np.array(np.where(adj != 0))
adj_nonzero = adj_norm.shape[0]

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.Accuracy(name="acc")],
    )

    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
    )

    return history





class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gru":
            self.update_fn = create_gru(hidden_units, dropout_rate)
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)


class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges = inputs

        # Linearly transform node states
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # (1) Compute pair-wise attention scores
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)

        # (2) Normalize attention scores
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # (3) Gather node states of neighbors, apply attention scores and aggregate
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs

        # Obtain outputs from each attention head
        outputs = [
            attention_layer([atom_features, pair_indices])
            for attention_layer in self.attention_layers
        ]
        # Concatenate or average the node states from each head
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        # Activate and return node states
        return tf.nn.relu(outputs)

class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        enzyme = enzyme,
        pathway = pathway,
        structure = structure,
        target = target,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )

        self.conv3 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv3",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        #self.compute_logits = layers.Dense(units=num_classes, name="logits")
        self.dense_proj = layers.Dense(units = 867, name="lasto")

        self.enzyme = enzyme
        self.pathway = pathway
        self.structure = structure
        self.target = target

        self.pathway_1 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.enzyme_1 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.target_1 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.structure_1 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)

        self.pathway_2 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.enzyme_2 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.target_2 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.structure_2 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)

        self.pathway_3 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.enzyme_3 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.target_3 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)
        self.structure_3 = layers.Dense(hidden_units[0], activation = tf.nn.gelu)

        self.gal = MultiHeadGraphAttention(hidden_units, num_heads=3)

    def call(self, num_nodes):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        p_o = self.pathway_1(self.pathway)
        t_o = self.target_1(self.target)
        s_o = self.structure_1(self.structure)
        e_o = self.enzyme_1(self.enzyme)

        n_o = layers.Add()([p_o,t_o,s_o,e_o])
        n_o = tf.pad(n_o, [[0, 867-269], [0, 0]])
        # Skip connection.
        x = x1 + x + n_o
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))

        p_o = self.pathway_2(self.pathway)
        t_o = self.target_2(self.target)
        s_o = self.structure_2(self.structure)
        e_o = self.enzyme_2(self.enzyme)
        # Skip connection.
        n_o = layers.Add()([p_o,t_o,s_o,e_o])
        n_o = tf.pad(n_o, [[0, 867-269], [0, 0]])
        # Skip connection.
        x = x2 + x + n_o

        x3 = self.conv2((x, self.edges, self.edge_weights))

        p_o = self.pathway_3(self.pathway)
        t_o = self.target_3(self.target)
        s_o = self.structure_3(self.structure)
        e_o = self.enzyme_3(self.enzyme)
        # Skip connection.
        n_o = layers.Add()([p_o,t_o,s_o,e_o])
        n_o = tf.pad(n_o, [[0, 867-269], [0, 0]])
        # Skip connection.
        x = x3 + x + n_o
        #x = self.gal(x)
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = self.dense_proj(x)
        # Compute logits
        return node_embeddings

adj_ = np.array(np.where(adj != 0))

index_matrix = np.mat(np.where(adj == 1))
association_nam = index_matrix.shape[1]
random_index = index_matrix.T.tolist()
random.seed(0.1)
random.shuffle(random_index)
k_folds = 5
CV_size = int(association_nam / k_folds)
temp = np.array(random_index[:association_nam - association_nam %
                              k_folds]).reshape(k_folds, CV_size,  -1).tolist()
temp[k_folds - 1] = temp[k_folds - 1] + \
    random_index[association_nam - association_nam % k_folds:]
random_index = temp
metric = np.zeros((1, 7))
print("seed=%d, evaluating drug-disease...." % (0.1))
for k in range(k_folds):
    print("------this is %dth cross validation------" % (k+1))
    adj_cv = np.matrix(adj, copy=True)
    adj_cv[tuple(np.array(random_index[k]).T)] = 0

    adj_ = np.array(np.where(adj_cv != 0))
    edge_weights = tf.ones(shape=adj_.shape[1])
    graph_info = (features, adj_, edge_weights)

    # set these hyperparameters to propoer values
    hidden_units = [128, 128]
    learning_rate = 0.0001
    dropout_rate = 0.5
    num_epochs = 10000
    batch_size = 867

    gnn_model = GNNNodeClassifier(
        graph_info=graph_info,
        num_classes=1,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        name="gnn_model",
        enzyme = enzyme,
        pathway = pathway,
        structure = structure,
        target = target,
    )

    gnn_model.summary()
    history = run_experiment(gnn_model, adj_norm, adj_norm)
    out = gnn_model.predict(adj_norm)
    out_final = out[0:269,269:867]
    out_real = adj[0:269,269:867]
    print(24276/867)

    a = out_real
    b = out_final
    from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
    for i in range(0,b.shape[0]):
      b[i,:] = b[i,:]/np.sum(b[i,:])
    print('accuracy:', (np.sum(a == (b>0.5)))/(b.shape[0]*b.shape[1]))

    precision, recall, _ = precision_recall_curve(np.reshape(a, (-1,1)), np.reshape(b, (-1,1)))

    # Compute AUPR
    aupr = auc(recall, precision)

    # Compute false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(np.reshape(a, (-1,1)), np.reshape(b, (-1,1)))

    # Compute AUC
    auc_value = roc_auc_score(np.reshape(a, (-1,1)), np.reshape(b, (-1,1)))

    print("AUPR: ", aupr)
    print("AUC: ", auc_value)

zero_indices = np.argwhere(a == 0)

# Step 2: Extract corresponding elements from matrix B
b_elements = b[zero_indices[:, 0], zero_indices[:, 1]]

# Step 3: Sort the extracted elements in descending order
sorted_b_elements = np.sort(b_elements)[::-1]

# Step 4: Get the indices of the sorted elements
# First, get the indices of the sorted elements within the subset of b_elements
sorted_indices_in_subset = np.argsort(b_elements)[::-1]

# Then, use these indices to get the original indices in matrix B
original_indices = zero_indices[sorted_indices_in_subset]

# Print the results
print("Sorted elements from matrix B:", sorted_b_elements)
print("Indices of the sorted elements in matrix B:\n", original_indices)

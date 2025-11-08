%cd /content/drive/My Drive/DrugDisease/LAGCN/code
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import pandas as pd
from utils import constructHNet, constructNet

# Model configuration parameters
CONFIG = {
    'emb_dim': 256,           # Embedding dimension
    'dp': 0.4,                # Dropout probability
    'k_folds': 5,             # Number of cross-validation folds
    'hidden_units': [256, 128], # Hidden layer sizes in GNN
    'learning_rate': 0.0005,  # Learning rate for optimizer
    'dropout_rate': 0.4,      # Dropout rate for regularization
    'num_epochs': 10000,      # Maximum training epochs
    'batch_size': 1,          # Batch size (1 for whole-graph training)
    'patience': 50            # Early stopping patience
}

class DataLoader:
    """
    Data loading and preprocessing utility for drug-disease association data
    Handles loading of similarity matrices, drug features, and normalization
    """
    
    def __init__(self):
        # Initialize data attributes
        self.drug_sim = None      # Drug similarity matrix
        self.dis_sim = None       # Disease similarity matrix  
        self.drug_dis_matrix = None  # Drug-disease association matrix
        self.pathway = None       # Drug pathway features
        self.enzyme = None        # Drug enzyme features
        self.target = None        # Drug target features
        self.structure = None     # Drug structure features

    def load_data(self):
        """Load all required datasets including similarity matrices and drug features"""
        # Load drug and disease similarity matrices
        self.drug_sim = np.loadtxt(r'../data/drug_sim.csv', delimiter=',')
        self.dis_sim = np.loadtxt(r'../data/dis_sim.csv', delimiter=',')
        self.drug_dis_matrix = np.loadtxt(r'../data/drug_dis.csv', delimiter=',')

        # Load multi-view drug features from matlab file
        drug_features = scipy.io.loadmat(r'/content/drive/My Drive/DrugDisease/SCMFDD/SCMFDD_Dataset.mat')
        self.pathway = drug_features['pathway_feature_matrix']
        self.enzyme = drug_features['enzyme_feature_matrix']
        self.target = drug_features['target_feature_matrix']
        self.structure = drug_features['structure_feature_matrix']

        # Normalize all features to [0,1] range
        self._normalize_features()

    def _normalize_features(self):
        """Normalize feature matrices to [0, 1] range for stable training"""
        features = {
            'pathway': self.pathway,
            'enzyme': self.enzyme,
            'target': self.target,
            'structure': self.structure
        }

        for feature_name, feature in features.items():
            if feature is not None:
                feature_min = np.min(feature)
                feature_max = np.max(feature)
                if feature_max > feature_min:
                    # Min-max normalization
                    normalized_feature = (feature - feature_min) / (feature_max - feature_min)
                    setattr(self, feature_name, normalized_feature.astype(np.float32))

    def prepare_graph_data(self):
        """
        Prepare graph data for GNN training
        Returns:
            adj: Heterogeneous network adjacency matrix
            x: Initial node features
        """
        # Construct heterogeneous network combining drug-drug, disease-disease, and drug-disease relations
        adj = constructHNet(self.drug_dis_matrix, self.drug_sim, self.dis_sim)
        # Construct initial node features from drug-disease matrix
        x = constructNet(self.drug_dis_matrix)
        return adj, x


def create_ffn(hidden_units, dropout_rate, name=None):
    """
    Create a Feed Forward Network with batch normalization and dropout
    
    Args:
        hidden_units: List of layer sizes
        dropout_rate: Dropout rate for regularization
        name: Name for the sequential model
    
    Returns:
        Sequential model with specified architecture
    """
    fnn_layers = []
    for units in hidden_units:
        fnn_layers.extend([
            layers.BatchNormalization(),  # Normalize activations
            layers.Dropout(dropout_rate), # Regularization
            layers.Dense(units, activation=tf.nn.gelu)  # GELU activation for better performance
        ])
    return keras.Sequential(fnn_layers, name=name)


class GraphConvLayer(layers.Layer):
    """
    Graph Convolution Layer for message passing and node representation learning
    Implements neighborhood aggregation and feature transformation
    """
    
    def __init__(self, hidden_units, dropout_rate=0.2, aggregation_type="mean",
                 combination_type="add", normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregation_type = aggregation_type  # How to aggregate neighbor messages
        self.combination_type = combination_type  # How to combine with self features
        self.normalize = normalize  # Whether to L2 normalize outputs
        
        # MLP for transforming neighbor messages
        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        # MLP for updating node representations
        self.update_fn = create_ffn(hidden_units, dropout_rate)

    def call(self, inputs):
        """Execute graph convolution operation"""
        node_representations, edges, edge_weights = inputs
        node_indices, neighbour_indices = edges[0], edges[1]

        # Gather representations of neighboring nodes
        neighbour_representations = tf.gather(node_representations, neighbour_indices)

        # Transform neighbor features through MLP
        neighbour_messages = self.ffn_prepare(neighbour_representations)
        
        # Apply edge weights if provided (for weighted graphs)
        if edge_weights is not None:
            neighbour_messages *= tf.expand_dims(edge_weights, -1)

        # Aggregate messages from neighbors
        aggregated_message = None
        num_nodes = tf.shape(node_representations)[0]
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

        # Combine aggregated neighbor information with self information
        if self.combination_type == "concat":
            h = tf.concat([node_representations, aggregated_message], axis=1)
        else:  # "add" - residual connection
            h = node_representations + aggregated_message

        # Transform combined features through update MLP
        node_embeddings = self.update_fn(h)

        # Optional L2 normalization for stable training
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings


class FeatureProcessor:
    """
    Process multi-view drug features and project them to consistent dimensions
    Handles pathway, enzyme, target, and structure features
    """
    
    def __init__(self, pathway, enzyme, target, structure, hidden_units):
        # Store all feature matrices
        self.features = {
            'pathway': pathway.astype(np.float32) if pathway is not None else None,
            'enzyme': enzyme.astype(np.float32) if enzyme is not None else None,
            'target': target.astype(np.float32) if target is not None else None,
            'structure': structure.astype(np.float32) if structure is not None else None
        }
        self.hidden_units = hidden_units
        self.output_dim = hidden_units[-1]  # Target output dimension
        self._build_layers()

    def _build_layers(self):
        """Build projection layers for each feature type to ensure consistent dimensions"""
        for feature_name in self.features.keys():
            if self.features[feature_name] is not None:
                # Create a dense layer to project each feature to target dimension
                setattr(self, f'{feature_name}_proj',
                       layers.Dense(self.output_dim, activation=tf.nn.gelu))

    def process_features(self):
        """Process and combine all drug features into a unified representation"""
        processed_features = []

        # Project each feature type to common dimension
        for feature_name, feature in self.features.items():
            if feature is not None:
                proj_layer = getattr(self, f'{feature_name}_proj')
                projected = proj_layer(feature)
                processed_features.append(projected)

        if processed_features:
            # Average all processed features for fusion
            combined = tf.reduce_mean(processed_features, axis=0)
            # Pad to match graph size (867 nodes - 269 drugs + 598 diseases)
            padded = tf.pad(combined, [[0, 867 - tf.shape(combined)[0]], [0, 0]])
            return padded
        else:
            # Return zeros if no features available
            return tf.zeros((867, self.output_dim))


class ImprovedGNNNodeClassifier(tf.keras.Model):
    """
    Main GNN model for drug-disease association prediction
    Combines graph convolution with multi-view feature processing
    """
    
    def __init__(self, graph_info, hidden_units, dropout_rate=0.2,
                 pathway=None, enzyme=None, structure=None, target=None, **kwargs):
        super().__init__(**kwargs)

        # Unpack graph structure information
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features  # Initial node features
        self.edges = edges                  # Graph edges (adjacency list)
        self.edge_weights = edge_weights    # Edge weights for message passing

        # Normalize edge weights for stable training
        if edge_weights is not None:
            self.edge_weights = edge_weights / tf.reduce_sum(edge_weights)
        else:
            self.edge_weights = tf.ones(shape=edges.shape[1])

        # Preprocessing MLP for initial node features
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")

        # Stack multiple graph convolution layers
        self.conv_layers = []
        for i in range(2):
            self.conv_layers.append(
                GraphConvLayer(
                    hidden_units,
                    dropout_rate,
                    combination_type="add",  # Use residual connections
                    name=f"graph_conv_{i + 1}"
                )
            )

        # Feature processor for multi-view drug features
        self.feature_processor = FeatureProcessor(
            pathway, enzyme, target, structure, hidden_units
        )

        # Feature fusion layer to combine graph and feature information
        self.feature_fusion = layers.Dense(hidden_units[-1], activation='sigmoid')

        # Postprocessing MLP after graph convolutions
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")

        # Final layers for prediction
        self.dropout = layers.Dropout(dropout_rate)
        self.batch_norm = layers.BatchNormalization()
        # Final projection to prediction scores (867 nodes)
        self.dense_proj = layers.Dense(units=867, activation='sigmoid', name="last")

        # Mark model as built to prevent warnings
        self.built = True

    def build(self, input_shape=None):
        """Properly build the model to fix TensorFlow warnings"""
        self.built = True

    def call(self, inputs=None):
        """Forward pass of the GNN model"""
        # Preprocess initial node features
        x = self.preprocess(self.node_features)

        # Process and fuse multi-view drug features
        processed_features = self.feature_processor.process_features()
        feature_weights = self.feature_fusion(processed_features)
        weighted_features = processed_features * feature_weights

        # Apply graph convolution layers with residual connections
        for i, conv_layer in enumerate(self.conv_layers):
            # Graph convolution: message passing and aggregation
            x_conv = conv_layer((x, self.edges, self.edge_weights))

            # Residual connection with feature fusion and dropout
            x_residual = x_conv + x + self.dropout(weighted_features)
            x = self.batch_norm(x_residual)

        # Postprocess final node representations
        x = self.postprocess(x)
        x = self.batch_norm(x)
        # Final prediction scores for all node pairs
        return self.dense_proj(x)


def masked_binary_crossentropy(mask):
    """
    Create a masked loss function that only computes loss for specified elements
    
    Args:
        mask: Boolean mask indicating which elements to include in loss calculation
    
    Returns:
        Loss function that ignores masked-out elements
    """
    def loss(y_true, y_pred):
        # Remove batch dimension if present (for whole-graph training)
        if len(y_true.shape) == 3 and y_true.shape[0] == 1:
            y_true = tf.squeeze(y_true, axis=0)
        if len(y_pred.shape) == 3 and y_pred.shape[0] == 1:
            y_pred = tf.squeeze(y_pred, axis=0)

        # Convert mask to tensor
        mask_tensor = tf.constant(mask, dtype=tf.float32)

        # Compute focal crossentropy for handling class imbalance
        bce = tf.keras.losses.BinaryFocalCrossentropy()(y_true, y_pred)

        # Apply mask and compute mean only over non-masked elements
        masked_bce = bce * mask_tensor
        return tf.reduce_sum(masked_bce) / tf.reduce_sum(mask_tensor)

    return loss


def create_sample_weights(y_train, mask, positive_weight=2.0):
    """
    Create sample weights to handle class imbalance in drug-disease associations
    
    Args:
        y_train: Training labels
        mask: Boolean mask for valid positions
        positive_weight: Extra weight for positive samples
    
    Returns:
        Sample weights array with higher weights for positive samples
    """
    # Count positive and negative samples in the masked region
    masked_y = y_train[mask]
    positive_count = np.sum(masked_y == 1)
    negative_count = np.sum(masked_y == 0)

    # Calculate class weights using balanced approach
    total_count = positive_count + negative_count
    weight_for_0 = total_count / (2 * negative_count) if negative_count > 0 else 1.0
    weight_for_1 = (total_count / (2 * positive_count)) * positive_weight if positive_count > 0 else 1.0

    # Create sample weights array
    sample_weights = np.ones_like(y_train, dtype=np.float32)
    sample_weights[y_train == 0] = weight_for_0
    sample_weights[y_train == 1] = weight_for_1

    # Apply mask to focus only on relevant positions
    sample_weights = sample_weights * mask

    return sample_weights


def run_experiment(model, x_train, y_train, mask):
    """
    Compile and train the GNN model with enhanced training strategy
    
    Args:
        model: GNN model to train
        x_train: Input features
        y_train: Target labels  
        mask: Mask for loss calculation
    
    Returns:
        Training history
    """
    # Learning rate scheduler for adaptive learning
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=CONFIG['learning_rate'],
        decay_steps=1000,
        decay_rate=0.9
    )

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10000,
        restore_best_weights=True,
        verbose=1
    )

    # Compile model with optimizer and masked loss
    model.compile(
        optimizer=keras.optimizers.Adam(lr_schedule),
        loss=masked_binary_crossentropy(mask),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )

    # Create sample weights for imbalanced classification
    sample_weights = create_sample_weights(y_train, mask, positive_weight=2.0)
    sample_weights_expanded = np.expand_dims(sample_weights, axis=0)

    print(f"Sample weights - Min: {np.min(sample_weights[mask]):.2f}, Max: {np.max(sample_weights[mask]):.2f}")
    print(f"Positive samples: {np.sum(y_train[mask] == 1)}, Negative samples: {np.sum(y_train[mask] == 0)}")

    # For graph-level training, use batch_size=1 and treat whole graph as one batch
    # Add batch dimension to inputs for compatibility
    x_train_expanded = np.expand_dims(x_train, axis=0)
    y_train_expanded = np.expand_dims(y_train, axis=0)

    # Train the model
    return model.fit(
        x=x_train_expanded,
        y=y_train_expanded,
        sample_weight=sample_weights_expanded,
        epochs=CONFIG['num_epochs'],
        batch_size=CONFIG['batch_size'],  # Whole graph in one batch
        callbacks=[early_stopping],       # Early stopping only
        verbose=1,
        shuffle=False  # Don't shuffle graph data
    )


def evaluate_model(predictions, ground_truth):
    """
    Comprehensive model evaluation using multiple metrics
    
    Args:
        predictions: Model predictions
        ground_truth: True labels
    
    Returns:
        accuracy, aupr, auc_value: Evaluation metrics
    """
    a = ground_truth
    b = predictions

    # Calculate accuracy based on threshold 0.5
    try:
        accuracy = np.sum(a == (b >= 0.5)) / (b.shape[0] * b.shape[1])
        # Flatten for metric calculation
        a_flat = a.reshape(-1, 1)
        b_flat = b.reshape(-1, 1)
    except:
        accuracy = np.sum(a == (b >= 0.5)) / (b.shape[0])
        a_flat = a
        b_flat = b

    # Calculate Area Under Precision-Recall Curve (AUPR)
    precision, recall, _ = precision_recall_curve(a_flat, b_flat)
    aupr = auc(recall, precision)
    
    # Calculate Area Under ROC Curve (AUC)
    auc_value = roc_auc_score(a_flat, b_flat)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"AUC: {auc_value:.4f}")

    return accuracy, aupr, auc_value


def prepare_cross_validation_data(adj, k_folds=5, seed=0.1):
    """
    Prepare data for k-fold cross validation
    
    Args:
        adj: Adjacency matrix with known associations
        k_folds: Number of cross-validation folds
        seed: Random seed for reproducibility
    
    Returns:
        folds: List of test sets for each fold
    """
    # Get indices of known associations
    index_matrix = np.where(adj == 1)
    association_nam = len(index_matrix[0])

    # Create shuffled indices for cross-validation
    random_index = list(zip(index_matrix[0], index_matrix[1]))
    random.seed(seed)
    random.shuffle(random_index)

    # Split into k folds
    cv_size = association_nam // k_folds
    folds = []
    for i in range(k_folds):
        start_idx = i * cv_size
        end_idx = start_idx + cv_size if i < k_folds - 1 else association_nam
        folds.append(random_index[start_idx:end_idx])

    return folds


def debug_model_training(model, x_train, y_train, mask):
    """
    Debug function to check model training setup and gradient flow
    
    Args:
        model: Model to debug
        x_train: Training features
        y_train: Training labels
        mask: Loss mask
    
    Returns:
        Boolean indicating if model setup is successful
    """
    # Check model structure
    model.build(input_shape=(1,) + x_train.shape)
    model.summary()

    # Test forward pass with batch dimension
    print("Testing forward pass...")
    try:
        x_train_batch = np.expand_dims(x_train, axis=0)
        y_train_batch = np.expand_dims(y_train, axis=0)

        with tf.GradientTape() as tape:
            predictions = model(x_train_batch)
            loss = masked_binary_crossentropy(mask)(y_train_batch, predictions)

        print(f"Initial loss: {loss:.4f}")
        print(f"Input shape: {x_train_batch.shape}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Target shape: {y_train_batch.shape}")
        print(f"Predictions range: [{tf.reduce_min(predictions):.4f}, {tf.reduce_max(predictions):.4f}]")

        # Check gradients for all trainable variables
        gradients = tape.gradient(loss, model.trainable_variables)
        print("Gradient analysis:")
        for grad, var in zip(gradients, model.trainable_variables):
            if grad is not None:
                grad_norm = tf.norm(grad).numpy()
                print(f"Variable: {var.name}, Gradient norm: {grad_norm:.6f}")
            else:
                print(f"Variable: {var.name}, No gradient")
        return True
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_drug_disease_names():
    """
    Load drug and disease names for interpretable predictions
    
    Returns:
        drug_names, disease_names: Lists of drug and disease names
    """
    try:
        # Load drug names from CSV file
        drug_names_df = pd.read_csv(r'../data/drug_names.csv', header=None)
        drug_names = drug_names_df[0].tolist()
    except:
        # Fallback to generic names if file not found
        drug_names = [f"Drug_{i}" for i in range(1, 270)]
    
    try:
        # Load disease names from CSV file  
        disease_names_df = pd.read_csv(r'../data/disease_names.csv', header=None)
        disease_names = disease_names_df[0].tolist()
    except:
        # Fallback to generic names if file not found
        disease_names = [f"Disease_{i}" for i in range(1, 599)]
    
    return drug_names, disease_names


def print_top_predictions(predictions, test_indices, drug_names, disease_names, top_k=100):
    """
    Print top predicted drug-disease associations from test set
    
    Args:
        predictions: Model predictions
        test_indices: Indices of test samples
        drug_names: List of drug names
        disease_names: List of disease names
        top_k: Number of top predictions to display
    
    Returns:
        Top k predictions with scores
    """
    print(f"\n{'='*80}")
    print(f"TOP {top_k} PREDICTED DRUG-DISEASE ASSOCIATIONS")
    print(f"{'='*80}")
    
    # Extract test predictions with indices
    test_scores = []
    for i, (drug_idx, disease_idx) in enumerate(zip(test_indices[0], test_indices[1])):
        # Only consider drug-disease pairs (not drug-drug or disease-disease)
        if drug_idx < 269 and disease_idx >= 269:
            score = predictions[drug_idx, disease_idx]
            test_scores.append((drug_idx, disease_idx, score))
    
    # Sort by prediction score (descending)
    test_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Print top predictions in formatted table
    print(f"{'Rank':<6} {'Drug':<30} {'Disease':<40} {'Score':<8}")
    print(f"{'-'*6} {'-'*30} {'-'*40} {'-'*8}")
    
    for rank, (drug_idx, disease_idx, score) in enumerate(test_scores[:top_k], 1):
        drug_name = drug_names[drug_idx] if drug_idx < len(drug_names) else f"Drug_{drug_idx+1}"
        disease_name = disease_names[disease_idx-269] if (disease_idx-269) < len(disease_names) else f"Disease_{disease_idx-268}"
        
        print(f"{rank:<6} {drug_name:<30} {disease_name:<40} {score:.6f}")
    
    return test_scores[:top_k]


def print_all_test_predictions(predictions, ground_truth, test_indices, drug_names, disease_names):
    """
    Print all predictions on the test set with actual vs predicted values
    
    Args:
        predictions: Model predictions
        ground_truth: True labels
        test_indices: Indices of test samples
        drug_names: List of drug names
        disease_names: List of disease names
    
    Returns:
        Test set accuracy
    """
    print(f"\n{'='*80}")
    print("ALL TEST SET PREDICTIONS")
    print(f"{'='*80}")
    
    print(f"{'Drug':<30} {'Disease':<40} {'Predicted':<10} {'Actual':<8} {'Correct':<8}")
    print(f"{'-'*30} {'-'*40} {'-'*10} {'-'*8} {'-'*8}")
    
    correct_predictions = 0
    total_predictions = 0
    
    # Evaluate each test sample
    for i, (drug_idx, disease_idx) in enumerate(zip(test_indices[0], test_indices[1])):
        # Only consider drug-disease pairs
        if drug_idx < 269 and disease_idx >= 269:
            pred_score = predictions[drug_idx, disease_idx]
            actual_value = ground_truth[drug_idx, disease_idx]
            predicted_class = 1 if pred_score >= 0.5 else 0
            is_correct = predicted_class == actual_value
            
            drug_name = drug_names[drug_idx] if drug_idx < len(drug_names) else f"Drug_{drug_idx+1}"
            disease_name = disease_names[disease_idx-269] if (disease_idx-269) < len(disease_names) else f"Disease_{disease_idx-268}"
            
            print(f"{drug_name:<30} {disease_name:<40} {pred_score:.6f}   {actual_value:<8} {is_correct:<8}")
            
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
    
    # Calculate and print test accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nTest Set Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions} correct)")
    
    return accuracy


def save_predictions(predictions, drug_names, disease_names, filename="drug_disease_predictions.csv"):
    """
    Save all predictions to CSV file for further analysis
    
    Args:
        predictions: Model predictions
        drug_names: List of drug names
        disease_names: List of disease names
        filename: Output filename
    
    Returns:
        DataFrame containing all predictions
    """
    results = []
    # Iterate through all possible drug-disease pairs
    for drug_idx in range(269):
        for disease_idx in range(269, 867):
            drug_name = drug_names[drug_idx] if drug_idx < len(drug_names) else f"Drug_{drug_idx+1}"
            disease_name = disease_names[disease_idx-269] if (disease_idx-269) < len(disease_names) else f"Disease_{disease_idx-268}"
            score = predictions[drug_idx, disease_idx]
            
            results.append({
                'Drug': drug_name,
                'Disease': disease_name, 
                'Prediction_Score': score,
                'Drug_Index': drug_idx,
                'Disease_Index': disease_idx-269
            })
    
    # Create DataFrame and sort by prediction score
    df = pd.DataFrame(results)
    df = df.sort_values('Prediction_Score', ascending=False)
    df.to_csv(filename, index=False)
    print(f"\nPredictions saved to {filename}")
    
    return df


def print_top_novel_predictions(predictions, ground_truth, drug_names, disease_names, top_k=100):
    """
    Print top novel predictions (not in original training data)
    
    Args:
        predictions: Model predictions
        ground_truth: Original association matrix
        drug_names: List of drug names
        disease_names: List of disease names
        top_k: Number of top novel predictions to display
    
    Returns:
        Top k novel predictions
    """
    print(f"\n{'='*80}")
    print(f"TOP {top_k} NOVEL DRUG-DISEASE PREDICTIONS")
    print(f"{'='*80}")
    
    # Collect all novel predictions (where ground truth is 0)
    novel_predictions = []
    for drug_idx in range(269):  # Drugs are indices 0-268
        for disease_idx in range(269, 867):  # Diseases are indices 269-866
            # Only include novel predictions (not in original ground truth)
            if ground_truth[drug_idx, disease_idx] == 0:
                score = predictions[drug_idx, disease_idx]
                novel_predictions.append((drug_idx, disease_idx, score))
    
    # Sort by prediction score (descending)
    novel_predictions.sort(key=lambda x: x[2], reverse=True)
    
    # Print top novel predictions in formatted table
    print(f"{'Rank':<6} {'Drug':<30} {'Disease':<40} {'Score':<8}")
    print(f"{'-'*6} {'-'*30} {'-'*40} {'-'*8}")
    
    for rank, (drug_idx, disease_idx, score) in enumerate(novel_predictions[:top_k], 1):
        drug_name = drug_names[drug_idx] if drug_idx < len(drug_names) else f"Drug_{drug_idx+1}"
        disease_name = disease_names[disease_idx-269] if (disease_idx-269) < len(disease_names) else f"Disease_{disease_idx-268}"
        
        print(f"{rank:<6} {drug_name:<30} {disease_name:<40} {score:.6f}")
    
    print(f"\nTotal novel predictions found: {len(novel_predictions)}")
    print(f"Showing top {min(top_k, len(novel_predictions))} novel predictions")
    
    return novel_predictions[:top_k]


def main():
    """
    Main function to execute the complete drug-disease association prediction pipeline
    Includes data loading, model training, evaluation, and prediction analysis
    """
    # Load and prepare data
    print("Loading drug-disease association data...")
    data_loader = DataLoader()
    data_loader.load_data()
    adj, x = data_loader.prepare_graph_data()

    # Load drug and disease names for interpretable results
    drug_names, disease_names = get_drug_disease_names()

    # Prepare cross-validation folds
    print("Preparing cross-validation folds...")
    folds = prepare_cross_validation_data(adj, CONFIG['k_folds'])

    # Store metrics across folds
    metrics = []

    print("Starting cross-validation evaluation...")
    for k in range(min(2, CONFIG['k_folds'])):  # Test with fewer folds first for debugging
        print(f"------ {k + 1}th cross validation ------")

        # Create masked adjacency matrix for current fold
        adj_cv = adj.copy()
        test_indices = np.array(folds[k]).T
        adj_cv[test_indices[0], test_indices[1]] = 0  # Mask test associations

        # Prepare graph data for GNN
        adj_indices = np.where(adj_cv != 0)
        edge_weights = tf.ones(shape=len(adj_indices[0]))
        graph_info = (x, adj_indices, edge_weights)

        # Create GNN model with multi-view features
        print("Creating GNN model...")
        gnn_model = ImprovedGNNNodeClassifier(
            graph_info=graph_info,
            hidden_units=CONFIG['hidden_units'],
            dropout_rate=CONFIG['dropout_rate'],
            pathway=data_loader.pathway,
            enzyme=data_loader.enzyme,
            structure=data_loader.structure,
            target=data_loader.target
        )

        # Create mask for drug-disease pairs only (ignore drug-drug and disease-disease)
        mask_1 = np.zeros((867, 867), dtype=bool)
        mask_1[0:269, 269:867] = 1  # Only drug-disease pairs

        # Debug model training setup
        print("Debugging model setup...")
        if not debug_model_training(gnn_model, x, adj_cv, mask_1):
            print("Skipping this fold due to errors...")
            continue

        # Train model
        print("Starting model training...")
        try:
            history = run_experiment(gnn_model, x, adj_cv, mask_1)

            # Generate predictions using trained model
            predictions = gnn_model(None).numpy()

            # Remove batch dimension if present
            if len(predictions.shape) == 3 and predictions.shape[0] == 1:
                predictions = np.squeeze(predictions, axis=0)

            # Print detailed prediction analysis
            print("\n" + "="*100)
            print("DETAILED PREDICTION ANALYSIS")
            print("="*100)

            # Print all test set predictions with evaluation
            test_accuracy = print_all_test_predictions(predictions, adj, test_indices, drug_names, disease_names)

            # Print top 100 predictions from test set
            # top_predictions = print_top_predictions(predictions, test_indices, drug_names, disease_names, top_k=100)

            # Show top 100 NOVEL predictions only (not in original data)
            # top_novel_predictions = print_top_novel_predictions(predictions, adj, drug_names, disease_names, top_k=100)

            # Save predictions to CSV file for further analysis
            # predictions_df = save_predictions(predictions, drug_names, disease_names, filename=f"fold_{k+1}_predictions.csv")

            # Evaluate model using standard metrics
            print("Model Evaluation Results:")
            accuracy, aupr, auc_value = evaluate_model(predictions, adj)
            metrics.append((accuracy, aupr, auc_value))

            # Also evaluate on drug-disease submatrix only
            print("Drug-Disease Submatrix Evaluation:")
            out_final = predictions[0:269, 269:867]
            out_real = adj[0:269, 269:867]
            evaluate_model(out_final, out_real)

        except Exception as e:
            print(f"Error during training or evaluation: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print average metrics across all folds
    if metrics:
        avg_metrics = np.mean(metrics, axis=0)
        print(f"\nAverage Cross-Validation Metrics:")
        print(f"Accuracy: {avg_metrics[0]:.4f}")
        print(f"AUPR: {avg_metrics[1]:.4f}") 
        print(f"AUC: {avg_metrics[2]:.4f}")
    else:
        print("\nNo successful runs to average.")


if __name__ == "__main__":
    main()
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Concatenate, Dropout, BatchNormalization, Layer
from tensorflow.keras.models import Model
from keras_cv.losses import FocalLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from tensorflow.keras import saving
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.regularizers import l1_l2

# 1. Charger les données
data = np.load('lol_draft_data_v2.npz')
X_win = data['X_win']  # Remplace X_red
X_lose = data['X_lose']  # Remplace X_blue
y = data['y']
mask_value = data['mask_value'].item()

# 2. Nouveaux paramètres
n_champions = mask_value
team_size = 5
embed_dim = 128
dropout_rate = 0.5
alpha = 0.5
gamma = 1.5


@saving.register_keras_serializable(package="CustomLayers")
# @saving.register_keras_serializable()
class AttentionBlock(Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.query = Dense(self.embed_dim)
        self.keys = Dense(self.embed_dim)
        super().build(input_shape)
    
    def call(self, inputs):
        query = self.query(inputs)
        keys = self.keys(inputs)
        attention_scores = tf.matmul(query, keys, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        return tf.matmul(attention_weights, inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def build_improved_model():
    # Renommer les entrées pour plus de clarté
    win_input = Input(shape=(team_size,), name='winning_team')
    lose_input = Input(shape=(team_size,), name='losing_team')

    embedding = Embedding(
        input_dim=n_champions + 1,
        output_dim=embed_dim,
        embeddings_initializer='he_normal',
        name='champion_embedding'
    )

    # Embeddings pour les deux équipes
    win_embedded = embedding(win_input)
    lose_embedded = embedding(lose_input)
    
    # Couches d'attention séparées
    attention_layer = AttentionBlock(embed_dim)
    win_att = attention_layer(win_embedded)
    lose_att = attention_layer(lose_embedded)
    
    # Nouvelle stratégie de combinaison
    win_features = GlobalAveragePooling1D()(win_att + win_embedded)
    lose_features = GlobalAveragePooling1D()(lose_att + lose_embedded)
    
    # Combinaison asymétrique
    # merged = Concatenate()([
    #     win_features * 0.7,  # Poids plus important pour l'équipe gagnante
    #     lose_features * 0.3
    # ])
    merged = Concatenate()([win_features, lose_features])
    # Ajout de Dropout plus agressif
    merged = BatchNormalization()(merged)

    x = BatchNormalization()(merged)
    x = Dense(384, activation='swish', kernel_constraint=MaxNorm(3.0))(x)
    x = Dropout(0.6)(x)
    x = Dense(192, activation='swish', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(96, activation='swish')(x)
    x = Dropout(0.4)(x)
    
    output = Dense(n_champions, activation='sigmoid')(x)

    return Model(inputs=[win_input, lose_input], outputs=output)


    return Model(inputs=[win_input, lose_input], outputs=output)

def calculate_class_weights(y):
    n_samples = y.shape[0]
    positive = np.sum(y, axis=0)
    weights = np.log(n_samples / (positive + 1e-7))  # Poids de type TF-IDF
    return weights / weights.max()  # Normalisation

# 3. Préparation des données
X_train_win, X_val_win, X_train_lose, X_val_lose, y_train, y_val = train_test_split(
    X_win, X_lose, y, 
    test_size=0.15,
    random_state=42,
    #stratify=y  # Réactivé avec la nouvelle structure
)

# 4. Augmentation des données (seulement sur le train)
def augment_data(X_win, X_lose, y):
    pos_indices = np.where(np.any(y == 1, axis=1))[0]
    return (
        np.concatenate([X_win, X_win[pos_indices]]),
        np.concatenate([X_lose, X_lose[pos_indices]]),
        np.concatenate([y, y[pos_indices]])
    )

X_train_win, X_train_lose, y_train = augment_data(X_train_win, X_train_lose, y_train)

# 5. Configuration du modèle
model = build_improved_model()
class_weights = calculate_class_weights(y_train)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=FocalLoss(alpha=alpha, gamma=gamma, from_logits=False),
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(top_k=10, name='prec@10'),
        tf.keras.metrics.Recall(top_k=10, name='rec@10'),
        tf.keras.metrics.PrecisionAtRecall(0.3, name='prec@rec30'),  # Nouvelle métrique
    ]
)


if __name__ == "__main__":
    # 6. Entraînement
    history = model.fit(
        [X_train_win, X_train_lose],  # Utilisation des nouvelles entrées
        y_train,
        validation_data=([X_val_win, X_val_lose], y_val),  # Validation adaptée
        epochs=1000,  # Réduction du nombre d'epochs
        batch_size=128,  # Augmentation de la batch size
        class_weight={i: w for i, w in enumerate(class_weights)},
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                'best_model_HUGEV4.keras',
                save_best_only=True,
                monitor='val_prec@10',  # Focus sur la précision top 10
                mode='max'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=10,  # Au lieu de 20
                monitor='val_prec@10',  # Focus sur la précision
                mode='max',
                restore_best_weights=True
            )
        ]
    )
    # 6. Optimisation du seuil adaptée
    y_pred = model.predict([X_val_win, X_val_lose])  # Prédiction sur les nouvelles données
    # 7. Optimisation du seuil (version simplifiée)

    optimal_thresholds = []
    for champ_idx in range(n_champions):
        if np.sum(y_val[:, champ_idx]) > 0:  # Uniquement pour les champions présents
            fpr, tpr, thresholds = roc_curve(y_val[:, champ_idx], y_pred[:, champ_idx])
            optimal = thresholds[np.argmax(tpr - fpr)]
            optimal_thresholds.append(optimal)
        else:
            optimal_thresholds.append(0.5)  # Valeur par défaut

    np.save('model_config_v2.npy', np.array(optimal_thresholds))
    model.save('final_model_HUGEV4.keras')
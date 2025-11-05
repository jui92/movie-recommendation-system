from __future__ import annotations
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class FeaturesEmbedding(layers.Layer):
    def __init__(self, field_dims: List[int], embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.emb_layers = [
            layers.Embedding(input_dim=int(fd), output_dim=int(embed_dim),
                             embeddings_initializer="glorot_uniform")
            for fd in field_dims
        ]

    def call(self, x):
        embs = []
        for i, emb in enumerate(self.emb_layers):
            e = emb(tf.cast(x[:, i], tf.int32))  # [B, D]
            e = tf.expand_dims(e, axis=1)        # [B, 1, D]
            embs.append(e)
        return tf.concat(embs, axis=1)           # [B, F, D]

class MultiHeadSelfInteraction(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, att_res: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, output_shape=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(embed_dim * 2, activation="relu"),
                                        layers.Dense(embed_dim)])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.att_res = att_res

    def call(self, x, training=False):
        attn_out = self.mha(x, x, training=training)
        x = self.norm1(x + attn_out) if self.att_res else self.norm1(attn_out)
        ffn_out = self.ffn(x, training=training)
        return self.norm2(x + ffn_out)

class AutoIntModel(Model):
    def __init__(self, field_dims: np.ndarray, embed_dim=16, att_layer_num=3, att_head_num=2,
                 att_res=True, dnn_hidden_units=None, dnn_dropout=0.4,
                 l2_reg_dnn=0.0, dnn_use_bn=False, **kwargs):
        super().__init__(**kwargs)
        field_dims = [int(x) for x in list(field_dims)]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, name="features_embedding")
        self.att_blocks = [MultiHeadSelfInteraction(embed_dim, att_head_num, att_res, name=f"att_{i}")
                           for i in range(att_layer_num)]
        self.global_pool = layers.GlobalAveragePooling1D()
        if dnn_hidden_units is None: dnn_hidden_units = [64, 32]
        dnn_layers = []
        for i, u in enumerate(dnn_hidden_units):
            dnn_layers.append(layers.Dense(u, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn)))
            if dnn_use_bn: dnn_layers.append(layers.BatchNormalization())
            dnn_layers.append(layers.ReLU())
            if dnn_dropout: dnn_layers.append(layers.Dropout(dnn_dropout))
        self.dnn = tf.keras.Sequential(dnn_layers, name="dnn_head")
        self.logit = layers.Dense(1, activation=None, name="logit")

    def call(self, x, training=False):
        if isinstance(x, (list, tuple)):
            x = tf.convert_to_tensor(x)
        if x.shape.rank == 1:
            x = tf.expand_dims(x, axis=0)
        h = self.embedding(x)
        for blk in self.att_blocks:
            h = blk(h, training=training)
        pooled = self.global_pool(h)
        dnn_out = self.dnn(pooled, training=training)
        score = tf.keras.activations.sigmoid(self.logit(dnn_out))
        return score  # [B, 1]

def predict_model(model: AutoIntModel, df_encoded, topk: int = 20):
    import pandas as pd
    if not isinstance(df_encoded, pd.DataFrame) or 'movie_id' not in df_encoded.columns:
        raise ValueError("df_encoded must be a DataFrame with 'movie_id'.")
    X = df_encoded.values.astype("int64")
    mid = df_encoded['movie_id'].values.astype("int64")
    scores = model(X, training=False).numpy().reshape(-1)
    order = np.argsort(-scores)[:topk]
    return [(int(mid[i]), float(scores[i])) for i in order]

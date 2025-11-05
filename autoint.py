from __future__ import annotations
from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class FeaturesEmbedding(layers.Layer):
    def __init__(self, field_dims: List[int], embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.emb_layers = [
            layers.Embedding(input_dim=fd, output_dim=embed_dim, embeddings_initializer="glorot_uniform")
            for fd in field_dims
        ]

    def call(self, x):
        embs = []
        for i, emb in enumerate(self.emb_layers):
            e = emb(tf.cast(x[:, i], tf.int32))
            e = tf.expand_dims(e, axis=1)
            embs.append(e)
        out = tf.concat(embs, axis=1)
        return out 

class MultiHeadSelfInteraction(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, att_res: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, output_shape=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 2, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.att_res = att_res

    def call(self, x, training=False):
        attn_out = self.mha(x, x, training=training)
        if self.att_res:
            x = self.norm1(x + attn_out)
        else:
            x = self.norm1(attn_out)
        ffn_out = self.ffn(x, training=training)
        out = self.norm2(x + ffn_out)
        return out

class AutoIntModel(Model):
    def __init__(
        self,
        field_dims: np.ndarray,
        embed_dim: int = 16,
        att_layer_num: int = 3,
        att_head_num: int = 2,
        att_res: bool = True,
        dnn_hidden_units: List[int] | None = None,
        dnn_dropout: float = 0.0,
        l2_reg_dnn: float = 0.0,
        l2_reg_embedding: float = 0.0,
        dnn_use_bn: bool = False,
        init_std: float = 0.0001,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.field_dims = list(map(int, field_dims))
        self.embed_dim = int(embed_dim)
        self.embedding = FeaturesEmbedding(self.field_dims, self.embed_dim, name="features_embedding")
        self.att_blocks = [
            MultiHeadSelfInteraction(embed_dim=self.embed_dim, num_heads=att_head_num, att_res=att_res, name=f"att_block_{i}")
            for i in range(att_layer_num)
        ]
        self.global_pool = layers.GlobalAveragePooling1D(name="avg_pool_fields")
        if dnn_hidden_units is None:
            dnn_hidden_units = [64, 32]
        dnn_layers = []
        for i, units in enumerate(dnn_hidden_units):
            dnn_layers.append(layers.Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_reg_dnn), name=f"dnn_dense_{i}"))
            if dnn_use_bn:
                dnn_layers.append(layers.BatchNormalization(name=f"dnn_bn_{i}"))
            dnn_layers.append(layers.ReLU(name=f"dnn_relu_{i}"))
            if dnn_dropout and dnn_dropout > 0:
                dnn_layers.append(layers.Dropout(rate=dnn_dropout, name=f"dnn_dropout_{i}"))
        self.dnn = tf.keras.Sequential(dnn_layers, name="dnn_head")
        self.logit = layers.Dense(1, activation=None, name="logit")
        self._built = False

    def call(self, x, training=False):
        if isinstance(x, (list, tuple)):
            x = tf.convert_to_tensor(x)
        if x.shape.rank == 1:
            x = tf.expand_dims(x, axis=0)
        emb = self.embedding(x)
        h = emb
        for blk in self.att_blocks:
            h = blk(h, training=training)
        pooled = self.global_pool(h)
        dnn_out = self.dnn(pooled, training=training)
        logit = self.logit(dnn_out)
        score = tf.keras.activations.sigmoid(logit)
        return score

def predict_model(model: AutoIntModel, df_encoded, topk: int = 20):
    import pandas as pd
    if isinstance(df_encoded, pd.DataFrame):
        if 'movie_id' not in df_encoded.columns:
            raise ValueError("df_encoded must include an integer-encoded 'movie_id' column.")
        X = df_encoded.values.astype("int64")
        movie_ids = df_encoded['movie_id'].values.astype("int64")
    else:
        raise ValueError("df_encoded must be a pandas.DataFrame with a 'movie_id' column.")
    scores = model(X, training=False).numpy().reshape(-1)
    order = np.argsort(-scores)
    top_idx = order[:topk]
    return [(int(movie_ids[i]), float(scores[i])) for i in top_idx]

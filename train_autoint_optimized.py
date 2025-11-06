# train_autoint_optimized.py (patched to also save LEGACY H5)
# Produces:
#   model/autoInt_model.h5              (LEGACY HDF5 — safe for by_name loading)
#   model/autoInt_model.weights.h5      (Keras3 weights format)
#   data/field_dims.npy, data/label_encoders.pkl, model/metrics.json

import os, json, joblib, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from autoint import AutoIntModel

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
ML_DIR   = DATA_DIR / "ml-1m"
MODEL_DIR= BASE / "model"
for p in [DATA_DIR, ML_DIR, MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ['user_id','movie_id','movie_decade','movie_year','rating_year',
                'rating_month','rating_decade','genre1','genre2','genre3',
                'gender','age','occupation','zip']

def ensure_features(movies, ratings, users):
    df = ratings.merge(movies, on='movie_id', how='left').merge(users, on='user_id', how='left')
    if 'movie_decade' not in df:
        df['movie_decade'] = (df.get('movie_year', pd.Series([2000]*len(df))) // 10 * 10).astype(str) + 's'
    if 'rating_year' not in df:  df['rating_year']  = 2000
    if 'rating_month' not in df: df['rating_month'] = 1
    if 'rating_decade' not in df:
        df['rating_decade'] = (pd.to_numeric(df['rating_year']) // 10 * 10).astype(str) + 's'
    for g in ['genre1','genre2','genre3']:
        if g not in df: df[g] = 'no'
    for c in ['gender','age','occupation','zip']:
        if c not in df: df[c] = 'unknown'
    keep = FEATURE_COLS + (['rating'] if 'rating' in df else [])
    return df[keep]

def load_csvs():
    movies  = pd.read_csv(ML_DIR/'movies_prepro.csv')
    ratings = pd.read_csv(ML_DIR/'ratings_prepro.csv')
    users   = pd.read_csv(ML_DIR/'users_prepro.csv')
    return movies, ratings, users

def build_encoders(df):
    encs = {}
    for c in FEATURE_COLS:
        le = LabelEncoder()
        vals = df[c].fillna('no')
        if c in ['user_id','movie_id','movie_year','rating_year','rating_month','age','occupation','zip']:
            vals = pd.to_numeric(vals, errors='coerce').fillna(0).astype(int).astype(str)
        else:
            vals = vals.astype(str)
        le.fit(vals)
        encs[c] = le
    return encs

def transform(df, encs):
    out = df.copy()
    for c, le in encs.items():
        v = out[c].fillna('no')
        if c in ['user_id','movie_id','movie_year','rating_year','rating_month','age','occupation','zip']:
            v = pd.to_numeric(v, errors='coerce').fillna(0).astype(int).astype(str)
        else:
            v = v.astype(str)
        v = v.where(v.isin(le.classes_), le.classes_[0])
        out[c] = le.transform(v)
    return out

def precision_at_k(scores, y_true, k=10):
    order = np.argsort(-scores)[:k]
    return float(np.mean(y_true[order])) if len(order) else np.nan

if __name__ == "__main__":
    movies, ratings, users = load_csvs()
    df = ensure_features(movies, ratings, users)
    encs = build_encoders(df)
    df_enc = transform(df, encs)

    field_dims = np.array([len(encs[c].classes_) for c in FEATURE_COLS], dtype='int32')
    np.save(DATA_DIR/'field_dims.npy', field_dims)
    joblib.dump(encs, DATA_DIR/'label_encoders.pkl')

    y = (pd.to_numeric(df_enc['rating'], errors='coerce').fillna(0) >= 4).astype('int32').values
    X = df_enc[FEATURE_COLS].astype('int32').values
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=2025, stratify=y)

    pos = y_tr.sum(); neg = len(y_tr)-pos
    cw = {0: float(pos/(pos+neg)), 1: float(neg/(pos+neg))}

    model = AutoIntModel(field_dims=field_dims, embed_dim=32, att_layer_num=4, att_head_num=4,
                         att_res=True, dnn_hidden_units=[128,64,32], dnn_dropout=0.2)
    _ = model(tf.convert_to_tensor(X_tr[:1]))
    try:
        opt = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.BinaryAccuracy(name='acc')])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-5),
    ]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=30, batch_size=2048,
                     class_weight=cw, callbacks=cbs, verbose=1)

    y_pred = model(X_va, training=False).numpy().reshape(-1)
    auc  = roc_auc_score(y_va, y_pred)
    p10  = precision_at_k(y_pred, y_va, k=10)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({'val_auc': float(auc), 'val_precision_at_10': float(p10),
               'history': {k: float(v[-1]) for k, v in hist.history.items()}}, 
              open(MODEL_DIR/'metrics.json','w',encoding='utf-8'), ensure_ascii=False, indent=2)

    # Save both formats
    model.save_weights(MODEL_DIR/'autoInt_model.h5')            # LEGACY H5
    model.save_weights(MODEL_DIR/'autoInt_model.weights.h5')    # Keras3 weights
    print('Saved:', (MODEL_DIR/'autoInt_model.h5').as_posix(), 'and', (MODEL_DIR/'autoInt_model.weights.h5').as_posix())

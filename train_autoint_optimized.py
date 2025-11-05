
# train_autoint_optimized.py
# Usage:
#   python train_autoint_optimized.py
# Produces:
#   data/field_dims.npy, data/label_encoders.pkl, model/autoInt_model.weights.h5

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
    # 파생 컬럼 보정
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
        # OOV → 첫 클래스에 스무딩
        v = v.where(v.isin(le.classes_), le.classes_[0])
        out[c] = le.transform(v)
    return out

def precision_at_k(scores, y_true, k=10):
    order = np.argsort(-scores)[:k]
    return float(np.mean(y_true[order])) if len(order) else np.nan

if __name__ == "__main__":
    print(">> Loading CSVs ...")
    movies, ratings, users = load_csvs()
    df = ensure_features(movies, ratings, users)

    print(">> Building encoders ...")
    encs = build_encoders(df)
    df_enc = transform(df, encs)

    # 아티팩트 저장
    field_dims = np.array([len(encs[c].classes_) for c in FEATURE_COLS], dtype='int32')
    np.save(DATA_DIR/'field_dims.npy', field_dims)
    joblib.dump(encs, DATA_DIR/'label_encoders.pkl')

    # Label
    y = (pd.to_numeric(df_enc['rating'], errors='coerce').fillna(0) >= 4).astype('int32').values
    X = df_enc[FEATURE_COLS].astype('int32').values

    # Stratified split
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=2025, stratify=y)

    # Class weight (불균형 보정)
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    cw = {0: float(pos/(pos+neg)), 1: float(neg/(pos+neg))}
    print(">> Class weight:", cw)

    print(">> Building model ...")
    model = AutoIntModel(
        field_dims=field_dims,
        embed_dim=32,
        att_layer_num=4,
        att_head_num=4,
        att_res=True,
        dnn_hidden_units=[128, 64, 32],
        dnn_dropout=0.2,
    )

    _ = model(tf.convert_to_tensor(X_tr[:1]))

    try:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5)
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.BinaryAccuracy(name='acc')]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=2, min_lr=1e-5),
    ]

    print(">> Training ...")
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=30,
        batch_size=2048,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1
    )

    from sklearn.metrics import roc_auc_score
    y_pred = model(X_va, training=False).numpy().reshape(-1)
    auc = roc_auc_score(y_va, y_pred)
    p10 = precision_at_k(y_pred, y_va, k=10)
    print(f"AUC={auc:.4f}, Precision@10={p10:.4f}")

    import json
    final_hist = {k: float(v[-1]) for k, v in hist.history.items()}
    (BASE/'model').mkdir(exist_ok=True, parents=True)
    with open(BASE/'model'/'metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'val_auc': float(auc), 'val_precision_at_10': float(p10), 'history': final_hist}, f, ensure_ascii=False, indent=2)

    print(">> Saving weights ...")
    model.save_weights(BASE/'model'/'autoInt_model.weights.h5')
    print("Saved:", (BASE/'model'/'autoInt_model.weights.h5').as_posix())

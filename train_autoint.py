# python train_autoint.py
import os, joblib, numpy as np, pandas as pd, tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from autoint import AutoIntModel

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
ML_DIR = DATA_DIR / "ml-1m"
MODEL_DIR = BASE / "model"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ML_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ['user_id','movie_id','movie_decade','movie_year','rating_year',
                'rating_month','rating_decade','genre1','genre2','genre3',
                'gender','age','occupation','zip']

def ensure_features(movies, ratings, users):
    df = ratings.merge(movies, on='movie_id', how='left').merge(users, on='user_id', how='left')
    if 'movie_decade' not in df:
        df['movie_decade'] = (df.get('movie_year', pd.Series([2000]*len(df))) // 10 * 10).astype(str) + 's'
    df['rating_year']  = df.get('rating_year',  pd.Series([2000]*len(df)))
    df['rating_month'] = df.get('rating_month', pd.Series([1]*len(df)))
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
    label_encoders = {}
    for c in FEATURE_COLS:
        le = LabelEncoder()
        vals = df[c].fillna('no')
        # 숫자처럼 보이면 숫자로 고정 → 인코더 클래스 타입을 안정화
        if c in ['user_id','movie_id','movie_year','rating_year','rating_month','age','occupation','zip']:
            vals = pd.to_numeric(vals, errors='coerce').fillna(0).astype(int).astype(str)
        else:
            vals = vals.astype(str)
        le.fit(vals)
        label_encoders[c] = le
    return label_encoders

def transform(df, encs):
    out = df.copy()
    for c, le in encs.items():
        v = out[c].fillna('no')
        if c in ['user_id','movie_id','movie_year','rating_year','rating_month','age','occupation','zip']:
            v = pd.to_numeric(v, errors='coerce').fillna(0).astype(int).astype(str)
        else:
            v = v.astype(str)
        # OOV는 첫 클래스 폴백
        v = v.where(v.isin(le.classes_), le.classes_[0])
        out[c] = le.transform(v)
    return out

def precision_at_k(model, df_enc, k=10):
    # 간단 평가: 사용자별 상위 k 중 실제 rating>=4 비율
    if 'rating' not in df_enc: return np.nan
    users = df_enc['user_id'].unique()
    precs = []
    for u in users[:200]:  # 과부하 방지용 최대 200명
        tmp = df_enc[df_enc['user_id']==u]
        X = tmp[FEATURE_COLS].values
        y = (tmp['rating']>=4).astype(int).values
        s = model(X, training=False).numpy().reshape(-1)
        top = np.argsort(-s)[:k]
        precs.append(y[top].mean() if len(top)>0 else 0.0)
    return float(np.mean(precs)) if precs else np.nan

if __name__ == "__main__":
    print(">> Loading CSVs...")
    movies, ratings, users = load_csvs()
    df = ensure_features(movies, ratings, users)

    print(">> Building encoders...")
    encs = build_encoders(df)
    df_enc = transform(df, encs)

    print(">> Saving artifacts...")
    field_dims = np.array([len(encs[c].classes_) for c in FEATURE_COLS], dtype='int32')
    np.save(DATA_DIR/'field_dims.npy', field_dims)
    joblib.dump(encs, DATA_DIR/'label_encoders.pkl')

    y = (df_enc['rating'] >= 4).astype('int32').values if 'rating' in df_enc else np.ones(len(df_enc), dtype='int32')
    X = df_enc[FEATURE_COLS].astype('int32').values

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    print(">> Building model...")
    model = AutoIntModel(field_dims, embed_dim=16, att_layer_num=3, att_head_num=2, att_res=True,
                         dnn_hidden_units=[64,32], dnn_dropout=0.4)
    _ = model(tf.convert_to_tensor(X_tr[:1]))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.BinaryAccuracy(name='acc')])

    print(">> Training...")
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=5, batch_size=2048, verbose=1)
    print({k: float(v[-1]) for k, v in hist.history.items()})

    # AUC
    y_pred = model(X_va, training=False).numpy().reshape(-1)
    auc = roc_auc_score(y_va, y_pred)
    print(f">> Val AUC: {auc:.4f}")

    # Precision@10 (간단 버전)
    va_df = pd.DataFrame(X_va, columns=FEATURE_COLS)
    va_df['rating'] = y_va
    p10 = precision_at_k(model, va_df, k=10)
    print(f">> Precision@10 (approx): {p10:.4f}")

    print(">> Saving weights...")
    # TF 2.16+ 규칙: .weights.h5
    model.save_weights(MODEL_DIR/'autoInt_model.weights.h5')
    print("Saved:", (MODEL_DIR/'autoInt_model.weights.h5').as_posix())

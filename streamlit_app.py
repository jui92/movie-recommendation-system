# streamlit_app.py
# ---------------------------------------------------------
# MovieLens 1M + AutoInt 추천 데모 (Streamlit)
# - Folder: data/ml-1m | artifacts | model
# ---------------------------------------------------------

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ========== PATH 설정 ==========
DATA_DIR = Path("data/ml-1m")
ART_DIR  = Path("artifacts")
MODEL_DIR = Path("model")

USERS_FILE   = DATA_DIR / "users.dat"
MOVIES_FILE  = DATA_DIR / "movies.dat"
RATINGS_FILE = DATA_DIR / "ratings.dat"
FIELD_DIMS_PATH = ART_DIR / "field_dims.npy"
ENCODER_PATH    = ART_DIR / "label_encoders.pkl"
WEIGHTS_PATH    = MODEL_DIR / "autoInt_model.weights.h5"

# ========== Streamlit 기본 설정 ==========
st.set_page_config(page_title="🎬 MovieLens AutoInt", layout="wide")
st.title("🎬 MovieLens 1M AutoInt 추천 시스템")

# ========== TensorFlow Import 및 초기화 ==========
tf_loaded = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf_loaded = True
except Exception as e:
    st.error("❌ TensorFlow import 실패 — requirements.txt 및 파이썬 버전을 확인하세요.")
    st.exception(e)
    # st.stop() # TensorFlow 오류 시에도 다른 기능을 테스트할 수 있도록 강제 종료 제거

# ========== 데이터 로드 ==========
@st.cache_data(show_spinner=False)
def load_tables():
    """MovieLens 데이터셋 로드"""
    users = pd.read_csv(
        USERS_FILE, sep="::", engine="python",
        names=["user_id","gender","age","occupation","zip"],
        encoding="latin-1"
    )
    movies = pd.read_csv(
        MOVIES_FILE, sep="::", engine="python",
        names=["movie_id","title","genres"],
        encoding="latin-1"
    )
    ratings = pd.read_csv(
        RATINGS_FILE, sep="::", engine="python",
        names=["user_id","movie_id","rating","timestamp"],
        encoding="latin-1"
    )
    ratings["label"] = (ratings["rating"] >= 4).astype(int)
    ratings["ts"] = pd.to_datetime(ratings["timestamp"], unit="s")
    movies["main_genre"] = movies["genres"].str.split("|").str[0]
    return users, movies, ratings


# ========== 아티팩트 및 모델 로드 ==========
@st.cache_resource(show_spinner=False)
def load_artifacts_and_model():
    """artifacts 및 모델 로드 (형식 자동 감지)"""

    # 1) field_dims
    try:
        field_dims = np.load(FIELD_DIMS_PATH, allow_pickle=True)
        field_dims = np.asarray(field_dims).astype(np.int64).ravel()
    except Exception as e:
        st.error("❌ field_dims.npy 로드 실패")
        st.exception(e)
        raise

    # 2) label_encoders.pkl
    try:
        with open(ENCODER_PATH, "rb") as f:
            enc_raw = pickle.load(f)
    except pickle.UnpicklingError as e:
        # pickle.UnpicklingError 발생 시, 버전 불일치 가능성을 명확히 안내
        st.error("❌ label_encoders.pkl 로드 실패: Pickle 오류 발생 (라이브러리 버전 불일치 가능성 높음)")
        st.warning("경고: 이 오류는 일반적으로 모델을 저장할 때 사용했던 Python/scikit-learn/Pandas 버전과 현재 환경의 버전이 일치하지 않을 때 발생합니다. `requirements.txt` 파일을 확인하세요.")
        st.exception(e)
        raise
    except Exception as e:
        st.error("❌ label_encoders.pkl 로드 실패")
        st.exception(e)
        raise

    default_cat_cols = ["user_id","movie_id","gender","age","occupation","zip","main_genre"]

    # --- 구조 자동 해석 ---
    if isinstance(enc_raw, dict) and "label_encoders" in enc_raw:
        label_encoders = enc_raw["label_encoders"]
        cat_cols = enc_raw.get("cat_cols", default_cat_cols)
    elif isinstance(enc_raw, dict):
        label_encoders = enc_raw
        cat_cols = default_cat_cols
        st.warning("label_encoders.pkl에 cat_cols 키가 없어 기본 순서 사용")
    elif isinstance(enc_raw, (tuple, list)) and len(enc_raw) == 2:
        cat_cols = list(enc_raw[0]) if isinstance(enc_raw[0], (list, tuple)) else default_cat_cols
        label_encoders = enc_raw[1]
    else:
        # 최종 ValueError 발생 지점: 버전 불일치 외에 파일 구조 자체가 예상과 다를 때
        st.error("❌ label_encoders.pkl 구조 해석 불가")
        st.warning("경고: 파일 내부 구조가 코드가 예상하는 딕셔너리, 튜플, 리스트 형식이 아닙니다. 모델 저장 로직을 확인하세요.")
        raise ValueError("label_encoders.pkl 구조를 해석할 수 없습니다.")

    # --- field_dims 보정 ---
    if len(field_dims) != len(cat_cols):
        try:
            field_dims = np.array([len(label_encoders[c]) for c in cat_cols], dtype=np.int64)
            st.info("field_dims 길이를 label_encoders 기반으로 재계산했습니다.")
        except Exception as e:
            st.error("field_dims 재계산 실패")
            st.exception(e)
            raise
    
    # --- 모델 구성 ---
    if not tf_loaded:
        st.error("TensorFlow 로드 실패로 모델을 구성할 수 없습니다.")
        return cat_cols, label_encoders, field_dims, None # 모델 객체 대신 None 반환

    num_fields  = len(cat_cols)
    embed_dim   = 32
    num_heads   = 4
    attn_layers = 2
    dropout_rate= 0.2
    mlp_units   = [128, 64]

    inp = keras.Input(shape=(num_fields,), dtype="int32")
    embeds = []
    for i, dim in enumerate(field_dims):
        vi = layers.Lambda(lambda x, idx=i: tf.gather(x, indices=idx, axis=1))(inp)
        vi = layers.Reshape((1,))(vi)
        ei = layers.Embedding(input_dim=int(dim), output_dim=embed_dim)(vi)
        embeds.append(ei)
    E = layers.Concatenate(axis=1)(embeds)

    x = E
    for _ in range(attn_layers):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    for u in mlp_units:
        x = layers.Dense(u, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    _ = model.predict(np.zeros((1, num_fields), dtype=np.int32), verbose=0)
    try:
        model.load_weights(str(WEIGHTS_PATH))
    except Exception as e:
        st.error("❌ 가중치 로드 실패 — model/autoInt_model.weights.h5 확인 필요")
        st.exception(e)
        raise

    return cat_cols, label_encoders, field_dims, model


# ========== 추천 로직 ==========
def map_single(label_encoders, col, val):
    m = label_encoders[col]
    return m.get(str(val), 0)

def recommend_for_user(users, movies, ratings, cat_cols, label_encoders, model, user_id: int, topn: int = 10):
    # 모델 로드 실패 시 추천 로직 실행 방지
    if model is None:
        st.error("모델이 로드되지 않아 추천을 진행할 수 없습니다.")
        return pd.DataFrame(columns=["movie_id","title","genres","score"])

    u = users[users["user_id"] == user_id]
    if len(u) == 0:
        g, a, o, z = "M", 25, 0, "00000"
    else:
        g, a, o, z = u.iloc[0][["gender","age","occupation","zip"]]

    seen = set(ratings.loc[ratings["user_id"]==user_id, "movie_id"].tolist())
    cand = movies[~movies["movie_id"].isin(seen)].copy()
    if cand.empty:
        return pd.DataFrame(columns=["movie_id","title","genres","score"])

    cand["main_genre"] = cand["genres"].str.split("|").str[0]

    mg_idx = cand["main_genre"].astype(str).map(label_encoders["main_genre"]).fillna(0).astype(int).values
    m_idx  = cand["movie_id"].astype(str).map(label_encoders["movie_id"]).fillna(0).astype(int).values

    g_idx = map_single(label_encoders, "gender", g)
    a_idx = map_single(label_encoders, "age", a)
    o_idx = map_single(label_encoders, "occupation", o)
    z_idx = map_single(label_encoders, "zip", z)
    u_idx = map_single(label_encoders, "user_id", user_id)

    n = len(cand)
    U = np.full((n,), u_idx, dtype=np.int32)
    M = m_idx # movie_id
    G = np.full((n,), g_idx, dtype=np.int32)
    A = np.full((n,), a_idx, dtype=np.int32)
    O = np.full((n,), o_idx, dtype=np.int32)
    Z = np.full((n,), z_idx, dtype=np.int32)
    MG = mg_idx # main_genre

    # 주의: X를 cat_cols 순서에 맞게 스택해야 합니다.
    # default_cat_cols = ["user_id","movie_id","gender","age","occupation","zip","main_genre"]
    X = np.stack([U, M, G, A, O, Z, MG], axis=1)

    scores = model.predict(X, batch_size=65536, verbose=0).ravel()
    out = cand.assign(score=scores).sort_values("score", ascending=False).head(topn)
    return out[["movie_id","title","genres","score"]]


# ========== 실행 ==========
try:
    users, movies, ratings = load_tables()
    cat_cols, label_encoders, field_dims, model = load_artifacts_and_model()

    if model is None:
        st.error("추천 시스템 핵심 모듈(TensorFlow/모델) 로드에 실패했습니다. 위의 오류 메시지를 확인하세요.")
        st.stop()
        
except Exception as e:
    st.error("초기 데이터 또는 모델 로드 중 심각한 오류가 발생했습니다. 앱을 더 이상 실행할 수 없습니다.")
    st.exception(e)
    st.stop()


uid = st.selectbox("👤 User ID 선택", sorted(users["user_id"].unique().tolist()))
topn = st.slider("추천 개수", 5, 50, 10, 1)

st.markdown("#### 최근 시청 이력")
hist = (
    ratings[ratings["user_id"]==uid]
    .sort_values("ts", ascending=False)
    .head(10)
    .merge(movies[["movie_id","title","genres"]], on="movie_id", how="left")
)
st.dataframe(hist[["movie_id","rating","ts","title","genres"]], use_container_width=True)

if st.button("🔍 추천 보기"):
    with st.spinner("추천 계산 중..."):
        recs = recommend_for_user(users, movies, ratings, cat_cols, label_encoders, model, int(uid), topn)
    st.markdown("#### 추천 결과")
    st.dataframe(recs.reset_index(drop=True), use_container_width=True, height=400)
else:
    st.info("User ID와 추천 개수를 설정하고 버튼을 눌러주세요.")
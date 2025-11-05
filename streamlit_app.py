# streamlit_app.py
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import streamlit as st

DEBUG = False 

# ===== 필수 경로 =====
DATA_DIR  = Path("data/ml-1m")
ART_DIR   = Path("artifacts")
MODEL_DIR = Path("model")

USERS_FILE   = DATA_DIR / "users.dat"
MOVIES_FILE  = DATA_DIR / "movies.dat"
RATINGS_FILE = DATA_DIR / "ratings.dat"
FIELD_DIMS_PATH = ART_DIR  / "field_dims.npy"
ENCODER_PATH    = ART_DIR  / "label_encoders.pkl"
WEIGHTS_PATH    = MODEL_DIR/ "autoInt_model.weights.h5"   # 반드시 .weights.h5

# ===== Streamlit page config =====
st.set_page_config(page_title="🎬 MovieLens AutoInt Recommender", layout="wide")

# ===== 빠른 자체 점검 =====
missing = [p for p in [FIELD_DIMS_PATH, ENCODER_PATH, WEIGHTS_PATH] if not p.exists()]
if DEBUG:
    st.sidebar.title("🛠 DEBUG")
    st.sidebar.write("CWD:", Path(".").resolve())
    try:
        import sys
        st.sidebar.write("Python:", sys.version)
    except Exception:
        pass
    st.sidebar.write("Root entries:", sorted([p.name for p in Path(".").iterdir()]))

if missing:
    st.error("❌ 다음 파일이 필요합니다(학습 후 생성됨):\n" + "\n".join(str(p) for p in missing))
    st.stop()

# ===== 안전한 TensorFlow import =====
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception as e:
    st.error("TensorFlow 임포트 실패. requirements/runtime을 확인하세요.")
    st.exception(e)
    st.stop()

if DEBUG:
    st.sidebar.write("TensorFlow:", tf.__version__)

# ===== 데이터 로딩 (캐시) =====
@st.cache_data(show_spinner=False)
def load_tables():
    # latin-1 인코딩으로 명시 (UnicodeDecodeError 방지)
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
    ratings["rating_year"]  = ratings["ts"].dt.year
    ratings["rating_month"] = ratings["ts"].dt.month
    movies["main_genre"] = movies["genres"].str.split("|").str[0]
    return users, movies, ratings

# ===== 아티팩트 & 모델 로딩 (캐시) =====
@st.cache_resource(show_spinner=False)
def load_artifacts_and_model():
    # artifacts 로드
    try:
        field_dims = np.load(FIELD_DIMS_PATH)
        with open(ENCODER_PATH, "rb") as f:
            enc = pickle.load(f)
        # enc는 {"cat_cols": [...], "label_encoders": {...}} 형태여야 함
        cat_cols       = enc["cat_cols"]
        label_encoders = enc["label_encoders"]
    except Exception as e:
        st.error("아티팩트 로드 실패(artifacts/*.npy, *.pkl). 파일 내부 구조를 확인하세요.")
        raise

    # AutoInt 모델 골격 (학습과 동일해야 함)
    num_fields  = len(cat_cols)
    embed_dim   = 32
    num_heads   = 4
    attn_layers = 2
    dropout_rate= 0.2
    mlp_units   = [128, 64]

    inp = keras.Input(shape=(num_fields,), dtype="int32")

    embeds = []
    for i, dim in enumerate(field_dims):
        vi = layers.Lambda(lambda x, idx=i: tf.gather(x, indices=idx, axis=1))(inp)  # (B,)
        vi = layers.Reshape((1,))(vi)
        ei = layers.Embedding(input_dim=int(dim), output_dim=embed_dim)(vi)          # (B,1,E)
        embeds.append(ei)
    E = layers.Concatenate(axis=1)(embeds)  # (B,F,E)

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

    # build 후 가중치 로드
    _ = model.predict(np.zeros((1, num_fields), dtype=np.int32), verbose=0)
    try:
        model.load_weights(str(WEIGHTS_PATH))
    except Exception as e:
        st.error("가중치 로드 실패(model/autoInt_model.weights.h5). 확장자/경로/모델구조를 확인하세요.")
        raise

    return cat_cols, label_encoders, field_dims, model

# ===== 유틸 =====
def map_single(label_encoders, col, val):
    m = label_encoders[col]
    return m.get(str(val), 0)

def recommend_for_user(users, movies, ratings, cat_cols, label_encoders, model, user_id: int, topn: int = 10):
    # 사용자 특성
    u = users[users["user_id"] == user_id]
    if len(u) == 0:
        g, a, o, z = "M", 25, 0, "00000"
    else:
        g, a, o, z = u.iloc[0][["gender","age","occupation","zip"]]

    # 이미 본 영화 제외
    seen = set(ratings.loc[ratings["user_id"]==user_id, "movie_id"].tolist())
    cand = movies[~movies["movie_id"].isin(seen)].copy()
    if cand.empty:
        return pd.DataFrame(columns=["movie_id","title","genres","score"])

    cand["main_genre"] = cand["genres"].str.split("|").str[0]

    # 인덱싱
    mg_idx = cand["main_genre"].astype(str).map(label_encoders["main_genre"]).fillna(0).astype(int).values
    m_idx  = cand["movie_id"].astype(str).map(label_encoders["movie_id"]).fillna(0).astype(int).values

    g_idx = map_single(label_encoders, "gender", g)
    a_idx = map_single(label_encoders, "age", a)
    o_idx = map_single(label_encoders, "occupation", o)
    z_idx = map_single(label_encoders, "zip", z)
    u_idx = map_single(label_encoders, "user_id", user_id)

    # 입력 행렬 (학습 cat_cols 순서와 동일해야 함)
    # 기본: ["user_id","movie_id","gender","age","occupation","zip","main_genre"]
    n = len(cand)
    U = np.full((n,), u_idx, dtype=np.int32)
    G = np.full((n,), g_idx, dtype=np.int32)
    A = np.full((n,), a_idx, dtype=np.int32)
    O = np.full((n,), o_idx, dtype=np.int32)
    Z = np.full((n,), z_idx, dtype=np.int32)
    X = np.stack([U, m_idx, G, A, O, Z, mg_idx], axis=1)

    scores = model.predict(X, batch_size=65536, verbose=0).ravel()
    out = cand.assign(score=scores).sort_values("score", ascending=False).head(topn)
    return out[["movie_id","title","genres","score"]]

# ===== 데이터/모델 로딩 =====
users, movies, ratings = load_tables()
cat_cols, label_encoders, field_dims, model = load_artifacts_and_model()

# ===== UI =====
st.title("🎬 MovieLens AutoInt 추천")
st.caption("데이터: MovieLens 1M | 모델: AutoInt (TensorFlow/Keras)")

left, mid, right = st.columns([2,2,1])
with left:
    uid = st.selectbox("User ID", options=sorted(users["user_id"].unique().tolist()), index=0)
with mid:
    topn = st.slider("추천 개수", 5, 50, 10, 1)
with right:
    st.write("")

st.divider()
st.markdown("#### 사용자의 최근 시청 이력(평점 순)")
hist = (
    ratings[ratings["user_id"]==uid]
    .sort_values("ts", ascending=False)
    .head(10)
    .merge(movies[["movie_id","title","genres"]], on="movie_id", how="left")
)
st.dataframe(hist[["user_id","movie_id","rating","ts","title","genres"]], use_container_width=True, height=260)

if st.button("🔎 추천 결과 보기", type="primary"):
    with st.spinner("추천 계산 중…"):
        recs = recommend_for_user(users, movies, ratings, cat_cols, label_encoders, model, int(uid), topn=topn)
    st.markdown("#### 추천 결과")
    st.dataframe(recs.reset_index(drop=True), use_container_width=True, height=420)
else:
    st.info("상단에서 사용자/추천 개수를 설정하고 버튼을 눌러 주세요.")

# ===== (옵션) 간단 자가 점검 =====
with st.expander("✅ Self-check (필요 시 열기)"):
    checks = {
        "field_dims.npy": FIELD_DIMS_PATH.exists(),
        "label_encoders.pkl": ENCODER_PATH.exists(),
        "weights (.weights.h5)": WEIGHTS_PATH.exists(),
    }
    st.write({k: ("OK" if v else "MISSING") for k, v in checks.items()})

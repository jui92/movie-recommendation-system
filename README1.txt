
# AutoInt MovieLens Demo

Project tree created at: /mnt/data/autoint_project

Run (from inside this folder):
```
pip install -r requirements.txt
streamlit run show_st.py --server.port=8501
```
Place your artifacts:
- data/field_dims.npy
- data/label_encoders.pkl
- data/ml-1m/{movies_prepro.csv, ratings_prepro.csv, users_prepro.csv}
- model/autoInt_model_weights.h5

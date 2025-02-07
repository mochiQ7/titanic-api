import joblib
import pandas as pd 
import xgboost as xgb

# データ読み込み
df = pd.read_csv("../train.csv") # Titanicの訓練データを読み込む

# 必要な特徴量を選択
features = ["Pclass", "Sex", "Age", "SibSp", "Fare"]
df2 = df[features + ["Survived"]].dropna() #欠損値を除去

# "Sex"を0,1に変換
df2["Sex"] = df2["Sex"].map({"male": 0, "female": 1})

# 特徴量とラベルを分ける
X = df2[features]
y = df2["Survived"]

# XGBoostモデルを学習
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X,y)

# モデルを保存
joblib.dump(model, "Titanic_model.pkl")
print("Titanicモデルを保存しました!")
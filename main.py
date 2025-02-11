from fastapi import FastAPI 
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# FastAPIのインスタンスを作成
app = FastAPI()

# 事前に保存したモデルを読み込む
model = joblib.load("Titanic_model.pkl")

@app.get("/")
def root():
    return {"message": "Titanic API is running"}

# データバリエーション用のモデル
class PassengerData(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Sex: str = Field(..., regex="^(male|female)$", description="Sex (male or female)")
    Age: float = Field(..., ge=0, le=100, description="Age (0-100)")
    SibSp: int = Field(..., ge=0, le=10, description="Number of siblings/spouses")
    Fare: float = Field(..., ge=0, description="Ticket fare (must be positive)")


# 予測用のAPIエンドポイント
@app.post("/predict") #POST/predictにリクエストを送ると、この関数が動く
def  predict(data: PassengerData): # クライアント(ブラウザ)から送られたデータをdata()
    # 必要な特徴量を取得
    features = ["Pclass", "Sex", "Age", "SibSp", "Fare"]

    # JSONデータをDataFrameに変換
    df = pd.DataFrame([data.dict()])

    # "Sex"を0,1に変換
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # モデルで予測
    prediction = model.predict(df[features]) #生存、死亡を予測(1 or 0)
    survival_probability = model.predict_proba(df[features])[:, 1] # 生存確率（1の確率）のみを取得

    # 結果を返す
    return{
        "prediction": int(prediction[0]), # 0 or 1
        "survival_probability": float(survival_probability[0]) # 確率
    }
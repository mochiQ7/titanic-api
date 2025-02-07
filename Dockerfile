# 1.Pythonの公式イメージをベースにする
FROM python:3.10

# 2.作業ディレクトリを設定
WORKDIR /app

# 3.必要なファイルをコピー
COPY main.py Titanic_model.pkl requirements.txt /app/

# 4.依存ライブラリをインストール
RUN  pip install --no-cache-dir -r requirements.txt

# 5.FastAPIサーバーを起動
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


# 自動学習版の設定

1. app.py、digit_models.joblib、requirements.txt、corrections.jsonをGitHubへアップロードします。
2. GitHubでFine-grained personal access tokenを作り、number-ocr-app3へのContents: Read and writeを許可します。
3. StreamlitアプリのSettings > Secretsへ次を追加します。

GITHUB_TOKEN = "作成したトークン"

4. アプリで誤認識を正しいXX.Xへ直し、「修正結果を学習」を押します。
5. corrections.jsonへ自動保存され、次回の判定から近い画像へ反映されます。

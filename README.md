# AmiVoice Recognize for ゆかりねっと

ゆかりねっとのサードパーティ音声認識エンジンとして AmiVoice Cloud Platform の API を使う
0.5秒ごとに雑に無音判定を行い、なんかしゃべってそうだったら、また無音になるまで録音したやつを ACP API で音声認識して、その結果をゆかりねっとに渡す

## 設定

pipenv で依存パッケージをインストール

```sh
pipenv install --dev
```

`config.json` に ACP の APPKEY やゆかりねっとの待ち受けポート番号を指定
他にもマイクの設定とかに合わせて `config.py` を設定しなければいけないと思う

```json
{
    "ami_appkey": "<APPKEY>",
    "ami_engine_name": "-a-general",
    "ami_engine_uri": "https://acp-api.amivoice.com/v1/nolog/recognize",
    "yukari_net_port": 49513
}
```

実行

```sh
pipenv run python main.py -c <config.jsonのパス>
```

# onnx-speech-language-detection

Cited: https://github.com/PINTO0309/whisper-onnx-cpu

```bash
sudo apt-get update \
&& sudo apt-get upgrade -y \
&& sudo apt-get install -y --no-install-recommends \
    gcc \
    curl \
    wget \
    sudo \
    python3-all-dev \
    python-is-python3 \
    python3-pip \
    ffmpeg \
    portaudio19-dev \
&& pip install -U pip \
    requests==2.31.0 \
    psutil==5.9.5 \
    tqdm==4.65.0 \
    more-itertools==8.10.0 \
    ffmpeg-python==0.2.0 \
    transformers==4.29.2 \
    soundfile==0.12.1 \
    SpeechRecognition==3.10.0 \
    PyAudio==0.2.13 \
    onnx==1.16.2 \
    onnxruntime==1.18.1 \
    onnxsim==0.4.30 \
    protobuf==3.20.3 \
    h5py==3.7.0
```

```bash
curl -o whisper/assets/tiny_decoder_11.onnx https://github.com/PINTO0309/onnx-speech-language-detection/releases/download/1.0/tiny_decoder_11.onnx
curl -o whisper/assets/tiny_encoder_11.onnx https://github.com/PINTO0309/onnx-speech-language-detection/releases/download/1.0/tiny_encoder_11.onnx
```

- Japanese

    https://github.com/user-attachments/assets/f06bb5d1-8fd8-441c-a640-3ae122a6a2f0

- English

    https://github.com/user-attachments/assets/5c07e39f-db18-4c0d-9cc6-e7d8954391f2

```
    [
        ["en", "english", "en-US", "アメリカ英語", "american english"],
        ["zh", "chinese", "zh-CN", "中国の標準中国語", "Chinese standard Chinese"],
        ["de", "german", "de-DE", "ドイツのドイツ語", "german german"],
        ["es", "spanish", "es-ES", "スペインのスペイン語", "spain spanish"],
        ["ru", "russian", "ru-RU", "ロシアのロシア語", "russian in russia"],
        ["ko", "korean", "ko-KR", "韓国の韓国語", "korean korean"],
        ["fr", "french", "fr-FR", "フランスのフランス語", "french french"],
        ["ja", "japanese", "ja-JP", "日本の日本語", "japanese language"],
        ["pt", "portuguese", "pt-PT", "ポルトガルのポルトガル語", "Portuguese in Portugal"],
        ["tr", "turkish", "tr-TR", "トルコのトルコ語", "turkish in turkey"],
        ["pl", "polish", "pl-PL", "ポーランドのポーランド語", "polish in poland"],
        ["ca", "catalan", "ca-ES", "スペインのカタルーニャ語", "Spanish Catalan"],
        ["nl", "dutch", "nl-NL", "オランダのオランダ語", "Dutch in the Netherlands"],
        ["ar", "arabic", "ar-SA", "サウジアラビアのアラビア語", "arabic in saudi arabia"],
        ["sv", "swedish", "sv-SE", "スウェーデンのスウェーデン語", "swedish in sweden"],
        ["it", "italian", "it-IT", "イタリアのイタリア語", "italian in italy"],
        ["id", "indonesian", "id-ID", "インドネシアのインドネシア語", "Indonesian language in Indonesia"],
        ["hi", "hindi", "hi-IN", "インドのヒンディー語", "indian hindi"],
        ["fi", "finnish", "fi-FI", "フィンランドのフィンランド語", "Finnish in Finland"],
        ["vi", "vietnamese", "vi-VN", "ベトナムのベトナム語", "Vietnamese in Vietnam"],
        ["iw", "hebrew", "he-IL", "イスラエルのヘブライ語（'iw' は 'he' に変更されています）", "Hebrew for Israel ('iw' changed to 'he')"],
        ["uk", "ukrainian", "uk-UA", "ウクライナのウクライナ語", "Ukrainian language in Ukraine"],
        ["el", "greek", "el-GR", "ギリシャのギリシャ語", "greek greek"],
        ["ms", "malay", "ms-MY", "マレーシアのマレー語", "malay malay"],
        ["cs", "czech", "cs-CZ", "チェコのチェコ語", "czech czech"],
        ["ro", "romanian", "ro-RO", "ルーマニアのルーマニア語", "romanian in romania"],
        ["da", "danish", "da-DK", "デンマークのデンマーク語", "Danish in Denmark"],
        ["hu", "hungarian", "hu-HU", "ハンガリーのハンガリー語", "Hungarian in Hungary"],
        ["ta", "tamil", "ta-IN", "インドのタミル語", "indian tamil"],
        ["no", "norwegian", "no-NO", "ノルウェーのノルウェー語", "norwegian in norway"],
        ["th", "thai", "th-TH", "タイのタイ語", "thai language in thailand"],
        ["ur", "urdu", "ur-PK", "パキスタンのウルドゥ語", "pakistani urdu"],
        ["hr", "croatian", "hr-HR", "クロアチアのクロアチア語", "Croatian language in Croatia"],
        ["bg", "bulgarian", "bg-BG", "ブルガリアのブルガリア語", "Bulgarian in Bulgaria"],
        ["lt", "lithuanian", "lt-LT", "リトアニアのリトアニア語", "Lithuanian in Lithuania"],
        ["la", "latin", "la-VA", "バチカンのラテン語", "vatican latin"],
        ["mi", "maori", "mi-NZ", "ニュージーランドのマオリ語", "New Zealand Maori"],
        ["ml", "malayalam", "ml-IN", "インドのマラヤーラム語", "Indian Malayalam"],
        ["cy", "welsh", "cy-GB", "イギリスのウェールズ語", "British Welsh"],
        ["sk", "slovak", "sk-SK", "スロバキアのスロバキア語", "Slovak in Slovakia"],
        ["te", "telugu", "te-IN", "インドのテルグ語", "indian telugu"],
        ["fa", "persian", "fa-IR", "イランのペルシャ語", "iranian persian"],
        ["lv", "latvian", "lv-LV", "ラトビアのラトビア語", "Latvian language in Latvia"],
        ["bn", "bengali", "bn-IN", "インドのベンガル語", "bengali in india"],
        ["sr", "serbian", "sr-RS", "セルビアのセルビア語", "Serbian language in Serbia"],
        ["az", "azerbaijani", "az-AZ", "アゼルバイジャンのアゼルバイジャン語", "Azerbaijani language in Azerbaijan"],
        ["sl", "slovenian", "sl-SI", "スロベニアのスロベニア語", "Slovenian in Slovenia"],
        ["kn", "kannada", "kn-IN", "インドのカンナダ語", "indian kannada"],
        ["et", "estonian", "et-EE", "エストニアのエストニア語", "Estonian in Estonia"],
        ["mk", "macedonian", "mk-MK", "北マケドニアのマケドニア語", "Macedonian in North Macedonia"],
        ["br", "breton", "br-FR", "フランスのブルトン語", "french breton"],
        ["eu", "basque", "eu-ES", "スペインのバスク語", "spanish basque"],
        ["is", "icelandic", "is-IS", "アイスランドのアイスランド語", "Icelandic in Iceland"],
        ["hy", "armenian", "hy-AM", "アルメニアのアルメニア語", "Armenian language in Armenia"],
        ["ne", "nepali", "ne-NP", "ネパールのネパール語", "Nepali language in Nepal"],
        ["mn", "mongolian", "mn-MN", "モンゴルのモンゴル語", "Mongolian language in Mongolia"],
        ["bs", "bosnian", "bs-BA", "ボスニア・ヘルツェゴビナのボスニア語", "Bosnian in Bosnia and Herzegovina"],
        ["kk", "kazakh", "kk-KZ", "カザフスタンのカザフ語", "Kazakh language in Kazakhstan"],
        ["sq", "albanian", "sq-AL", "アルバニアのアルバニア語", "Albanian in Albania"],
        ["sw", "swahili", "sw-KE", "ケニアのスワヒリ語", "Kenyan Swahili"],
        ["gl", "galician", "gl-ES", "スペインのガリシア語", "Spanish Galician"],
        ["mr", "marathi", "mr-IN", "インドのマラーティー語", "marathi in india"],
        ["pa", "punjabi", "pa-IN", "インドのパンジャブ語", "Indian punjabi"],
        ["si", "sinhala", "si-LK", "スリランカのシンハラ語", "Sri Lankan Sinhala"],
        ["km", "khmer", "km-KH", "カンボジアのクメール語", "Cambodian Khmer"],
        ["sn", "shona", "sn-ZW", "ジンバブエのショナ語", "Shona language of Zimbabwe"],
        ["yo", "yoruba", "yo-NG", "ナイジェリアのヨルバ語", "Yoruba language in Nigeria"],
        ["so", "somali", "so-SO", "ソマリアのソマリ語", "Somali language of Somalia"],
        ["af", "afrikaans", "af-ZA", "南アフリカのアフリカーンス語", "Afrikaans in South Africa"],
        ["oc", "occitan", "oc-FR", "フランスのオック語", "French Occitan"],
        ["ka", "georgian", "ka-GE", "ジョージアのジョージア語", "Georgian language in Georgia"],
        ["be", "belarusian", "be-BY", "ベラルーシのベラルーシ語", "Belarusian language in Belarus"],
        ["tg", "tajik", "tg-TJ", "タジキスタンのタジク語", "Tajik language in Tajikistan"],
        ["sd", "sindhi", "sd-PK", "パキスタンのシンディ語", "pakistani sindhi"],
        ["gu", "gujarati", "gu-IN", "インドのグジャラート語", "Gujarati in India"],
        ["am", "amharic", "am-ET", "エチオピアのアムハラ語", "Ethiopian Amharic"],
        ["yi", "yiddish", "yi-DE", "ドイツのイディッシュ語", "German Yiddish"],
        ["lo", "lao", "lo-LA", "ラオスのラオ語", "Lao language in Laos"],
        ["uz", "uzbek", "uz-UZ", "ウズベキスタンのウズベク語", "Uzbek language in Uzbekistan"],
        ["fo", "faroese", "fo-FO", "フェロー諸島のフェロー語", "Faroese of the Faroe Islands"],
        ["ht", "haitian creole", "ht-HT", "ハイチのハイチクレオール語", "Haitian Creole in Haiti"],
        ["ps", "pashto", "ps-AF", "アフガニスタンのパシュト語", "Afghan Pashto"],
        ["tk", "turkmen", "tk-TM", "トルクメニスタンのトルクメン語", "Turkmen language of Turkmenistan"],
        ["nn", "nynorsk", "nn-NO", "ノルウェーのノルウェー語ニーノシュク", "Norwegian Nynorsk in Norway"],
        ["mt", "maltese", "mt-MT", "マルタのマルタ語", "Maltese in Malta"],
        ["sa", "sanskrit", "sa-IN", "インドのサンスクリット語", "indian sanskrit"],
        ["lb", "luxembourgish", "lb-LU", "ルクセンブルクのルクセンブルク語", "Luxembourgish in Luxembourg"],
        ["my", "myanmar", "my-MM", "ミャンマーのミャンマー語", "Myanmar language in Myanmar"],
        ["bo", "tibetan", "bo-CN", "中国のチベット語", "china tibetan"],
        ["tl", "tagalog", "tl-PH", "フィリピンのタガログ語", "Tagalog in the Philippines"],
        ["mg", "malagasy", "mg-MG", "マダガスカルのマラガシ語", "Malagasy language of Madagascar"],
        ["as", "assamese", "as-IN", "インドのアッサム語", "Indian Assamese"],
        ["tt", "tatar", "tt-RU", "ロシアのタタール語", "Russian Tatar"],
        ["haw", "hawaiian", "haw-US", "アメリカのハワイ語", "American Hawaiian"],
        ["ln", "lingala", "ln-CD", "コンゴ民主共和国のリンガラ語", "Lingala language of the Democratic Republic of the Congo"],
        ["ha", "hausa", "ha-NG", "ナイジェリアのハウサ語", "Hausa language in Nigeria"],
        ["ba", "bashkir", "ba-RU", "ロシアのバシキール語", "Russian Bashkir"],
        ["jw", "javanese", "jw-ID", "インドネシアのジャワ語", "Indonesian Javanese"],
        ["su", "sundanese", "su-ID", "インドネシアのスンダ語", "Indonesian Sundanese"],
    ]
```

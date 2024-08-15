import io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['USE_TF'] = '-1'
os.environ['USE_TORCH'] = '-1'
os.environ['USE_JAX'] = '-1'
import numpy as np
import psutil
import pyaudio
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Iterable, TYPE_CHECKING
import onnx
from onnx.serialization import ProtoSerializer
import onnxruntime as ort

from functools import lru_cache
from transformers import GPT2TokenizerFast
import soundfile as sf
import speech_recognition as sr
import ffmpeg

_MODELS: List = [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
]

ONNX_DTYPE_NP_DTYPE = {
    "tensor(int64)": np.int64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
}

@dataclass(frozen=True)
class Langage:
    language_code: str
    language: str
    rfc5646: str
    remarks_ja: str
    remarks_en: str

LANGUAGES = \
{language_code: Langage(language_code, language, rfc5646, remarks_ja, remarks_en) for language_code, language, rfc5646, remarks_ja, remarks_en in \
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
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{lang.language: code for code, lang in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}


# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000: number of frames in a mel spectrogram input

@dataclass(frozen=True)
class Tokenizer:
    """A thin wrapper around `GPT2TokenizerFast` providing quick access to special tokens"""

    tokenizer: "GPT2TokenizerFast"
    language: Optional[str]
    sot_sequence: Tuple[int]

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: Union[int, List[int], np.ndarray], **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)

    def decode_with_timestamps(self, tokens) -> str:
        """
        Timestamp tokens are above the special tokens' id range and are ignored by `decode()`.
        This method decodes given tokens with timestamps tokens annotated, e.g. "<|1.08|>".
        """
        outputs = [[]]
        for token in tokens:
            if token >= self.timestamp_begin:
                timestamp = f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
                outputs.append(timestamp)
                outputs.append([])
            else:
                outputs[-1].append(token)
        outputs = [s if isinstance(s, str) else self.tokenizer.decode(s) for s in outputs]
        return "".join(outputs)

    @property
    @lru_cache()
    def eot(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    @lru_cache()
    def sot(self) -> int:
        return self._get_single_token_id("<|startoftranscript|>")

    @property
    @lru_cache()
    def sot_lm(self) -> int:
        return self._get_single_token_id("<|startoflm|>")

    @property
    @lru_cache()
    def sot_prev(self) -> int:
        return self._get_single_token_id("<|startofprev|>")

    @property
    @lru_cache()
    def no_speech(self) -> int:
        return self._get_single_token_id("<|nospeech|>")

    @property
    @lru_cache()
    def no_timestamps(self) -> int:
        return self._get_single_token_id("<|notimestamps|>")

    @property
    @lru_cache()
    def timestamp_begin(self) -> int:
        return self.tokenizer.all_special_ids[-1] + 1

    @property
    @lru_cache()
    def language_token(self) -> int:
        """Returns the token id corresponding to the value of the `language` field"""
        if self.language is None:
            raise ValueError(f"This tokenizer does not have language token configured")

        additional_tokens = dict(
            zip(
                self.tokenizer.additional_special_tokens,
                self.tokenizer.additional_special_tokens_ids,
            )
        )
        candidate = f"<|{self.language}|>"
        if candidate in additional_tokens:
            return additional_tokens[candidate]

        raise KeyError(f"Language {self.language} not found in tokenizer.")

    @property
    @lru_cache()
    def all_language_tokens(self) -> Tuple[int]:
        result = []
        for token, token_id in zip(
            self.tokenizer.additional_special_tokens,
            self.tokenizer.additional_special_tokens_ids,
        ):
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)

    @property
    @lru_cache()
    def all_language_codes(self) -> Tuple[str]:
        return tuple(self.decode([l]).strip("<|>") for l in self.all_language_tokens)

    @property
    @lru_cache()
    def sot_sequence_including_notimestamps(self) -> Tuple[int]:
        return tuple(list(self.sot_sequence) + [self.no_timestamps])

    @property
    @lru_cache()
    def non_speech_tokens(self) -> Tuple[int]:
        """
        Returns the list of tokens to suppress in order to avoid any speaker tags or non-speech
        annotations, to prevent sampling texts that are not actually spoken in the audio, e.g.

        - ♪♪♪
        - ( SPEAKING FOREIGN LANGUAGE )
        - [DAVID] Hey there,

        keeping basic punctuations like commas, periods, question marks, exclamation points, etc.
        """
        symbols = list("\"#()*+/:;<=>@[\\]^_`{|}~「」『』")
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()

        # symbols that may be a single token or multiple tokens depending on the tokenizer.
        # In case they're multiple tokens, suppress the first token, which is safe because:
        # These are between U+2640 and U+267F miscellaneous symbols that are okay to suppress
        # in generations, and in the 3-byte UTF-8 representation they share the first two bytes.
        miscellaneous = set("♩♪♫♬♭♮♯")
        assert all(0x2640 <= ord(c) <= 0x267F for c in miscellaneous)

        # allow hyphens "-" and single quotes "'" between words, but not at the beginning of a word
        result = {self.tokenizer.encode(" -")[0], self.tokenizer.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in [self.tokenizer.encode(symbol), self.tokenizer.encode(" " + symbol)]:
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])

        return tuple(sorted(result))

    def _get_single_token_id(self, text) -> int:
        tokens = self.tokenizer.encode(text)
        assert len(tokens) == 1, f"{text} is not encoded as a single token"
        return tokens[0]

@dataclass(frozen=True)
class DecodingOptions:
    task: str = "transcribe"  # whether to perform X->X "transcribe" or X->English "translate"
    language: Optional[str] = None  # language that the audio is in; uses detected language if None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None     # number of independent samples to collect, when t > 0
    beam_size: Optional[int] = None   # number of beams in beam search, when t == 0
    patience: Optional[float] = None  # patience in beam search (https://arxiv.org/abs/2204.05424)

    # options for ranking generations (either beams or best-of-N samples)
    length_penalty: Optional[float] = None   # "alpha" in Google NMT, None defaults to length norm

    # prompt, prefix, and token suppression
    prompt: Optional[Union[str, List[int]]] = None   # text or tokens for the previous context
    prefix: Optional[Union[str, List[int]]] = None   # text or tokens to prefix the current context
    suppress_blank: bool = True                      # this will suppress blank outputs

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"

    # timestamp sampling options
    without_timestamps: bool = False              # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 0.0  # the initial timestamp cannot be later than this

@dataclass(frozen=True)
class DecodingResult:
    audio_features: np.ndarray
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

def softmax(x, dim=-1):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / (np.sum(e_x, axis=dim, keepdims=True))

def model_download(name: str, onnx_file_save_path: str='./whisper/assets') -> onnx.ModelProto:
    onnx_file_path = f'{onnx_file_save_path}/{name}_11.onnx'
    onnx_serialized_graph = None
    if not os.path.exists(onnx_file_path):
        url = f'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/381_Whisper/onnx/{name}_11.onnx'
        onnx_serialized_graph = requests.get(url).content
        with io.BytesIO(onnx_serialized_graph) as f:
            onnx_graph: onnx.ModelProto = onnx.load(f)
            onnx.save(onnx_graph, f'{onnx_file_save_path}/{name}_11.onnx')
    else:
        serializer: ProtoSerializer = onnx._get_serializer(fmt='protobuf')
        onnx_graph: onnx.ModelProto = onnx.load(onnx_file_path)
        onnx_serialized_graph = serializer.serialize_proto(proto=onnx_graph)
    return onnx_serialized_graph

def load_model(name: str):
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if name == "tiny":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "tiny.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "base":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "base.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "small":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "small.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "medium":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    elif name == "medium.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    else:
        raise ValueError(f"model type {name} not supported")

    dims = ModelDimensions(**dims_config)
    model = Whisper(dims=dims, model_name=name)
    return model

def available_models() -> List[str]:
    """Returns the names of available models"""
    return _MODELS

def onnx_dtype_to_np_dtype_convert(onnx_dtype: str):
    return ONNX_DTYPE_NP_DTYPE[onnx_dtype]

@lru_cache(maxsize=None)
def get_tokenizer(
    multilingual: bool,
    *,
    task: Optional[str] = None,  # Literal["transcribe", "translate", None]
    language: Optional[str] = None,
) -> Tokenizer:
    if language is not None:
        language = language.lower()
        if language not in LANGUAGES:
            if language in TO_LANGUAGE_CODE:
                language = TO_LANGUAGE_CODE[language]
            else:
                raise ValueError(f"Unsupported language: {language}")

    if multilingual:
        tokenizer_name = "multilingual"
        task = task or "transcribe"
        language = language or "en"
    else:
        tokenizer_name = "gpt2"
        task = None
        language = None

    tokenizer = build_tokenizer(name=tokenizer_name)
    all_special_ids: List[int] = tokenizer.all_special_ids
    sot: int = all_special_ids[1]
    translate: int = all_special_ids[-6]
    transcribe: int = all_special_ids[-5]

    langs = tuple(LANGUAGES.keys())
    sot_sequence = [sot]
    if language is not None:
        sot_sequence.append(sot + 1 + langs.index(language))
    if task is not None:
        sot_sequence.append(transcribe if task == "transcribe" else translate)

    return Tokenizer(tokenizer=tokenizer, language=language, sot_sequence=tuple(sot_sequence))

@lru_cache(maxsize=None)
def build_tokenizer(name: str = "gpt2"):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    path = os.path.join(os.path.dirname(__file__), "whisper", "assets", name)
    tokenizer = GPT2TokenizerFast.from_pretrained(path, local_files_only=True, device_map='cpu')

    specials = [
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES.keys()],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]

    tokenizer.add_special_tokens(dict(additional_special_tokens=specials))
    return tokenizer

@lru_cache(maxsize=None)
def mel_filters(n_mels: int = N_MELS) -> np.ndarray:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "whisper", "assets", "mel_filters.npz")) as f:
        return f[f"mel_{n_mels}"]

def detect_language(model: "Whisper", mel: np.ndarray, tokenizer: Tokenizer = None) -> Tuple[np.ndarray, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : np.ndarray, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError(f"This model doesn't have language tokens so it can't perform lang id")

    single = mel.ndim == 2
    if single:
        mel = mel[np.newaxis, ...]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = np.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = np.ones(logits.shape[-1], dtype=np.bool_)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(axis=-1)
    language_token_probs = softmax(logits, dim=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs

class OnnxAudioEncoder():
    def __init__(
        self,
        model: str,
    ):
        super().__init__()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.sess = \
            ort.InferenceSession(
                path_or_bytes=model_download(name=f'{model}_encoder'),
                sess_options=sess_options,
                providers=[
                    'CPUExecutionProvider'
                ],
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        mel: np.ndarray
    ) -> np.ndarray:
        result: np.ndarray = \
            self.sess.run(
                output_names=[
                    "output",
                ],
                input_feed={
                    "mel": mel.astype(self.inputs["mel"]),
                }
            )[0]
        return result

class OnnxTextDecoder():
    def __init__(
        self,
        model: str,
    ):
        super().__init__()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True) - 1
        self.sess = \
            ort.InferenceSession(
                path_or_bytes=model_download(name=f'{model}_decoder'),
                sess_options=sess_options,
                providers=[
                    'CPUExecutionProvider'
                ],
            )
        self.inputs = {
            input.name: onnx_dtype_to_np_dtype_convert(input.type) \
                for input in self.sess.get_inputs()
        }

    def __call__(
        self,
        x: np.ndarray,
        xa: np.ndarray,
        kv_cache: np.ndarray,
        offset: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs = \
            self.sess.run(
                output_names=[
                    "logits",
                    "output_kv_cache",
                    "cross_attention_qks",
                ],
                input_feed={
                    "tokens": x.astype(self.inputs["tokens"]),
                    "audio_features": xa.astype(self.inputs["audio_features"]),
                    "kv_cache": kv_cache.astype(self.inputs["kv_cache"]),
                    "offset": np.array([offset], dtype=self.inputs["offset"]),
                }
            )
        logits: np.ndarray = outputs[0]
        output_kv_cache: np.ndarray = outputs[1]
        cross_attention_qks: np.ndarray = outputs[2]
        return logits.astype(np.float32), output_kv_cache.astype(np.float32)

class Whisper():
    def __init__(
        self,
        dims: ModelDimensions,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name
        self.dims = dims
        self.encoder = OnnxAudioEncoder(model=model_name)
        self.decoder = OnnxTextDecoder(model=model_name)

    def embed_audio(
        self,
        mel: np.ndarray,
    ):
        return self.encoder(mel)

    def logits(
        self,
        tokens: np.ndarray,
        audio_features: np.ndarray,
    ):
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, audio_features, kv_cache=kv_cache, offset=0)
        return output

    def __call__(
        self,
        mel: np.ndarray,
        tokens: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        kv_cache = self.new_kv_cache(tokens.shape[0], tokens.shape[-1])
        output, _ = self.decoder(tokens, self.encoder(mel), kv_cache=kv_cache, offset=0)
        return output

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def new_kv_cache(
        self,
        n_group: int,
        length: int,
    ):
        if self.model_name == "tiny.en" or self.model_name == "tiny":
            size = [8, n_group, length, 384]
        elif self.model_name == "base.en" or self.model_name == "base":
            size = [12, n_group, length, 512]
        elif self.model_name == "small.en" or self.model_name == "small":
            size = [24, n_group, length, 768]
        elif self.model_name == "medium.en" or self.model_name == "medium":
            size = [48, n_group, length, 1024]
        else:
            raise ValueError(f"Unsupported model type: {self.type}")
        return np.zeros(size, dtype=np.float32)

    detect_language = detect_language

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def sliding_window_view(x: np.ndarray, window_shape, step=1):
    shape = ((x.shape[-1] - window_shape) // step + 1,) + (window_shape,)
    strides = (step * x.strides[-1],) + x.strides
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def numpy_stft(audio: np.ndarray, N_FFT: int, HOP_LENGTH: int):
    window = np.hanning(N_FFT)
    num_frames = 1 + (audio.size - N_FFT) // HOP_LENGTH
    if (audio.size - N_FFT) % HOP_LENGTH > 0:
        num_frames += 1
    audio_padded = np.pad(audio, pad_width=(N_FFT//2, N_FFT//2), mode='constant')
    frames = sliding_window_view(audio_padded, N_FFT, HOP_LENGTH)
    frames = frames[:num_frames]
    stft = np.fft.rfft(frames * window, axis=-1)

    cpstft = (np.abs(stft[:,:N_FFT//2 + 1]) ** 2).T
    magnitudes = cpstft.astype(audio.dtype)
    return magnitudes

def log_mel_spectrogram(audio: Union[str, np.ndarray], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray], shape = (*)
        The path to audio or either a NumPy array containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    np.ndarray, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if isinstance(audio, str):
        audio = load_audio(audio)

    magnitudes = numpy_stft(audio, N_FFT, HOP_LENGTH)

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, np.max(log_spec) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

def pad_or_trim(array: np.ndarray, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array

def detect_language(model: "Whisper", mel: np.ndarray, tokenizer: Tokenizer = None) -> Tuple[np.ndarray, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : np.ndarray, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError(f"This model doesn't have language tokens so it can't perform lang id")

    single = mel.ndim == 2
    if single:
        mel = mel[np.newaxis, ...]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = np.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = np.ones(logits.shape[-1], dtype=np.bool_)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(axis=-1)
    language_token_probs = softmax(logits, dim=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs

model_name = 'tiny'

def detect_lang_from_mic(device_index: int) -> str:
    model = load_model(model_name)
    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=16_000, device_index=device_index)
    while True:
        try:
            with mic as audio_source:
                recognizer.adjust_for_ambient_noise(audio_source)
                audio = recognizer.listen(audio_source)
            wav_data = audio.get_wav_data()
            wav_stream = io.BytesIO(wav_data)
            audio_array, _ = sf.read(wav_stream)
            audio_array = audio_array.astype(np.float32)
            mel: np.ndarray = log_mel_spectrogram(audio_array)
            mel = pad_or_trim(mel, N_FRAMES)
            tokenizer = get_tokenizer(model.is_multilingual)
            if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
                raise ValueError(f"This model doesn't have language tokens so it can't perform lang id")
            single = mel.ndim == 2
            if single:
                mel = mel[np.newaxis, ...]
            # skip encoder forward pass if already-encoded audio features were given
            if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
                mel = model.encoder(mel)
            # forward pass using a single token, startoftranscript
            n_audio = mel.shape[0]
            x = np.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
            logits = model.logits(x, mel)[:, 0]
            # collect detected languages; suppress all non-language tokens
            mask = np.ones(logits.shape[-1], dtype=np.bool_)
            mask[list(tokenizer.all_language_tokens)] = False
            logits[:, mask] = -np.inf
            language_tokens = logits.argmax(axis=-1)
            language_token_probs = softmax(logits, dim=-1)
            language_probs = [
                {
                    c: language_token_probs[i, j].item()
                    for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
                }
                for i in range(n_audio)
            ]
            if single:
                language_tokens = language_tokens[0]
                language_probs = language_probs[0]
            lang = LANGUAGES[max(language_probs, key=language_probs.get)]

            print(f"Detected language: {lang.language}")
            return lang.rfc5646

        except sr.UnknownValueError:
            return 'unknown'
        except sr.RequestError as e:
            return 'unknown'

def detect_lang_from_wav(wav_data: bytes) -> str:
    model = load_model(model_name)
    wav_stream = io.BytesIO(wav_data)
    audio_array, _ = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)
    mel: np.ndarray = log_mel_spectrogram(audio_array)
    mel = pad_or_trim(mel, N_FRAMES)
    tokenizer = get_tokenizer(model.is_multilingual)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError(f"This model doesn't have language tokens so it can't perform lang id")
    single = mel.ndim == 2
    if single:
        mel = mel[np.newaxis, ...]
    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)
    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = np.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]
    # collect detected languages; suppress all non-language tokens
    mask = np.ones(logits.shape[-1], dtype=np.bool_)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(axis=-1)
    language_token_probs = softmax(logits, dim=-1)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]
    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]
    lang = LANGUAGES[max(language_probs, key=language_probs.get)]

    print(f"Detected language: {lang.language}")
    return lang.rfc5646


if __name__ == "__main__":
    model = load_model(model_name)
    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount", 0)
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get("maxInputChannels") > 0:
            print(f"Input Device ID {i}, - {device_info.get('name')}")
    device_index: int = int(input("Please input your microphone Device ID: "))
    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=16_000, device_index=device_index)
    try:
        print("Speak now! (CTRL + C to exit the application)")
        while True:
            with mic as audio_source:
                recognizer.adjust_for_ambient_noise(audio_source)
                audio = recognizer.listen(audio_source)
            try:
                wav_data = audio.get_wav_data()
                wav_stream = io.BytesIO(wav_data)
                audio_array, _ = sf.read(wav_stream)
                audio_array = audio_array.astype(np.float32)
                mel: np.ndarray = log_mel_spectrogram(audio_array)
                mel = pad_or_trim(mel, N_FRAMES)
                tokenizer = get_tokenizer(model.is_multilingual)
                if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
                    raise ValueError(f"This model doesn't have language tokens so it can't perform lang id")
                single = mel.ndim == 2
                if single:
                    mel = mel[np.newaxis, ...]

                # skip encoder forward pass if already-encoded audio features were given
                if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
                    mel = model.encoder(mel)

                # forward pass using a single token, startoftranscript
                n_audio = mel.shape[0]
                x = np.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
                logits = model.logits(x, mel)[:, 0]

                # collect detected languages; suppress all non-language tokens
                mask = np.ones(logits.shape[-1], dtype=np.bool_)
                mask[list(tokenizer.all_language_tokens)] = False
                logits[:, mask] = -np.inf
                language_tokens = logits.argmax(axis=-1)
                language_token_probs = softmax(logits, dim=-1)
                language_probs = [
                    {
                        c: language_token_probs[i, j].item()
                        for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
                    }
                    for i in range(n_audio)
                ]

                if single:
                    language_tokens = language_tokens[0]
                    language_probs = language_probs[0]

                print(f"Detected language: {LANGUAGES[max(language_probs, key=language_probs.get)].language}")
                break

            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                pass

    except KeyboardInterrupt:
        # allow CTRL + C to exit the application
        pass

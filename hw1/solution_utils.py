import json, os

import joblib
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, SVC

from lid.predict import load_lid_model, predict_language

REACTIONS = ["negative", "neutral", "positive"]
MAX_CHARS = 2000     # длинные вставки (код, эссе) режем: иначе они забивают словарь
ASST_HEAD_WORDS = 40 
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")


def read_dialogs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Stemmer():
    _stem_cache: dict[str, str]
    _language_cache: dict[str, str]
    _snowball_stemmers: dict[str, SnowballStemmer]
    _model: SVC
    _vectorizer: CountVectorizer

    def __init__(self):
        self._stem_cache = {}
        self._language_cache = {}
        self._snowball_stemmers = {}
        self._model, self._vectorizer = load_lid_model()

    def _get_stemmer(self, language: str):
        """Возвращает стеммер, создавая его при первом обращении."""
        if language not in self._snowball_stemmers:
            self._snowball_stemmers[language] = SnowballStemmer(language)
        return self._snowball_stemmers[language]

    def stem(self, word: str):
        if word in self._language_cache:
            language = self._language_cache[word]
        else:
            language = predict_language(word, self._model, self._vectorizer)
            self._language_cache[word] = language

        if language == "other":
            return word
        else:
            if word not in self._stem_cache:
                stemmer = self._get_stemmer(language)
                self._stem_cache[word] = stemmer.stem(word)
            return self._stem_cache[word]


class TextNormalizer():
    _stemmer: Stemmer

    def __init__(self):
        self._stemmer = Stemmer()

    def clean_text(self, text):
        text = (text or "").replace("\\n", " ")
        return " ".join(text.split())[:MAX_CHARS]

    def normalize(self, text):
        """Токены wordpunct -> нижний регистр -> основы слов. Стоп-слова НЕ убираем: в списке
        есть «not», «no», а для реакции отрицание важно (семинар 4). Знаки «!» и «?» оставляем:
        они несут эмоцию, а стандартный token_pattern sklearn их бы выбросил."""
        out = []
        for tok in wordpunct_tokenize(self.clean_text(text).lower()):
            if tok.isalpha():
                out.append(self._stemmer.stem(tok))
            elif tok in ("!", "?"):
                out.append(tok)
        return " ".join(out)


def to_frame(dialogs):
    """Переводит диалоги в датафрейм.

    - dialog_id: id диалога из исходных данных.
    - text: очищенный и стеммированный текст всех сообщений диалога, объединенных в одну строку.
    - all_user: текст всех последующих сообщений пользователя (исключая самый первый промпт, если сообщений несколько).
    - last_user: текст самого последнего сообщения пользователя в диалоге.
    - first_user: текст самого первого сообщения пользователя в диалоге.
    - all_asst: текст всех сообщений ассистента, объединенных в одну строку.
    - last_asst: текст самого последнего сообщения ассистента.
    - last_asst_head: Нормализованный текст первых ASST_HEAD_WORDS слов из последнего сообщения ассистента.
    - last_raw: Очищенный от переносов строк, но не стеммированный текст последнего сообщения пользователя.
    - n_user: Целочисленное количество сообщений, отправленных пользователем в данном диалоге.
    - len_last: Длина в символах очищенного текста последнего сообщения пользователя.
    - len_asst: Среднее арифметическое длины в символах всех сообщений ассистента в диалоге.
    - group: Первые 80 символов из перевого запроса пользователя. (Для отлова дублей)
    - reaction: реакция.
    """
    t_n = TextNormalizer()
    rows = []
    for d in dialogs:
        reaction = (d.get("labels") or {}).get("reaction")
        if reaction is None:
            continue

        turns = d["turns"]
        us = [t["text"] or "" for t in turns if t["role"] == "user"]
        asst = [t["text"] or "" for t in turns if t["role"] == "assistant"]
        all_messages = [t["text"] or "" for t in turns]
        follow = us[1:] if len(us) > 1 else us          # реакция живёт в ответах на ответы модели
        gkey = "".join(ch for ch in us[0].lower() if ch.isalnum())[:80] if us else d["dialog_id"]
        rows.append(dict(
            dialog_id=d["dialog_id"],
            text = t_n.normalize(" ".join(all_messages)),            # единый текст для базовых моделей
            all_user = t_n.normalize(" ".join(follow)),
            last_user = t_n.normalize(us[-1]) if us else "",
            first_user = t_n.normalize(us[0]) if us else "",
            all_asst = t_n.normalize(" ".join(asst)),
            last_asst = t_n.normalize(asst[-1]) if asst else "",
            last_asst_head = t_n.normalize(
                " ".join(asst[-1].split()[:ASST_HEAD_WORDS])) if asst else "",
            last_raw = t_n.clean_text(us[-1]) if us else "",   # сырой текст: для разбора ошибок
            n_user = len(us),
            len_last = len(t_n.clean_text(us[-1])) if us else 0,
            len_asst = float(np.mean([len(a) for a in asst])) if asst else 0.0,
            group = gkey,                                  # шаблонные/почти-дубли первой реплики
            reaction = reaction,
        ))
    return pd.DataFrame(rows)

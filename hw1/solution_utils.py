import json
import os
import joblib
import re
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.stem import SnowballStemmer
from wordfreq import top_n_list

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    adjusted_rand_score, completeness_score, homogeneity_score,
    normalized_mutual_info_score, v_measure_score,
    confusion_matrix, f1_score, precision_recall_fscore_support,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import Normalizer

from artifacts import load_lid_model, predict_language

REACTIONS = ["negative", "neutral", "positive"]
MAX_CHARS = 2000  # длинные вставки (код, эссе) режем
ASST_HEAD_WORDS = 40
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")

TOPIC_HEAD_WORDS = 40  # сколько первых слов первого ответа модели берём


def show_table(frame, digits=3, axis=0):
    """Таблица с заливкой: большие значения ярче.
    axis=0 — каждый столбец со своей шкалой, axis=None — одна шкала на всю таблицу.
    """
    numeric = frame.select_dtypes("number").columns
    styled = frame.style.background_gradient(
        cmap="Greens", subset=numeric, axis=axis).format(
        precision=digits, subset=numeric)
    return styled

# загрузка данных и обработка
def read_dialogs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Удаляет все пробельные символы
    _REMOVE_WS = str.maketrans('', '', ' \t\n\r\v\f')
    _CONTROL_CHARS = str.maketrans(
        "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f",
        " " * 30)  # Заменяет управляющие ASCII-символы на пробел
    # Схлопывает множественные слэши в один
    _RE_MULTI_SLASH = re.compile(r'[/\\]+')

    MAX, HEAD, MIN = 2000, 1200, 200
    SYM = set("{}[]()<>=;_#$\\|^*+-&%~`")
    CODE_STARTS = ("def ", "class ", "import ", "#include", "return ", "function ",
                   "const ", "public ", "private ", "void ", "package ",
                   "using namespace", "select ", "<?php", "$ ", ">>> ",
                   '"""', "'''", "```python", "```cpp", "```c")
    MATH_MARKS = ("\\frac", "\\sum", "\\int", "\\begin", "\\end", "\\left", "\\cdot",
                  "\\sqrt", "\\lim", "$$", "\\(", "\\[", "\\times", "\\alpha", "\\beta")
    SHIELD = ("\\neq", "\\nabla", "\\nu", "\\not", "\\theta", "\\tau",
              "\\times", "\\tan", "\\to", "\\rho", "\\right")

    # 1. Настоящие и экранированные переводы строк
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    shield = any(m in text for m in MATH_MARKS)
    if shield:
        for c in SHIELD:
            text = text.replace(c, "\x01" + c[1:])

    text = text.replace(
        "\\r\\n",
        "\n").replace(
        "\\n",
        "\n").replace(
            "\\r",
            "\n").replace(
                "\\t",
        " ")

    if shield:
        text = text.replace("\x01", "\\")

    # 2. Ссылки
    had_url = False
    lines = []
    for line in text.split("\n"):
        words = line.split(" ")
        kept = []
        for w in words:
            w_low = w.lower().lstrip("([{<\"'")
            if "://" not in w_low and not w_low.startswith("www."):
                kept.append(w)

        if len(kept) < len(words):
            had_url = True
        lines.append(" ".join(kept))
    text = "\n".join(lines)

    # 3. Код и математика — построчно, только в длинных репликах
    had_code = False
    had_math = False

    if len(text) >= MIN:
        kept = []
        fence = False
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue

            if stripped.startswith("```"):
                fence = not fence
                had_code = True
                continue
            if fence:
                had_code = True
                continue

            # is_math
            is_m = False
            if any(m in stripped for m in MATH_MARKS):
                is_m = True
            else:
                s_no_ws = stripped.translate(_REMOVE_WS)
                if s_no_ws:
                    num_count = sum(
                        1 for c in s_no_ws if c.isdigit() or c in "+-*/=^<>().,%")
                    if num_count / len(s_no_ws) >= 0.5:
                        # Подсчет длинных слов (вызываем split один раз, а не
                        # два)
                        alpha_long = sum(
                            1 for w in stripped.split() if w.isalpha() and len(w) >= 4)
                        if alpha_long <= 2:
                            is_m = True

            if is_m:
                had_math = True
                continue

            # is_code
            s_no_ws = stripped.translate(_REMOVE_WS)
            r = sum(1 for c in s_no_ws if c in SYM) / \
                len(s_no_ws) if s_no_ws else 0.0

            is_c = False
            if stripped.endswith((";", "{", "}")) and r > 0.04:
                is_c = True
            elif stripped.lower().startswith(CODE_STARTS) and (r > 0.02 or stripped.endswith(":") or line[0].isspace()):
                is_c = True
            elif (line.startswith("    ") or line.startswith("\t")) and r > 0.05:
                is_c = True
            elif r > 0.18:
                is_c = True

            if is_c:
                had_code = True
            else:
                kept.append(line)
        text = "\n".join(kept)

    # 4. Слэши подряд в один; непечатные в пробел; пробелы в один
    text = text.translate(_CONTROL_CHARS)
    text = _RE_MULTI_SLASH.sub('/', text)
    text = " ".join(text.split())

    # 5. Обрезка и теги
    if len(text) > MAX:
        text = text[:HEAD] + " " + text[-(MAX - HEAD):]

    tags = (" zzcode" if had_code else "") + \
        (" zzmath" if had_math else "") + (" zzurl" if had_url else "")
    return (text + tags).strip()


class Stemmer():
    _stem_cache: dict[str, str]
    _language_cache: dict[str, str]
    _english_words: set
    _snowball_stemmers: dict[str, SnowballStemmer]
    _model: SVC
    _vectorizer: CountVectorizer

    def __init__(self):
        self._stem_cache = {}
        self._language_cache = {}
        self._english_words = set(top_n_list("en", 235000))
        self._snowball_stemmers = {"english": SnowballStemmer("english")}
        self._model, self._vectorizer = load_lid_model()

    def _get_stemmer(self, language: str):
        """Возвращает стеммер, создавая его при первом обращении."""
        if language not in self._snowball_stemmers:
            self._snowball_stemmers[language] = SnowballStemmer(language)
        return self._snowball_stemmers[language]

    def stem(self, word: str):
        if word in self._english_words:
            self._language_cache[word] = "english"
            if word not in self._stem_cache:
                self._stem_cache[word] = self._snowball_stemmers["english"].stem(
                    word)
            return self._stem_cache[word]
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


class TextNormalizer:
    def __init__(self):
        self._stemmer = Stemmer()
        self._norm_cache = {}

    def normalize(self, text: str, pre_cleaned: str = None) -> str:
        if text in self._norm_cache:
            return self._norm_cache[text]

        cleaned = pre_cleaned if pre_cleaned is not None else clean_text(text)

        out = []
        tokens = re.findall(r'[^\W\d_]+|[!?]', cleaned.lower())

        for tok in tokens:
            if tok in ("!", "?"):
                out.append(tok)
            else:
                out.append(self._stemmer.stem(tok))

        res = " ".join(out)
        self._norm_cache[text] = res
        return res


def tfidf(**kw):
    """Обёртка TfidfVectorizer с настройками под нормализованный текст"""
    params = dict(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2), min_df=2,
                  max_features=60000, sublinear_tf=True)
    params.update(kw)
    return TfidfVectorizer(**params)


def to_frame(dialogs):
    """Переводит диалоги в датафрейм: одна строка на диалог, порядок как во входе
    (ответы моделей поэтому можно сопоставлять с dialog_id по позиции).
    Диалоги без разметки и без реплик пользователя не выбрасываются: на скрытом тесте
    поля labels нет, фильтровать по reaction.notna() нужно в обучающем коде.

    Идентификатор и разметка:
    - dialog_id: id диалога из исходных данных.
    - reaction: разметка реакции (None, если её нет).
    - group: первые 80 букв и цифр первого запроса пользователя (для отлова дублей и GroupKFold).

    Тексты (очищены и стеммированы TextNormalizer):
    - text: все сообщения диалога одной строкой.
    - first_user: первое сообщение пользователя.
    - rest_user: остальные сообщения пользователя (каждое нормализуется отдельно).
    - all_user: остальные сообщения пользователя одной строкой (если сообщение одно - оно само).
    - last_asst_head: первые ASST_HEAD_WORDS слов последнего сообщения ассистента.
    - first_asst: первые ASST_HEAD_WORDS слов первого сообщения ассистента.
    - raw_first / last_raw: очищенные, но не стеммированные первое / последнее сообщение пользователя
      (чтобы показывать примеры).

    Числа:
    - n_user: количество сообщений пользователя.
    - len_last: длина очищенного последнего сообщения пользователя в символах.
    - len_asst: средняя длина сообщения ассистента в символах.
    """
    normalize = TextNormalizer().normalize
    _RE_NON_ALNUM = re.compile(r'[^a-z0-9]+')

    rows = []
    for i, d in enumerate(dialogs, 1):
        if i % 1000 == 0:
            print(f"Обработано {i} диалогов...")

        turns = d["turns"]
        us = [t["text"] or "" for t in turns if t["role"] == "user"]
        asst = [t["text"] or "" for t in turns if t["role"] == "assistant"]
        labels = d.get("labels") or {}

        first_raw = clean_text(us[0]) if us else ""
        last_raw = clean_text(us[-1]) if us else ""
        group = _RE_NON_ALNUM.sub("", us[0].lower())[:80] if us else ""

        rows.append({
            "dialog_id": d["dialog_id"],
            "text": normalize(" ".join(t["text"] or "" for t in turns)),
            "first_user": normalize(us[0], pre_cleaned=first_raw) if us else "",
            "rest_user": " ".join(normalize(u) for u in us[1:]),
            "all_user": normalize(" ".join(us[1:])) if len(us) > 1 else normalize(us[-1], pre_cleaned=last_raw) if us else "",
            "last_asst_head": normalize(" ".join(asst[-1].split()[:ASST_HEAD_WORDS])) if asst else "",
            "first_asst": normalize(" ".join(asst[0].split()[:ASST_HEAD_WORDS])) if asst else "",
            "raw_first": first_raw,
            "last_raw": last_raw,
            "n_user": len(us),
            "len_last": len(last_raw),
            "len_asst": float(sum(len(a) for a in asst) / len(asst)) if asst else 0.0,
            "group": group or d["dialog_id"],
            "reaction": labels.get("reaction"),
        })

    return pd.DataFrame(rows)


def single_text(col="text"):
    """Удобная обёртка ColumnTransformer в случае,
    когда признаки строятся из одной текстовой колонки"""
    return ColumnTransformer([("tfidf", tfidf(), col)])

def topic_lsa(n_components=100, seed=42):
    """TF-IDF -> SVD -> нормировка длины: в этом пространстве кластеризуем."""
    return make_pipeline(topic_features(), TruncatedSVD(n_components, random_state=seed), Normalizer(copy=False))

def topic_features(weights=(1.0, 0.5, 0.5)):
    """
    Строит ColumnTransformer, который по трём текстовым колонкам
    (first_user - первая реплика пользователя,
    rest_user - остальные реплики пользователя,
    first_asst - начало ответа модели) 
    делает TF-IDF-векторизацию с весами weights
    """
    return ColumnTransformer(
        [("first", tfidf(), "first_user"), ("rest", tfidf(), "rest_user"),
         ("asst", tfidf(), "first_asst")],
        transformer_weights=dict(zip(["first", "rest", "asst"], weights)))


# Модели к 1 задаче
def reaction_models():
    """
    dummy_most_frequent - самый частый класс.

    Базовые модели:
    наивный байес, логистическая регрессия, линейный SVM.
    Обучаются по одиному TF-IDF-блоку (по колонке text — весь диалог)

    Свои модели:
    наивный байес, логистическая регрессия, линейный SVM.
    Обучаются по двум TF-IDF-блокам начало последнего ответа ассистента
    + все последующие реплики пользователя.
    """
    return {
        # самый частый класс:
        "dummy_most_frequent": (Pipeline([("m", DummyClassifier(strategy="most_frequent"))]), {}),
        # базовые:
        "nb": (Pipeline([("f", single_text()), ("m", MultinomialNB(fit_prior=False))]),
               {"m__alpha": [0.1, 0.3, 1.0]}),
        "logreg": (Pipeline([("f", single_text()),
                             ("m", LogisticRegression(class_weight="balanced", max_iter=3000))]),
                   {"m__C": [0.3, 1, 3, 10]}),
        "linear_svm": (Pipeline([("f", single_text()), ("m", LinearSVC(class_weight="balanced"))]),
                       {"m__C": [0.03, 0.1, 0.3, 1]}),
        # свои:
        "nb_my": (Pipeline([("f", ColumnTransformer([
            ("asst_head", tfidf(), "last_asst_head"),
            ("all", tfidf(), "all_user"), 
            ])),
            ("m", MultinomialNB(fit_prior=False))]),
            {"m__alpha": [0.1, 0.3, 1.0]}),
        "logreg_my": (Pipeline([("f", ColumnTransformer([
            ("asst_head", tfidf(), "last_asst_head"),
            ("all", tfidf(), "all_user"), 
            ])),
            ("m", LogisticRegression(class_weight="balanced", max_iter=3000))]),
            {"m__C": [0.3, 1, 3, 10]}),
        "linear_svm_my": (Pipeline([("f", ColumnTransformer([
            ("asst_head", tfidf(), "last_asst_head"),
            ("all", tfidf(), "all_user"), 
            ])),
            ("m", LinearSVC(class_weight="balanced"))]),
            {"m__C": [0.03, 0.1, 0.3, 1]}),
    }


# Модели к 2 задаче
def topic_models(seed=42):
    """
    dummy_one_topic - самый частый класс (заглушка).

    Базовые модели:
    наивный байес, логистическая регрессия, линейный SVM.
    Обучаются по трём TF-IDF-блокам (first_user - первая реплика
    пользователя, rest_user - его остальные реплики, first_asst -
    начало первого ответа модели).

    Своя модель:
    LSA (TruncatedSVD) + SVM с RBF-ядром.
    Обучается по тем же трём TF-IDF-блокам, затем понижает размерность
    через SVD, нормирует и подаёт в SVM с RBF-ядром
    """
    return {
        # самый частый класс:
        "dummy_one_topic": (Pipeline([("m", DummyClassifier(strategy="most_frequent"))]), {}),
        # базовые:
        "nb": (Pipeline([("f", topic_features()), ("m", MultinomialNB(fit_prior=False))]),
               {"m__alpha": [0.03, 0.1, 0.3]}),
        "logreg": (Pipeline([("f", topic_features()),
                             ("m", LogisticRegression(class_weight="balanced", max_iter=2000))]),
                   {"m__C": [1, 10, 30]}),
        "linear_svm": (Pipeline([("f", topic_features()), ("m", LinearSVC(class_weight="balanced"))]),
                       {"m__C": [0.1, 0.3, 1]}),
        # свои:
        "own_lsa_rbf_svm": (Pipeline(
            [("f", topic_features()),
             ("svd", TruncatedSVD(100, random_state=seed)),
             ("norm", Normalizer()),
             ("m", SVC(kernel="rbf", class_weight="balanced"))]),
            {"m__C": [1, 10]}),
    }


# Метрики к задаче 1
def reaction_metrics(y_true, y_pred):
    """
    Считает метрики macro-F1 (средняя F1 по всем классам)
    macro-F1 сбалансированная (F1 с весом класса 1/размер класса),
    accuracy, balanced_acc,
    precision, recall, F1 по каждому классу из REACTIONS.
    """
    y_true = np.asarray(y_true)
    counts = pd.Series(y_true).value_counts()
    w = 1.0 / pd.Series(y_true).map(counts).to_numpy()
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=REACTIONS, zero_division=0)
    out = dict(
        macro_f1=f1_score(y_true, y_pred, average="macro", labels=REACTIONS, zero_division=0),  
        macro_f1_bal=f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=REACTIONS,
            sample_weight=w,
            zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        balanced_acc=balanced_accuracy_score(y_true, y_pred)
        )
    for i, c in enumerate(REACTIONS):
        out[f"P_{c}"], out[f"R_{c}"], out[f"F1_{c}"] = p[i], r[i], f[i]
    return out


def confusion(y_true, y_pred):
    """Матрица ошибок как DataFrame"""
    return pd.DataFrame(confusion_matrix(
        y_true, y_pred, labels=REACTIONS), index=REACTIONS, columns=REACTIONS)


def plot_confusions(y_true, preds, ncols=3):
    """Матрицы ошибок нескольких моделей рядом.
    В клетке: число объектов и доля от истинного класса"""
    names = list(preds)
    ncols = min(ncols, len(names))
    nrows = -(-len(names) // ncols)
    _, axes = plt.subplots(
        nrows, ncols, figsize=(
            4.3 * ncols, 3.9 * nrows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    k = len(REACTIONS)
    for ax, name in zip(axes.ravel(), names):
        cm = confusion_matrix(
            np.asarray(y_true), np.asarray(
                preds[name]), labels=REACTIONS)
        share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.5 else "black")
        ax.set_xticks(range(k), REACTIONS, fontsize=8)
        ax.set_yticks(range(k), REACTIONS, fontsize=8)
        ax.set_xlabel("предсказано")
        ax.set_ylabel("истина")
        ax.set_title(name, fontsize=10)
        ax.grid(False)
    plt.tight_layout()
    plt.show()


# Метрики к задаче 2
def purity_score(y_ref, y_pred):
    """Считает чистоту разбиения (насколько «однородны» кластеры относительно эталонных меток)
    Каждому кластеру приписывает его самую частую эталонную метку.
    Чистота = какой долей объектов кластеры совпали с эталоном.
    y_ref — эталонные метки, y_pred — предсказанные"""
    cm = contingency_matrix(y_ref, y_pred)
    return cm.max(axis=0).sum() / cm.sum()


def partition_scores(y_ref, y_pred):
    """Набор метрик сравнения разбиений.
    y_ref — эталонные метки, y_pred — предсказанные"""
    return {
        "ARI": adjusted_rand_score(y_ref, y_pred),
        "NMI": normalized_mutual_info_score(y_ref, y_pred),
        "однородность": homogeneity_score(y_ref, y_pred),
        "полнота": completeness_score(y_ref, y_pred),
        "V-мера": v_measure_score(y_ref, y_pred),
        "чистота": purity_score(y_ref, y_pred)
    }

# Удобный интерфейс для задачи 1
def init_reaction_models(path=ARTIFACTS):
    """Все сохранённые модели задачи 1: {имя: конвейер}."""
    return {f[len("reaction_"):-len(".joblib")]: joblib.load(os.path.join(path, f))
            for f in sorted(os.listdir(path)) if f.startswith("reaction_") and f.endswith(".joblib")}

def read_labels(path: str, task: str) -> dict[str, str]:
    """Ответы по задаче task ("reaction" или "topic"): dialog_id -> метка.

    Диалоги, у которых такой метки нет, пропускаются.
    """
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                label = (record.get("labels") or {}).get(task)
                if label is not None:
                    labels[record["dialog_id"]] = label
    return labels


def predict(model, data):
    """По одному ответу на диалог. data - список диалогов или уже готовый to_frame():
    признаки дорого считать, поэтому для цикла по моделям фрейм строят один раз."""
    frame = data if isinstance(data, pd.DataFrame) else to_frame(data)
    return [str(p) for p in model.predict(frame)]

def compute_metrics(dialogs, predictions, labels, metrics):
    """metrics(y_true, y_pred) по тем диалогам, для которых известен ответ (labels: {dialog_id: метка}).
    Для реакции metrics=reaction_metrics, для тем - partition_scores (названия тем не совпадают)."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    return metrics([true for true, _ in pairs], [pred for _, pred in pairs])

def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, dict[str, list[str]]]) -> None:
    """Сохранить ответы всех моделей: dialog_id, task, model, prediction."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for task, by_model in predictions.items():
            for name, answers in by_model.items():
                for dialog, answer in zip(dialogs, answers):
                    writer.writerow([dialog["dialog_id"], task, name, answer])


# Удобный интерфейс для задачи 2
def init_topic_models(path=ARTIFACTS):
    """Все сохранённые модели задачи 2: {имя: конвейер}."""
    return {f[len("topic_"):-len(".joblib")]: joblib.load(os.path.join(path, f))
            for f in sorted(os.listdir(path)) if f.startswith("topic_") and f.endswith(".joblib")}



def topic_metrics(y_ref, y_pred):
    return {"accuracy": accuracy_score(y_ref, y_pred),
            "macro_f1": f1_score(y_ref, y_pred, average="macro", zero_division=0),
            **partition_scores(y_ref, y_pred)}
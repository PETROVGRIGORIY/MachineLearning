import csv
import json
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nltk.stem import SnowballStemmer
from wordfreq import top_n_list

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from artifacts import load_lid_model, predict_language

REACTIONS = ["negative", "neutral", "positive"]
ASST_HEAD_WORDS = 40
HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS, "reaction_models")

META_INPUT = ["raw_first", "prev_raw", "last_raw", "rest_raw", "n_user", "n_asst",
              "len_last", "len_asst", "len_asst_last", "len_asst_first"]

_THANKS = re.compile(r"thank|thx|спасибо|благодар|great|perfect|awesome|excellent|nice|cool|"
                     r"good job|well done|love it|amazing|wonderful|отлично|супер|класс|круто|"
                     r"здорово|молодец|прекрасно|шикарно|превосходно|божественно")
_NEG = re.compile(r"wrong|incorrect|not correct|not right|that'?s not|doesn'?t work|does not work|"
                  r"not working|didn'?t work|error|bad|worse|useless|nonsense|stupid|terrible|awful|"
                  r"again|redo|неправильн|неверн|не так|ошибк|не работает|не помог|плохо|бред|ерунд|"
                  r"чушь|ужас|опять|снова|не то|что за")
_POLITE = re.compile(r"please|pls|plz|пожалуйста|будь добр|прошу|умоляю")
_CONT = re.compile(r"continue|go on|next|more|продолж|дальше|ещё|еще")
_ACK = re.compile(r"\b(ok|okay|got it|i see|понятно|ясно|ладно|хорошо|понял|поняла)\b")
_LEXICONS = [("thanks", _THANKS), ("neg", _NEG), ("polite", _POLITE), ("cont", _CONT), ("ack", _ACK)]
_TAGS = ["zzcode", "zzmath", "zzurl"]
_CYR = re.compile(r"[а-яё]")
_WORD = re.compile(r"\w+")

META_NAMES = (
    ["n_user", "n_asst", "len_first", "len_last", "len_rest_mean", "len_asst", "len_asst_last",
     "len_asst_first", "len_ratio_last_first", "excl_last", "ques_last", "excl_rest", "ques_rest",
     "last_is_question", "jaccard_last_prev", "jaccard_last_first"]
    + [f"{n}_last" for n, _ in _LEXICONS] + [f"{n}_rest" for n, _ in _LEXICONS]
    + [f"{t}_last" for t in _TAGS] + [f"{t}_rest" for t in _TAGS]
    + ["cyr_share_last", "upper_share_last", "digit_share_last", "n_words_last"])


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


#  Данные: чтение, метки, датафрейм признаков
def read_labels(path: str, task: str = "reaction") -> dict[str, str]:
    """Ответы по задаче task: dialog_id -> метка.

    Поддерживаются два формата строки jsonl:
    - вложенный, как в hw1_train.jsonl: {"dialog_id": ..., "labels": {"reaction": ..., ...}};
    - плоский, как в файле ответов скрытого теста: {"dialog_id": ..., "reaction": ...}.
    Диалоги, у которых такой метки нет, пропускаются.
    """
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                label = (record.get("labels") or record).get(task)
                if label is not None:
                    labels[record["dialog_id"]] = label
    return labels


def to_frame(dialogs):
    """Переводит диалоги в датафрейм: одна строка на диалог, порядок как во входе
    (ответы моделей поэтому можно сопоставлять с dialog_id по позиции).
    Диалоги без разметки и без реплик пользователя не выбрасываются: на скрытом тесте
    поля labels нет, фильтровать по reaction.notna() нужно в обучающем коде.

    Колонки HW1 (значения те же, что в HW1, поэтому модели HW1 работают без изменений):
    - dialog_id, reaction (None, если метки нет), group (первые 80 букв и цифр
      первого запроса пользователя — для StratifiedGroupKFold и отлова дублей);
    - text, first_user, rest_user, all_user, last_asst_head, first_asst — очищенные
      и стеммированные тексты (см. TextNormalizer);
    - raw_first, last_raw — очищенные, но не стеммированные первое и последнее
      сообщения пользователя;
    - n_user, len_last, len_asst — числа.

    Новые колонки HW2 (дешёвые, без стеммера; на них считаются признаки для деревьев):
    - prev_raw: предпоследнее сообщение пользователя (очищенное, не стеммированное);
    - rest_raw: все сообщения пользователя, кроме первого, одной строкой;
    - n_asst: число сообщений ассистента;
    - len_asst_first, len_asst_last: длина первого и последнего ответа ассистента.
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

        us_clean = [clean_text(u) for u in us]  # каждое сообщение чистим один раз
        first_raw = us_clean[0] if us else ""
        last_raw = us_clean[-1] if us else ""
        group = _RE_NON_ALNUM.sub("", us[0].lower())[:80] if us else ""

        rows.append({
            "dialog_id": d["dialog_id"],
            "text": normalize(" ".join(t["text"] or "" for t in turns)),
            "first_user": normalize(us[0], pre_cleaned=first_raw) if us else "",
            "rest_user": " ".join(normalize(u, pre_cleaned=c) for u, c in zip(us[1:], us_clean[1:])),
            "all_user": normalize(" ".join(us[1:])) if len(us) > 1 else normalize(us[-1], pre_cleaned=last_raw) if us else "",
            "last_asst_head": normalize(" ".join(asst[-1].split()[:ASST_HEAD_WORDS])) if asst else "",
            "first_asst": normalize(" ".join(asst[0].split()[:ASST_HEAD_WORDS])) if asst else "",
            "raw_first": first_raw,
            "last_raw": last_raw,
            "n_user": len(us),
            "len_last": len(last_raw),
            "len_asst": float(sum(len(a) for a in asst) / len(asst)) if asst else 0.0,
            "prev_raw": us_clean[-2] if len(us) > 1 else "",
            "rest_raw": " ".join(us_clean[1:]),
            "n_asst": len(asst),
            "len_asst_first": len(asst[0]) if asst else 0,
            "len_asst_last": len(asst[-1]) if asst else 0,
            "group": group or d["dialog_id"],
            "reaction": labels.get("reaction"),
        })

    return pd.DataFrame(rows)


# Сбор признаков для деревьев
def jaccard_text_similarity(a: str, b: str) -> float:
    """Вычисляет сходство Жаккара между двумя строками на основе множества уникальных слов.
    Игнорирует регистр и порядок слов. Возвращает отношение количества общих слов 
    к общему числу уникальных слов в обоих текстах (значение от 0.0 до 1.0)"""
    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def meta_features(X: pd.DataFrame) -> np.ndarray:
    """
    Вход: таблица, в которой каждая строка описывает один диалог. Из неё берутся
    колонки, перечисленные в META_INPUT: тексты первой, предыдущей и последней
    реплик пользователя, текст всех остальных его реплик, число реплик пользователя
    и ассистента, длины ответов ассистента.

    Выход: двумерный массив чисел формы (число диалогов, число признаков).
    Названия признаков и порядок столбцов в этом массиве заданы списком META_NAMES.

    Какие признаки считаются:

    1. Размеры диалога (все длины сглажены логарифмом ln(1 + x)):
       - число реплик пользователя и число ответов ассистента;
       - длина первой реплики пользователя;
       - длина последней реплики пользователя;
       - средняя длина остальных реплик пользователя (всех, кроме последней);
       - суммарная длина ответов ассистента;
       - длина первого ответа ассистента и длина последнего ответа ассистента;
       - отношение длины последней реплики пользователя к длине первой (не больше 10).

    2. Пунктуация:
       - число восклицательных и число вопросительных знаков в последней реплике пользователя;
       - число восклицательных и число вопросительных знаков в остальных его репликах;
       - заканчивается ли последняя реплика пользователя вопросительным знаком (0 или 1).

    3. Повторение запроса — сходство Жаккара, то есть доля общих слов среди всех
       уникальных слов двух текстов (от 0 до 1):
       - между последней и предыдущей репликами пользователя;
       - между последней и первой репликами пользователя.

    4. Тематические словари — сколько раз найдены слова (не больше 5), отдельно
       в последней реплике пользователя и отдельно в остальных его репликах:
       - благодарность;
       - недовольство и сообщения об ошибке;
       - вежливая просьба;
       - просьба продолжить;
       - согласие, подтверждение понимания.
       Регистр не учитывается, русские слова ищутся по основам.

    5. Служебные метки, которыми при чистке текста заменены блоки кода, формулы и ссылки:
       - встречается ли метка в последней реплике пользователя (0 или 1) —
         отдельно для кода, для формул, для ссылок;
       - сколько таких меток в остальных репликах пользователя —
         отдельно для кода, для формул, для ссылок.

    6. Состав символов последней реплики пользователя:
       - доля кириллических букв среди всех букв;
       - доля заглавных букв среди всех букв;
       - доля цифр среди всех символов;
       - число слов, сглаженное логарифмом ln(1 + x).
    """
    rows = []
    cols = [X[c].tolist() for c in META_INPUT]
    for first, prev, last, rest, n_user, n_asst, len_last, len_asst, len_al, len_af in zip(*cols):
        last_l, rest_l = last.lower(), rest.lower()
        letters = sum(ch.isalpha() for ch in last)
        row = [
            n_user, n_asst, np.log1p(len(first)), np.log1p(len_last),
            np.log1p(len(rest) / max(n_user - 1, 1)), np.log1p(len_asst),
            np.log1p(len_al), np.log1p(len_af), min(len_last / (len(first) + 1), 10.0),
            last.count("!"), last.count("?"), rest.count("!"), rest.count("?"),
            float(last.rstrip().endswith("?")), jaccard_text_similarity(last, prev), jaccard_text_similarity(last, first),
        ]
        row += [min(len(rx.findall(last_l)), 5) for _, rx in _LEXICONS]
        row += [min(len(rx.findall(rest_l)), 5) for _, rx in _LEXICONS]
        row += [float(t in last) for t in _TAGS] + [float(rest.count(t)) for t in _TAGS]
        row += [len(_CYR.findall(last_l)) / max(letters, 1),
                sum(ch.isupper() for ch in last) / max(letters, 1),
                sum(ch.isdigit() for ch in last) / max(len(last), 1),
                np.log1p(len(last.split()))]
        rows.append(row)
    return np.asarray(rows, dtype=float).reshape(len(rows), len(META_NAMES))


def tree_features(blocks=("all", "asst", "first", "meta"), n_components=(64, 32, 32), seed=42):
    """Плотные признаки для деревьев (ColumnTransformer над фреймом to_frame).

    Блоки:
    - all   — LSA по остальным репликам пользователя (all_user);
    - asst  — LSA по началу последнего ответа ассистента (last_asst_head);
    - first — LSA по первому запросу пользователя (first_user);
    - meta  — ручные признаки (meta_features).
    n_components — число компонент LSA для блоков all, asst, first.
    """
    columns = {"all": "all_user", "asst": "last_asst_head", "first": "first_user"}
    sizes = dict(zip(("all", "asst", "first"), n_components))

    def lsa(n_components, max_features=20000, seed=42):
        return make_pipeline(
            tfidf(max_features=max_features),
            TruncatedSVD(n_components, random_state=seed)
        )

    def meta_names_out(transformer, input_features):
        return np.asarray(META_NAMES, dtype=object)

    parts = [(b, lsa(sizes[b], seed=seed), columns[b]) for b in blocks if b != "meta"]
    if "meta" in blocks:
        parts.append((
            "meta",
            FunctionTransformer(
                meta_features,
                feature_names_out=meta_names_out
            ),
            META_INPUT))
    return ColumnTransformer(parts)


# Модели hw2
class GroupedFolds(StratifiedGroupKFold):
    """StratifiedGroupKFold, который берёт группы из колонки group фрейма to_frame.

    Нужен стекингу (StackingClassifier): он делит данные на блоки сам и групп не получает,
    а диалоги с одинаковым началом не должны попадать и в обучение, и в предсказание
    метапризнаков (как в HW1).
    """

    def split(self, X, y=None, groups=None):
        return super().split(X, y, X["group"] if groups is None else groups)


def hw2_reaction_models(featurizer=tree_features, seed=42):
    """Новые модели HW2: имя -> (конвейер, сетка параметров для GridSearchCV).

    rf_tfidf — случайный лес на сыром TF-IDF (то, что деревьям не подходит: точка сравнения).
    rf_lsa   — случайный лес на LSA и ручных признаках.
    hgb_lsa  — HistGradientBoostingClassifier на тех же признаках.
    gb_lsa   — GradientBoostingClassifier (вторая, классическая реализация бустинга).
    featurizer — функция без аргументов, возвращающая преобразователь признаков для деревьев.
    Число итераций бустинга hgb_lsa подбирает ранняя остановка (max_iter — потолок), поэтому в сетке только скорость обучения и глубина.
    Стекинг (StackingClassifier с GroupedFolds) собирается в ноутбуке обучения: ему нужны подобранные параметры базовых моделей.
    """
    return {
        "rf_tfidf": (Pipeline([
            ("f", ColumnTransformer([("all", tfidf(max_features=5000), "all_user"),
                                     ("asst", tfidf(max_features=3000), "last_asst_head")])),
            ("m", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
                                         n_jobs=1, random_state=seed))]),
            {}),   # без подбора: это контр-пример, на разреженном входе лес медленный, а параметры его не спасают
        "rf_lsa": (Pipeline([
            ("f", featurizer()),
            ("m", RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                         n_jobs=1, random_state=seed))]),
            {"m__min_samples_leaf": [3, 5, 10], "m__max_features": ["sqrt", 0.3]}),
        "hgb_lsa": (Pipeline([
            ("f", featurizer()),
            ("m", HistGradientBoostingClassifier(class_weight="balanced", l2_regularization=1.0, max_iter=300,
                                                 early_stopping=True, n_iter_no_change=10, random_state=seed))]),
            {"m__learning_rate": [0.05, 0.1], "m__max_depth": [3, None]}),
        "gb_lsa": (Pipeline([
            ("f", featurizer()),
            ("m", GradientBoostingClassifier(n_estimators=200, max_depth=3, subsample=0.8,
                                       max_features="sqrt", random_state=seed))]),
            {"m__learning_rate": [0.05, 0.1], "m__n_estimators": [100, 250]}),
    }


# Метрики
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


# Инференс для загрузки моделей, ответов, метрик, сохранения
def init_reaction_models(path=MODELS_DIR):
    """Все сохранённые модели: {имя: модель}, сначала hw1_, затем hw2_.

    В path лежат файлы reaction_<имя>.joblib. Имена с префиксами hw1_ / hw2_ берутся
    как есть, файлы без префикса (так HW1 сохранял свои модели) считаются моделями
    HW1 и получают префикс hw1_.
    """
    models = {}
    for f in sorted(os.listdir(path)):
        if f.startswith("reaction_") and f.endswith(".joblib"):
            name = f[len("reaction_"):-len(".joblib")]
            if not name.startswith(("hw1_", "hw2_")):
                name = "hw1_" + name
            models[name] = joblib.load(os.path.join(path, f))
    return dict(sorted(models.items(), key=lambda kv: (not kv[0].startswith("hw1_"), kv[0])))


def predict_reaction(model, data):
    """По одному ответу на диалог: "positive", "neutral" или "negative".
    data — список диалогов или уже готовый to_frame(): признаки дорого считать,
    поэтому для цикла по моделям фрейм строят один раз."""
    frame = data if isinstance(data, pd.DataFrame) else to_frame(data)
    return [str(p) for p in model.predict(frame)]


def compute_metrics(dialogs, predictions, labels, metrics=reaction_metrics):
    """metrics(y_true, y_pred) по тем диалогам, для которых известен ответ (labels: {dialog_id: метка})."""
    pairs = [(labels[d["dialog_id"]], p) for d, p in zip(dialogs, predictions) if d["dialog_id"] in labels]
    return metrics([true for true, _ in pairs], [pred for _, pred in pairs])


def save_predictions(path: str, dialogs: list[dict], predictions: dict[str, list[str]]) -> None:
    """Сохранить ответы всех моделей: dialog_id, task, model, prediction."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dialog_id", "task", "model", "prediction"])
        for name, answers in predictions.items():
            for dialog, answer in zip(dialogs, answers):
                writer.writerow([dialog["dialog_id"], "reaction", name, answer])

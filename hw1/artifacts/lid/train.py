import os
import re
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import balanced_accuracy_score


def word_language_df():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "text_examples")

    languages = [
        "arabic", "danish", "dutch", "english", "french",
        "german", "hungarian", "italian", "norwegian",
        "other", "portuguese", "romanian", "russian",
        "spanish", "swedish"
    ]

    rows = []

    for lang in languages:
        file_path = os.path.join(folder_path, f"{lang}.txt")
        
        if not os.path.exists(file_path):
            print(f"Файл не найден: {file_path}")
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        words = re.findall(r'[^\W\d_]+', text, flags=re.UNICODE)
        
        for word in words:
            word_lower = word.lower()
            rows.append({"word": word_lower, "language": lang})

    df = pd.DataFrame(rows)
    print("Датафрейм сформирован")
    return df


def fit_lid_model():

    df = word_language_df()

    X_train, X_test, y_train, y_test = train_test_split(
        df["word"], df["language"],
        test_size=0.2, random_state=42, stratify=df["language"]
    )

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 4)) #n граммы

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"Balanced Accuracy: {bal_acc}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    joblib.dump(model, os.path.join(base_dir, "lid_model.pkl"))
    joblib.dump(vectorizer, os.path.join(base_dir, "lid_vectorizer.pkl"))
    
    print("Модель и векторайзер сохранены.")

fit_lid_model()

# NLP utilities for text classification

Набор небольших Python-утилит для задач классификации текстов:

- подготовка и очистка текстовых датасетов;
- кодирование текстов в эмбеддинги через `transformers` или `sentence-transformers`;
- быстрый прогон нескольких классификаторов и сравнение метрик.

Код разбит на три независимых модуля:

- `text_data_utils.py` - подготовка текстов, меток и train/test-разбиения;
- `embedding_utils.py` - получение эмбеддингов и инференс transformer-моделей;
- `model_tester.py` - обучение и оценка sklearn-подобных классификаторов.

## Установка зависимостей

Минимально для части функций нужен только стандартный Python. Дополнительные пакеты импортируются лениво, только когда вызывается соответствующая функция.

```bash
pip install pandas scikit-learn tqdm numpy
pip install torch transformers sentence-transformers
pip install catboost
```

`catboost` необязателен: если он не установлен, базовая модель CatBoost будет пропущена при `error_policy="warn"`.

## Быстрый пример

```python
from text_data_utils import prepare_text_classification_data
from embedding_utils import encode_train_test_texts
from model_tester import evaluate_classifiers

data = prepare_text_classification_data(
    texts=[
        "Отличный препарат",
        "Помог быстро",
        "Состояние улучшилось",
        "Появилась сыпь",
        "Стало хуже",
        "Был побочный эффект",
    ],
    labels=["positive", "positive", "positive", "negative", "negative", "negative"],
    test_size=0.33,
)

X_train, X_test = encode_train_test_texts(
    data["X_train_texts"],
    data["X_test_texts"],
    model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

result = evaluate_classifiers(
    X_train,
    X_test,
    data["y_train"],
    data["y_test"],
)

print(result.best_model_name)
print(result.to_frame())
```

## `text_data_utils.py`

Модуль для подготовки исходных текстов и меток перед обучением модели.

| Объект | Для чего нужен |
| --- | --- |
| `TextDataset` | Контейнер для списка текстов, опциональных меток и метаданных обработки. Проверяет, что число текстов и меток совпадает. |
| `TextDataset.to_frame()` | Преобразует `TextDataset` в `pandas.DataFrame` с колонками `text` и, если есть, `label`. |
| `clean_text()` | Очищает один текст: HTML, URL, email, лишние пробелы, опционально приводит к нижнему регистру. |
| `normalize_texts()` | Применяет `clean_text()` ко списку текстов, при наличии `tqdm` показывает progress bar. |
| `deduplicate_texts()` | Удаляет дубли текстов и возвращает новый `TextDataset` с метаданными о количестве удаленных строк. |
| `filter_by_text_length()` | Оставляет только тексты в заданном диапазоне длины по символам. |
| `get_text_length_stats()` | Считает количество текстов и статистику длины в символах и словах: `min`, `max`, `mean`, `median`. |
| `get_class_distribution()` | Возвращает распределение классов как количества или доли, если `normalize=True`. |
| `build_label_mapping()` | Создает словарь соответствия исходных меток числовым id. |
| `encode_labels()` | Кодирует метки в числа и возвращает пару: список id и словарь `label_to_id`. |
| `decode_labels()` | Преобразует числовые id обратно в исходные метки. |
| `split_text_dataset()` | Делит тексты и метки на train/test через `sklearn.model_selection.train_test_split`. |
| `sample_texts_by_class()` | Ограничивает число примеров каждого класса, чтобы быстро собрать уменьшенную выборку для экспериментов. |
| `batch_texts()` | Разбивает список текстов на батчи фиксированного размера. |
| `read_text_dataset_csv()` | Читает CSV с текстами и, если указана колонка, метками. Может сразу очистить тексты. |
| `prepare_text_classification_data()` | Выполняет полный базовый пайплайн подготовки: очистка, дедупликация, фильтр длины, кодирование меток, train/test-разбиение, статистика. |

Пример подготовки:

```python
from text_data_utils import read_text_dataset_csv, prepare_text_classification_data

dataset = read_text_dataset_csv(
    "reviews.csv",
    text_column="review",
    label_column="sentiment",
    clean=True,
)

data = prepare_text_classification_data(dataset.texts, dataset.labels)
```

## `embedding_utils.py`

Модуль для получения эмбеддингов и предсказаний BERT-подобных моделей.

| Объект | Для чего нужен |
| --- | --- |
| `EmbeddingResult` | Контейнер с эмбеддингами и метаданными: число текстов, размерность, pooling, модель, device, batch size. |
| `TransformerPredictionResult` | Контейнер результата инференса классификатора: logits, probabilities, predictions, текстовые labels и mapping id-to-label. |
| `batch_iter()` | Универсально разбивает последовательность на батчи фиксированного размера. |
| `load_transformer_encoder()` | Загружает `AutoTokenizer` и `AutoModel` из `transformers` для извлечения эмбеддингов. |
| `load_transformer_classifier()` | Загружает `AutoTokenizer` и `AutoModelForSequenceClassification` для классификации последовательностей. |
| `mean_pooling()` | Усредняет token embeddings с учетом `attention_mask`, чтобы получить один вектор на текст. |
| `cls_pooling()` | Возвращает embedding первого CLS-токена из `last_hidden_state`. |
| `normalize_embeddings()` | Делает L2-нормализацию torch-тензора или numpy-массива по оси эмбеддингов. |
| `encode_texts()` | Кодирует список текстов через `transformers`-энкодер. Поддерживает `mean` и `cls` pooling, нормализацию, numpy/torch-вывод. |
| `encode_train_test_texts()` | Кодирует train и test тексты одной загруженной transformer-моделью, чтобы не загружать ее дважды. |
| `encode_texts_sentence_transformer()` | Кодирует тексты через пакет `sentence-transformers`. Удобно для готовых sentence embedding моделей. |
| `predict_transformer_classifier()` | Делает батчевый инференс `AutoModelForSequenceClassification` и возвращает logits, вероятности и предсказанные классы. |

Пример кодирования:

```python
from embedding_utils import encode_texts

embeddings = encode_texts(
    ["Болит горло", "После лекарства появилась сыпь"],
    model_name_or_path="cointegrated/rubert-tiny2",
    pooling="mean",
    normalize=True,
)
```

Пример инференса классификатора:

```python
from embedding_utils import predict_transformer_classifier

result = predict_transformer_classifier(
    ["Препарат помог", "Стало хуже после приема"],
    model_name_or_path="path/to/classifier",
)

print(result.predictions)
print(result.predicted_labels)
```

## `model_tester.py`

Модуль для сравнения моделей классификации на уже подготовленных числовых признаках.

| Объект | Для чего нужен |
| --- | --- |
| `ModelRunResult` | Полный результат одной модели: статус, ошибка, время обучения и предсказания, отчеты, confusion matrix, основные метрики. |
| `ModelTestResult` | Сводный результат нескольких моделей: все прогоны, таблица summary и имя лучшей модели. |
| `ModelTestResult.to_frame()` | Возвращает `summary` как `pandas.DataFrame`. |
| `evaluate_classifiers()` | Обучает и оценивает набор моделей. По умолчанию добавляет LogisticRegression, RandomForestClassifier и CatBoostClassifier, если зависимости доступны. |

`evaluate_classifiers()` ожидает уже готовые признаки: TF-IDF, эмбеддинги или другие числовые представления. Сырые тексты нужно подготовить заранее.

Пример с пользовательской моделью:

```python
from sklearn.svm import LinearSVC
from model_tester import evaluate_classifiers

result = evaluate_classifiers(
    X_train,
    X_test,
    y_train,
    y_test,
    models={"linear_svc": LinearSVC()},
    include_baselines=True,
)

print(result.best_model_name)
print(result.summary)
```

## Типовой пайплайн

1. Прочитать данные через `read_text_dataset_csv()` или передать списки текстов и меток напрямую.
2. Подготовить датасет через `prepare_text_classification_data()`.
3. Построить признаки:
   - `encode_train_test_texts()` для transformer-эмбеддингов;
   - `encode_texts_sentence_transformer()` для sentence-transformers;
   - либо любой внешний TF-IDF/vectorizer.
4. Сравнить классификаторы через `evaluate_classifiers()`.
5. Посмотреть `result.summary`, `result.to_frame()` и `result.best_model_name`.

## Важные замечания

- Функции проверяют базовые ошибки входных данных: несовпадение длин текстов и меток, некорректный `batch_size`, неизвестные label id.
- Большие зависимости (`torch`, `transformers`, `pandas`, `sklearn`) импортируются внутри функций, поэтому можно использовать легкие части проекта без установки всего ML-стека.
- Внутренние функции с префиксом `_` не считаются публичным API и могут меняться без предупреждения.

"""Вспомогательные функции для работы с текстовыми датасетами.

Модуль рассчитан на задачи классификации текстов: подготовка сырых строк,
проверка меток, разбиение на train/test и быстрый первичный анализ. Тяжелые
зависимости импортируются только внутри функций, которым они действительно
нужны.
"""

from __future__ import annotations

import html
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


__all__ = [
    "TextDataset",
    "batch_texts",
    "build_label_mapping",
    "clean_text",
    "decode_labels",
    "deduplicate_texts",
    "encode_labels",
    "filter_by_text_length",
    "get_class_distribution",
    "get_text_length_stats",
    "normalize_texts",
    "prepare_text_classification_data",
    "read_text_dataset_csv",
    "sample_texts_by_class",
    "split_text_dataset",
]


@dataclass
class TextDataset:
    """Контейнер для текстов, меток и служебной информации о датасете."""

    texts: List[str]
    labels: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Проверяет, что количество текстов и меток совпадает."""

        if self.labels is not None and len(self.texts) != len(self.labels):
            raise ValueError("Количество текстов и меток должно совпадать.")

    def to_frame(self) -> Any:
        """Возвращает датасет как pandas.DataFrame, если pandas установлен."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Для TextDataset.to_frame() нужен pandas. "
                "Установите его или используйте поля `.texts` и `.labels` напрямую."
            ) from exc

        data = {"text": self.texts}
        if self.labels is not None:
            data["label"] = self.labels
        return pd.DataFrame(data)


def clean_text(
    text: Any,
    lowercase: bool = True,
    strip_html: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_extra_spaces: bool = True,
    keep_line_breaks: bool = False,
) -> str:
    """Очищает один текст без изменения смысла: пробелы, HTML, ссылки, email.

    Функция специально не удаляет пунктуацию и цифры по умолчанию, потому что в
    медицинских и отзывных текстах они часто несут полезную информацию.
    """

    value = "" if text is None else str(text)

    if strip_html:
        value = re.sub(r"<[^>]+>", " ", value)
        value = html.unescape(value)

    if remove_urls:
        value = re.sub(r"https?://\S+|www\.\S+", " ", value)

    if remove_emails:
        value = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", " ", value)

    if lowercase:
        value = value.lower()

    if remove_extra_spaces:
        if keep_line_breaks:
            value = re.sub(r"[ \t\r\f\v]+", " ", value)
            value = re.sub(r"\n{3,}", "\n\n", value)
            value = "\n".join(part.strip() for part in value.splitlines())
        else:
            value = re.sub(r"\s+", " ", value)
            value = value.strip()

    return value


def normalize_texts(
    texts: Iterable[Any],
    show_progress: bool = True,
    **clean_kwargs: Any,
) -> List[str]:
    """Применяет clean_text ко всем текстам и показывает прогресс при наличии tqdm."""

    texts_list = list(texts)
    progress = _progress_iter(
        texts_list,
        enabled=show_progress,
        desc="Очистка текстов",
        total=len(texts_list),
    )
    return [clean_text(text, **clean_kwargs) for text in progress]


def deduplicate_texts(
    texts: Sequence[str],
    labels: Optional[Sequence[Any]] = None,
    keep: str = "first",
) -> TextDataset:
    """Удаляет дубликаты текстов и возвращает TextDataset.

    Если переданы метки, дубликат считается только по тексту, а метка берется из
    первой или последней найденной строки в зависимости от параметра keep.
    """

    if keep not in {"first", "last"}:
        raise ValueError("keep должен быть 'first' или 'last'.")
    if labels is not None and len(texts) != len(labels):
        raise ValueError("Количество текстов и меток должно совпадать.")

    indexes = range(len(texts)) if keep == "first" else range(len(texts) - 1, -1, -1)
    seen = set()
    kept_indexes = []

    for index in indexes:
        text = texts[index]
        if text in seen:
            continue
        seen.add(text)
        kept_indexes.append(index)

    if keep == "last":
        kept_indexes.reverse()

    new_texts = [texts[index] for index in kept_indexes]
    new_labels = [labels[index] for index in kept_indexes] if labels is not None else None
    return TextDataset(
        texts=new_texts,
        labels=new_labels,
        metadata={
            "original_size": len(texts),
            "deduplicated_size": len(new_texts),
            "removed_duplicates": len(texts) - len(new_texts),
        },
    )


def filter_by_text_length(
    texts: Sequence[str],
    labels: Optional[Sequence[Any]] = None,
    min_chars: int = 1,
    max_chars: Optional[int] = None,
) -> TextDataset:
    """Фильтрует тексты по длине в символах."""

    if min_chars < 0:
        raise ValueError("min_chars не может быть отрицательным.")
    if max_chars is not None and max_chars < min_chars:
        raise ValueError("max_chars должен быть больше или равен min_chars.")
    if labels is not None and len(texts) != len(labels):
        raise ValueError("Количество текстов и меток должно совпадать.")

    kept_texts = []
    kept_labels = [] if labels is not None else None
    removed = 0

    for index, text in enumerate(texts):
        text_len = len(text)
        is_valid = text_len >= min_chars and (max_chars is None or text_len <= max_chars)
        if is_valid:
            kept_texts.append(text)
            if kept_labels is not None:
                kept_labels.append(labels[index])
        else:
            removed += 1

    return TextDataset(
        texts=kept_texts,
        labels=kept_labels,
        metadata={
            "original_size": len(texts),
            "filtered_size": len(kept_texts),
            "removed_by_length": removed,
            "min_chars": min_chars,
            "max_chars": max_chars,
        },
    )


def get_text_length_stats(texts: Iterable[str]) -> Dict[str, Any]:
    """Считает простую статистику длины текстов в символах и словах."""

    texts_list = list(texts)
    char_lengths = [len(text) for text in texts_list]
    word_lengths = [len(text.split()) for text in texts_list]

    return {
        "count": len(texts_list),
        "chars": _numeric_stats(char_lengths),
        "words": _numeric_stats(word_lengths),
    }


def get_class_distribution(labels: Iterable[Any], normalize: bool = False) -> Dict[Any, Any]:
    """Возвращает распределение классов: количества или доли."""

    counts = Counter(labels)
    if not normalize:
        return dict(counts)

    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: count / total for label, count in counts.items()}


def build_label_mapping(labels: Iterable[Any], sort_labels: bool = True) -> Dict[Any, int]:
    """Создает словарь преобразования исходных меток в целые id."""

    unique_labels = list(dict.fromkeys(labels))
    if sort_labels:
        try:
            unique_labels = sorted(unique_labels)
        except TypeError:
            unique_labels = sorted(
                unique_labels,
                key=lambda label: (type(label).__name__, str(label)),
            )
    return {label: index for index, label in enumerate(unique_labels)}


def encode_labels(
    labels: Iterable[Any],
    label_to_id: Optional[Mapping[Any, int]] = None,
    sort_labels: bool = True,
) -> Tuple[List[int], Dict[Any, int]]:
    """Кодирует метки в числа и возвращает закодированные метки вместе со словарем."""

    labels_list = list(labels)
    mapping = dict(label_to_id) if label_to_id is not None else build_label_mapping(
        labels_list,
        sort_labels=sort_labels,
    )

    unknown_labels = [label for label in labels_list if label not in mapping]
    if unknown_labels:
        unknown_preview = ", ".join(map(str, unknown_labels[:5]))
        raise ValueError(f"Найдены неизвестные метки: {unknown_preview}")

    return [mapping[label] for label in labels_list], mapping


def decode_labels(encoded_labels: Iterable[int], label_to_id: Mapping[Any, int]) -> List[Any]:
    """Преобразует числовые id обратно в исходные метки."""

    id_to_label = {index: label for label, index in label_to_id.items()}
    decoded = []
    for index in encoded_labels:
        if index not in id_to_label:
            raise ValueError(f"Неизвестный id метки: {index}")
        decoded.append(id_to_label[index])
    return decoded


def split_text_dataset(
    texts: Sequence[str],
    labels: Sequence[Any],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[List[str], List[str], List[Any], List[Any]]:
    """Делит тексты и метки на train/test через sklearn train_test_split."""

    if len(texts) != len(labels):
        raise ValueError("Количество текстов и меток должно совпадать.")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise ImportError(
            "Для split_text_dataset нужен scikit-learn. "
            "Установите его или сделайте разбиение вручную."
        ) from exc

    stratify_labels = labels if stratify else None
    return train_test_split(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )


def sample_texts_by_class(
    texts: Sequence[str],
    labels: Sequence[Any],
    max_per_class: int,
    random_state: int = 42,
) -> TextDataset:
    """Ограничивает число примеров каждого класса, полезно для быстрых экспериментов."""

    if max_per_class <= 0:
        raise ValueError("max_per_class должен быть больше нуля.")
    if len(texts) != len(labels):
        raise ValueError("Количество текстов и меток должно совпадать.")

    rng = random.Random(random_state)
    grouped_indexes: Dict[Any, List[int]] = {}
    for index, label in enumerate(labels):
        grouped_indexes.setdefault(label, []).append(index)

    selected_indexes = []
    for indexes in grouped_indexes.values():
        indexes_copy = list(indexes)
        rng.shuffle(indexes_copy)
        selected_indexes.extend(indexes_copy[:max_per_class])

    selected_indexes.sort()
    return TextDataset(
        texts=[texts[index] for index in selected_indexes],
        labels=[labels[index] for index in selected_indexes],
        metadata={
            "max_per_class": max_per_class,
            "original_size": len(texts),
            "sampled_size": len(selected_indexes),
        },
    )


def batch_texts(texts: Sequence[str], batch_size: int) -> Iterator[List[str]]:
    """Разбивает список текстов на батчи фиксированного размера."""

    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше нуля.")

    for start in range(0, len(texts), batch_size):
        yield list(texts[start : start + batch_size])


def read_text_dataset_csv(
    path: str,
    text_column: str,
    label_column: Optional[str] = None,
    encoding: Optional[str] = None,
    dropna: bool = True,
    clean: bool = False,
    show_progress: bool = True,
    read_csv_kwargs: Optional[Dict[str, Any]] = None,
    clean_kwargs: Optional[Dict[str, Any]] = None,
) -> TextDataset:
    """Читает CSV-файл с текстами и, при наличии, метками."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Для read_text_dataset_csv нужен pandas. "
            "Установите его или прочитайте CSV другим способом."
        ) from exc

    kwargs = dict(read_csv_kwargs or {})
    if encoding is not None:
        kwargs["encoding"] = encoding

    frame = pd.read_csv(path, **kwargs)
    required_columns = [text_column]
    if label_column is not None:
        required_columns.append(label_column)

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"В CSV нет колонок: {', '.join(missing_columns)}")

    if dropna:
        frame = frame.dropna(subset=required_columns)

    texts = frame[text_column].astype(str).tolist()
    labels = frame[label_column].tolist() if label_column is not None else None

    if clean:
        texts = normalize_texts(
            texts,
            show_progress=show_progress,
            **(clean_kwargs or {}),
        )

    return TextDataset(
        texts=texts,
        labels=labels,
        metadata={
            "path": path,
            "text_column": text_column,
            "label_column": label_column,
            "rows": len(texts),
        },
    )


def prepare_text_classification_data(
    texts: Iterable[Any],
    labels: Iterable[Any],
    test_size: float = 0.2,
    random_state: int = 42,
    clean: bool = True,
    encode_target: bool = True,
    deduplicate: bool = True,
    min_chars: int = 1,
    max_chars: Optional[int] = None,
    stratify: bool = True,
    show_progress: bool = True,
    clean_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Готовит текстовый датасет для классификации одним вызовом.

    Возвращает словарь с train/test текстами, метками, mapping классов и
    статистикой. Эмбеддинги эта функция не строит: после нее можно вызвать
    encode_train_test_texts из embedding_utils.py.
    """

    texts_list = list(texts)
    labels_list = list(labels)
    if len(texts_list) != len(labels_list):
        raise ValueError("Количество текстов и меток должно совпадать.")

    if clean:
        texts_list = normalize_texts(
            texts_list,
            show_progress=show_progress,
            **(clean_kwargs or {}),
        )
    else:
        texts_list = ["" if text is None else str(text) for text in texts_list]

    dataset = TextDataset(texts=texts_list, labels=labels_list)
    processing_metadata = {
        "original_size": len(dataset.texts),
        "clean_applied": clean,
    }

    if deduplicate:
        dataset = deduplicate_texts(dataset.texts, dataset.labels)
        processing_metadata["deduplication"] = dataset.metadata

    dataset = filter_by_text_length(
        dataset.texts,
        dataset.labels,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    processing_metadata["length_filter"] = dataset.metadata

    final_labels = dataset.labels or []
    label_to_id = None
    if encode_target:
        final_labels, label_to_id = encode_labels(final_labels)

    X_train, X_test, y_train, y_test = split_text_dataset(
        dataset.texts,
        final_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "X_train_texts": X_train,
        "X_test_texts": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_to_id": label_to_id,
        "id_to_label": {index: label for label, index in label_to_id.items()}
        if label_to_id is not None
        else None,
        "class_distribution": get_class_distribution(final_labels),
        "text_length_stats": get_text_length_stats(dataset.texts),
        "metadata": processing_metadata,
    }


def _numeric_stats(values: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    sorted_values = sorted(values)
    count = len(sorted_values)
    middle = count // 2
    if count % 2:
        median = float(sorted_values[middle])
    else:
        median = (sorted_values[middle - 1] + sorted_values[middle]) / 2

    return {
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "mean": sum(sorted_values) / count,
        "median": median,
    }


def _progress_iter(
    values: Iterable[Any],
    enabled: bool,
    desc: Optional[str] = None,
    total: Optional[int] = None,
) -> Iterable[Any]:
    if not enabled:
        return values

    try:
        from tqdm.auto import tqdm
    except ImportError:
        return values

    return tqdm(values, desc=desc, total=total)

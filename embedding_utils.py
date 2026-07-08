"""Утилиты для эмбеддингов и BERT-подобных моделей.

Функции помогают получать эмбеддинги текстов, объединять скрытые состояния
и запускать батчевый инференс моделей классификации последовательностей. Тяжелые
зависимости импортируются только внутри функций, которым они нужны.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


__all__ = [
    "EmbeddingResult",
    "TransformerPredictionResult",
    "batch_iter",
    "cls_pooling",
    "encode_texts",
    "encode_texts_sentence_transformer",
    "encode_train_test_texts",
    "load_transformer_classifier",
    "load_transformer_encoder",
    "mean_pooling",
    "normalize_embeddings",
    "predict_transformer_classifier",
]


@dataclass
class EmbeddingResult:
    """Контейнер с эмбеддингами и метаданными их построения."""

    embeddings: Any = field(repr=False)
    texts_count: int
    embedding_dim: int
    pooling: str
    model_name: Optional[str]
    device: str
    batch_size: int


@dataclass
class TransformerPredictionResult:
    """Результат батчевого инференса transformer-классификатора."""

    logits: Any = field(repr=False)
    probabilities: Any = field(default=None, repr=False)
    predictions: Any = field(default=None, repr=False)
    predicted_labels: Optional[List[Any]] = None
    label_mapping: Optional[Dict[int, Any]] = None


def batch_iter(values: Sequence[Any], batch_size: int) -> Iterator[List[Any]]:
    """Разбивает последовательность на батчи фиксированного размера."""

    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше нуля.")

    for start in range(0, len(values), batch_size):
        yield list(values[start : start + batch_size])


def load_transformer_encoder(
    model_name_or_path: str,
    device: Optional[str] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, str]:
    """Загружает AutoTokenizer и AutoModel для извлечения эмбеддингов."""

    torch = _import_torch()
    transformers = _import_transformers()
    resolved_device = _resolve_device(device, torch)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        **(tokenizer_kwargs or {}),
    )
    model = transformers.AutoModel.from_pretrained(
        model_name_or_path,
        **(model_kwargs or {}),
    )
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def load_transformer_classifier(
    model_name_or_path: str,
    device: Optional[str] = None,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, str]:
    """Загружает AutoTokenizer и AutoModelForSequenceClassification."""

    torch = _import_torch()
    transformers = _import_transformers()
    resolved_device = _resolve_device(device, torch)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        **(tokenizer_kwargs or {}),
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        **(model_kwargs or {}),
    )
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def mean_pooling(token_embeddings: Any, attention_mask: Any) -> Any:
    """Усредняет токен-эмбеддинги с учетом attention mask."""

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    token_counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed_embeddings / token_counts


def cls_pooling(token_embeddings: Any) -> Any:
    """Возвращает CLS-эмбеддинг из тензора last_hidden_state."""

    return token_embeddings[:, 0]


def normalize_embeddings(embeddings: Any) -> Any:
    """L2-нормализует torch-тензор или numpy-массив по оси эмбеддингов."""

    if _is_torch_tensor(embeddings):
        torch = _import_torch()
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    numpy = _import_numpy()
    norms = numpy.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = numpy.maximum(norms, 1e-12)
    return embeddings / norms


def encode_texts(
    texts: Iterable[str],
    model_name_or_path: Optional[str] = None,
    tokenizer: Any = None,
    model: Any = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 256,
    pooling: str = "mean",
    normalize: bool = True,
    return_numpy: bool = True,
    return_result: bool = False,
    show_progress: bool = True,
    tqdm_desc: str = "Кодирование текстов",
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Кодирует тексты BERT-подобным энкодером из transformers.

    Передайте либо ``model_name_or_path``, либо уже загруженные ``tokenizer`` и
    ``model``. По умолчанию возвращается numpy-массив: одна строка на один текст.
    """

    texts_list = list(texts)
    if not texts_list:
        return _empty_embeddings(return_numpy=return_numpy, return_result=return_result)

    torch = _import_torch()
    if pooling not in {"mean", "cls"}:
        raise ValueError("pooling должен быть 'mean' или 'cls'.")

    if tokenizer is None or model is None:
        if model_name_or_path is None:
            raise ValueError(
                "Передайте model_name_or_path или одновременно tokenizer и model "
                "в encode_texts()."
            )
        tokenizer, model, resolved_device = load_transformer_encoder(
            model_name_or_path=model_name_or_path,
            device=device,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )
    else:
        resolved_device = _resolve_device(device, torch)
        model.to(resolved_device)
        model.eval()

    batches = batch_iter(texts_list, batch_size=batch_size)
    progress = _progress_iter(
        batches,
        enabled=show_progress,
        desc=tqdm_desc,
        total=_num_batches(len(texts_list), batch_size),
    )

    encoded_batches = []
    with torch.no_grad():
        for batch in progress:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state

            if pooling == "mean":
                embeddings = mean_pooling(token_embeddings, inputs["attention_mask"])
            else:
                embeddings = cls_pooling(token_embeddings)

            if normalize:
                embeddings = normalize_embeddings(embeddings)

            encoded_batches.append(embeddings.detach().cpu())

    embeddings_tensor = torch.cat(encoded_batches, dim=0)
    embeddings = embeddings_tensor.numpy() if return_numpy else embeddings_tensor

    if not return_result:
        return embeddings

    return EmbeddingResult(
        embeddings=embeddings,
        texts_count=len(texts_list),
        embedding_dim=int(embeddings_tensor.shape[1]),
        pooling=pooling,
        model_name=model_name_or_path,
        device=str(resolved_device),
        batch_size=batch_size,
    )


def encode_train_test_texts(
    train_texts: Iterable[str],
    test_texts: Iterable[str],
    model_name_or_path: Optional[str] = None,
    tokenizer: Any = None,
    model: Any = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 256,
    pooling: str = "mean",
    normalize: bool = True,
    show_progress: bool = True,
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Кодирует обучающие и тестовые тексты одной и той же transformer-моделью."""

    torch = _import_torch()
    if tokenizer is None or model is None:
        if model_name_or_path is None:
            raise ValueError(
                "Передайте model_name_or_path или одновременно tokenizer и model "
                "в "
                "encode_train_test_texts()."
            )
        tokenizer, model, resolved_device = load_transformer_encoder(
            model_name_or_path=model_name_or_path,
            device=device,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )
    else:
        resolved_device = _resolve_device(device, torch)
        model.to(resolved_device)
        model.eval()

    X_train = encode_texts(
        train_texts,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
        normalize=normalize,
        show_progress=show_progress,
        tqdm_desc="Кодирование обучающей выборки",
    )
    X_test = encode_texts(
        test_texts,
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
        normalize=normalize,
        show_progress=show_progress,
        tqdm_desc="Кодирование тестовой выборки",
    )
    return X_train, X_test


def encode_texts_sentence_transformer(
    texts: Iterable[str],
    model_name_or_model: Any = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = True,
    convert_to_numpy: bool = True,
    encode_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Кодирует тексты через sentence-transformers, если пакет установлен."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Для encode_texts_sentence_transformer() нужен sentence-transformers. "
            "Установите его командой `pip install sentence-transformers`."
        ) from exc

    model = (
        SentenceTransformer(model_name_or_model)
        if isinstance(model_name_or_model, str)
        else model_name_or_model
    )
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=convert_to_numpy,
        **(encode_kwargs or {}),
    )


def predict_transformer_classifier(
    texts: Iterable[str],
    model_name_or_path: Optional[str] = None,
    tokenizer: Any = None,
    model: Any = None,
    device: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 256,
    return_numpy: bool = True,
    show_progress: bool = True,
    tqdm_desc: str = "Предсказание текстов",
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> TransformerPredictionResult:
    """Предсказывает классы BERT-подобной моделью классификации последовательностей."""

    texts_list = list(texts)
    if not texts_list:
        return _empty_prediction_result(return_numpy=return_numpy)

    torch = _import_torch()
    if tokenizer is None or model is None:
        if model_name_or_path is None:
            raise ValueError(
                "Передайте model_name_or_path или одновременно tokenizer и model "
                "в "
                "predict_transformer_classifier()."
            )
        tokenizer, model, resolved_device = load_transformer_classifier(
            model_name_or_path=model_name_or_path,
            device=device,
            tokenizer_kwargs=tokenizer_kwargs,
            model_kwargs=model_kwargs,
        )
    else:
        resolved_device = _resolve_device(device, torch)
        model.to(resolved_device)
        model.eval()

    batches = batch_iter(texts_list, batch_size=batch_size)
    progress = _progress_iter(
        batches,
        enabled=show_progress,
        desc=tqdm_desc,
        total=_num_batches(len(texts_list), batch_size),
    )

    logits_batches = []
    with torch.no_grad():
        for batch in progress:
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(resolved_device) for key, value in inputs.items()}
            outputs = model(**inputs)
            logits_batches.append(outputs.logits.detach().cpu())

    logits = torch.cat(logits_batches, dim=0)
    probabilities = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    id2label = getattr(getattr(model, "config", None), "id2label", None)
    label_mapping = _normalize_id2label(id2label)
    predicted_labels = None
    if label_mapping:
        predicted_labels = [label_mapping[int(index)] for index in predictions.tolist()]

    if return_numpy:
        logits = logits.numpy()
        probabilities = probabilities.numpy()
        predictions = predictions.numpy()

    return TransformerPredictionResult(
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        predicted_labels=predicted_labels,
        label_mapping=label_mapping,
    )


def _empty_embeddings(return_numpy: bool, return_result: bool) -> Any:
    embeddings = _import_numpy().empty((0, 0)) if return_numpy else []
    if not return_result:
        return embeddings

    return EmbeddingResult(
        embeddings=embeddings if return_numpy else [],
        texts_count=0,
        embedding_dim=0,
        pooling="mean",
        model_name=None,
        device="unknown",
        batch_size=0,
    )


def _empty_prediction_result(return_numpy: bool) -> TransformerPredictionResult:
    if return_numpy:
        numpy = _import_numpy()
        empty_logits = numpy.empty((0, 0))
        empty_predictions = numpy.empty((0,), dtype=int)
        return TransformerPredictionResult(
            logits=empty_logits,
            probabilities=empty_logits,
            predictions=empty_predictions,
            predicted_labels=[],
            label_mapping=None,
        )

    return TransformerPredictionResult(
        logits=[],
        probabilities=[],
        predictions=[],
        predicted_labels=[],
        label_mapping=None,
    )


def _num_batches(total: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше нуля.")
    return (total + batch_size - 1) // batch_size


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


def _resolve_device(device: Optional[str], torch: Any) -> str:
    if device is not None:
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_id2label(id2label: Any) -> Optional[Dict[int, Any]]:
    if not id2label:
        return None
    return {int(index): label for index, label in dict(id2label).items()}


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch")


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Для transformer-утилит нужен torch. "
            "Установите его командой `pip install torch`."
        ) from exc

    return torch


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Для BERT-подобных моделей нужен transformers. "
            "Установите его командой `pip install transformers`."
        ) from exc

    return transformers


def _import_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "Для numpy-эмбеддингов нужен numpy. "
            "Установите его командой `pip install numpy`."
        ) from exc

    return numpy

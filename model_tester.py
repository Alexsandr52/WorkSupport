"""Утилиты для быстрого сравнения моделей классификации.

Модуль рассчитан на уже подготовленные признаки: например TF-IDF, эмбеддинги
или любые другие числовые представления текстов.
"""

from __future__ import annotations

import importlib.util
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


MetricDict = Dict[str, Any]

__all__ = [
    "ModelRunResult",
    "ModelTestResult",
    "evaluate_classifiers",
]


@dataclass
class ModelRunResult:
    """Полный результат проверки одной модели."""

    name: str
    model: Any = field(default=None, repr=False)
    status: str = "ok"
    error: Optional[str] = None
    fit_time_sec: Optional[float] = None
    predict_time_sec: Optional[float] = None
    y_pred: Any = field(default=None, repr=False)
    y_proba: Any = field(default=None, repr=False)
    classification_report_dict: Optional[MetricDict] = field(default=None, repr=False)
    classification_report_text: Optional[str] = field(default=None, repr=False)
    confusion_matrix: Any = field(default=None, repr=False)
    metrics: MetricDict = field(default_factory=dict)


@dataclass
class ModelTestResult:
    """Сводный результат проверки нескольких моделей."""

    runs: Dict[str, ModelRunResult]
    summary: List[MetricDict]
    best_model_name: Optional[str]

    def to_frame(self) -> Any:
        """Возвращает summary как pandas.DataFrame, если pandas установлен."""

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Для ModelTestResult.to_frame() нужен pandas. "
                "Установите его или используйте список `.summary` напрямую."
            ) from exc

        return pd.DataFrame(self.summary)


@dataclass
class _SkippedModel:
    reason: str


def evaluate_classifiers(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    models: Optional[Mapping[str, Any]] = None,
    include_baselines: bool = True,
    baseline_params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    labels: Optional[Iterable[Any]] = None,
    target_names: Optional[Iterable[str]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    error_policy: str = "warn",
    return_predictions: bool = True,
    show_progress: bool = True,
) -> ModelTestResult:
    """Обучает и оценивает несколько моделей классификации на готовом тестовом наборе.

    Параметры
    ----------
    X_train, X_test, y_train, y_test:
        Уже подготовленные train/test данные. Векторизацию текста или построение
        эмбеддингов нужно выполнить до вызова функции.
    models:
        Пользовательские модели в формате ``{"name": estimator}``. Модели
        должны поддерживать sklearn-подобные методы ``fit`` и ``predict``.
    include_baselines:
        Добавляет базовые LogisticRegression, RandomForestClassifier и
        CatBoostClassifier, если нужные зависимости доступны.
    baseline_params:
        Переопределения параметров для базовых моделей. Поддерживаемые ключи:
        ``"logistic_regression"``, ``"random_forest"`` и ``"catboost"``.
    labels, target_names:
        Передаются в ``classification_report`` и ``confusion_matrix``.
    error_policy:
        ``"warn"`` сохраняет ошибки отдельных моделей и продолжает прогон.
        ``"raise"`` пробрасывает первую ошибку наружу.
    return_predictions:
        Сохраняет предсказания и вероятности в результате каждой модели.
    show_progress:
        Показывает progress bar через tqdm, если tqdm установлен.
    """

    if error_policy not in {"warn", "raise"}:
        raise ValueError("error_policy должен быть 'warn' или 'raise'.")

    metrics_mod = _import_sklearn_metrics()
    model_map = _build_model_map(
        models=models,
        include_baselines=include_baselines,
        baseline_params=baseline_params,
        random_state=random_state,
        n_jobs=n_jobs,
        error_policy=error_policy,
    )
    labels_list = list(labels) if labels is not None else None
    target_names_list = list(target_names) if target_names is not None else None

    runs: Dict[str, ModelRunResult] = {}
    summary: List[MetricDict] = []

    model_items = _progress_iter(
        model_map.items(),
        enabled=show_progress,
        desc="Оценка моделей",
        total=len(model_map),
    )
    for name, model in model_items:
        run = _evaluate_one_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            labels=labels_list,
            target_names=target_names_list,
            metrics_mod=metrics_mod,
            error_policy=error_policy,
            return_predictions=return_predictions,
        )
        runs[name] = run
        summary.append(_summary_from_run(run))

    best_model_name = _select_best_model_name(runs)
    return ModelTestResult(
        runs=runs,
        summary=summary,
        best_model_name=best_model_name,
    )


def _build_model_map(
    models: Optional[Mapping[str, Any]],
    include_baselines: bool,
    baseline_params: Optional[Mapping[str, Mapping[str, Any]]],
    random_state: int,
    n_jobs: int,
    error_policy: str,
) -> Dict[str, Any]:
    model_map: Dict[str, Any] = {}

    if include_baselines:
        model_map.update(
            _create_baseline_models(
                baseline_params=baseline_params or {},
                random_state=random_state,
                n_jobs=n_jobs,
                error_policy=error_policy,
            )
        )

    if models:
        duplicate_names = set(model_map).intersection(models)
        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Имена моделей не должны повторяться: {duplicates}")
        model_map.update(models)

    if not model_map:
        raise ValueError("Нет моделей для оценки. Передайте models или включите базовые модели.")

    return model_map


def _create_baseline_models(
    baseline_params: Mapping[str, Mapping[str, Any]],
    random_state: int,
    n_jobs: int,
    error_policy: str,
) -> Dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "Для создания базовых моделей нужен scikit-learn. "
            "Установите scikit-learn или вызовите evaluate_classifiers(..., "
            "include_baselines=False, models=...)."
        ) from exc

    baselines: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            **_merge_params(
                {
                    "max_iter": 1000,
                    "n_jobs": n_jobs,
                    "random_state": random_state,
                },
                baseline_params.get("logistic_regression"),
            )
        ),
        "random_forest": RandomForestClassifier(
            **_merge_params(
                {
                    "n_estimators": 300,
                    "n_jobs": n_jobs,
                    "random_state": random_state,
                },
                baseline_params.get("random_forest"),
            )
        ),
    }

    catboost_model = _create_catboost_model(
        params=baseline_params.get("catboost"),
        random_state=random_state,
        error_policy=error_policy,
    )
    baselines["catboost"] = catboost_model

    return baselines


def _create_catboost_model(
    params: Optional[Mapping[str, Any]],
    random_state: int,
    error_policy: str,
) -> Any:
    if importlib.util.find_spec("catboost") is None:
        message = "catboost не установлен; базовая CatBoostClassifier будет пропущена."
        if error_policy == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return _SkippedModel(reason=message)
        raise ImportError(message)

    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        **_merge_params(
            {
                "verbose": False,
                "random_seed": random_state,
            },
            params,
        )
    )


def _evaluate_one_model(
    name: str,
    model: Any,
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    labels: Optional[Iterable[Any]],
    target_names: Optional[Iterable[str]],
    metrics_mod: Any,
    error_policy: str,
    return_predictions: bool,
) -> ModelRunResult:
    run = ModelRunResult(name=name, model=model)

    if isinstance(model, _SkippedModel):
        run.status = "skipped"
        run.error = model.reason
        return run

    try:
        started_at = time.perf_counter()
        model.fit(X_train, y_train)
        run.fit_time_sec = time.perf_counter() - started_at

        started_at = time.perf_counter()
        y_pred = model.predict(X_test)
        run.predict_time_sec = time.perf_counter() - started_at

        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
            except Exception as exc:  # noqa: BLE001 - вероятности не обязательны для отчета.
                warnings.warn(
                    f"Для модели '{name}' не удалось получить predict_proba: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        report_kwargs = {
            "labels": labels,
            "target_names": target_names,
            "zero_division": 0,
        }
        run.classification_report_dict = metrics_mod.classification_report(
            y_test,
            y_pred,
            output_dict=True,
            **report_kwargs,
        )
        run.classification_report_text = metrics_mod.classification_report(
            y_test,
            y_pred,
            output_dict=False,
            **report_kwargs,
        )
        run.confusion_matrix = metrics_mod.confusion_matrix(
            y_test,
            y_pred,
            labels=labels,
        )
        run.metrics = _extract_main_metrics(run.classification_report_dict)

        if return_predictions:
            run.y_pred = y_pred
            run.y_proba = y_proba

    except Exception as exc:  # noqa: BLE001 - проброс контролируется через error_policy.
        if error_policy == "raise":
            raise
        run.status = "error"
        run.error = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Модель '{name}' завершилась с ошибкой: {run.error}", RuntimeWarning, stacklevel=2)

    return run


def _extract_main_metrics(report: MetricDict) -> MetricDict:
    macro_avg = report.get("macro avg", {})
    weighted_avg = report.get("weighted avg", {})
    return {
        "accuracy": report.get("accuracy"),
        "macro_precision": macro_avg.get("precision"),
        "macro_recall": macro_avg.get("recall"),
        "macro_f1": macro_avg.get("f1-score"),
        "weighted_precision": weighted_avg.get("precision"),
        "weighted_recall": weighted_avg.get("recall"),
        "weighted_f1": weighted_avg.get("f1-score"),
    }


def _summary_from_run(run: ModelRunResult) -> MetricDict:
    row: MetricDict = {
        "model": run.name,
        "status": run.status,
        "error": run.error,
        "fit_time_sec": run.fit_time_sec,
        "predict_time_sec": run.predict_time_sec,
    }
    row.update(run.metrics)
    return row


def _select_best_model_name(runs: Mapping[str, ModelRunResult]) -> Optional[str]:
    successful_runs = [
        run
        for run in runs.values()
        if run.status == "ok" and run.metrics.get("weighted_f1") is not None
    ]
    if not successful_runs:
        return None

    return max(successful_runs, key=lambda run: run.metrics["weighted_f1"]).name


def _merge_params(
    defaults: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    return merged


def _import_sklearn_metrics() -> Any:
    try:
        from sklearn import metrics
    except ImportError as exc:
        raise ImportError(
            "Для оценки классификаторов нужен scikit-learn. "
            "Установите его командой `pip install scikit-learn`."
        ) from exc

    return metrics


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

#!/usr/bin/env python3
"""
Проверка целостности и корректности изображений и аннотаций.

Проверяет:
- Что изображения можно прочитать (не битые)
- Соответствие размеров ожидаемым
- Наличие пары изображение ↔ аннотация
- Корректность формата YOLO-разметки
- Диапазон значений координат (все в [0, 1])
- Соответствие class_id допустимому диапазону
- Пустые файлы аннотаций (отрицательные примеры)
- Дубликаты имён файлов

Использование:
    python check_dataset_integrity.py --root unified_dataset
    python check_dataset_integrity.py --root GC10-DET --labels GC10-DET/labels
"""

from pathlib import Path
import argparse
import sys
import cv2
import numpy as np
from collections import Counter, defaultdict
from typing import Optional


# ============================================================
# Конфигурация
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".npy"}

EXPECTED_IMAGE_SHAPES = {
    "gc10": (1000, 2048, 1),   # height, width, channels (grayscale)
    "severstal": (256, 1600, 3),  # height, width, channels (stored as RGB)
    "unified": (256, 256, 1),   # после предобработки: 256x256 grayscale
}

NUM_CLASSES = 14
CLASS_NAMES = [
    "crease", "crescent_gap", "inclusion", "oil_spot",
    "punching_hole", "rolled_pit", "silk_spot",
    "waist_folding", "water_spot", "welding_line",
    "severstal_pitted_surface", "severstal_crazing",
    "severstal_scratches", "severstal_patches",
]


# ============================================================
# Результаты проверки
# ============================================================

class CheckResult:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def add_error(self, msg: str):
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def add_info(self, msg: str):
        self.infos.append(msg)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    @property
    def has_issues(self) -> bool:
        return self.has_errors or self.has_warnings

    def summary(self) -> str:
        lines = [
            f"Ошибок: {len(self.errors)}",
            f"Предупреждений: {len(self.warnings)}",
            f"Информационных сообщений: {len(self.infos)}",
        ]
        if self.errors:
            lines.append("\n--- ОШИБКИ ---")
            for e in self.errors:
                lines.append(f"  [E] {e}")
        if self.warnings:
            lines.append("\n--- ПРЕДУПРЕЖДЕНИЯ ---")
            for w in self.warnings:
                lines.append(f"  [W] {w}")
        if self.infos:
            lines.append("\n--- ИНФО ---")
            for i in self.infos:
                lines.append(f"  [I] {i}")
        return "\n".join(lines)


# ============================================================
# Функции проверки
# ============================================================

def check_image_readable(image_path: Path) -> tuple[bool, Optional[np.ndarray], Optional[str]]:
    """Проверяет, можно ли прочитать изображение."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return False, None, f"cv2.imread вернул None (повреждённый файл или неподдерживаемый формат)"
        return True, img, None
    except Exception as exc:
        return False, None, f"Исключение при чтении: {exc}"


def check_image_dimensions(
    img: np.ndarray,
    image_path: Path,
    dataset_type: str,
    result: CheckResult
) -> None:
    """Проверяет размеры изображения."""
    expected = EXPECTED_IMAGE_SHAPES.get(dataset_type)
    if expected is None:
        return

    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]

    exp_h, exp_w, exp_c = expected

    if (h, w) != (exp_h, exp_w):
        result.add_error(
            f"{image_path.name}: размер {w}x{h}, ожидалось {exp_w}x{exp_h}"
        )
    if c != exp_c:
        result.add_warning(
            f"{image_path.name}: каналов {c}, ожидалось {exp_c}"
        )


def check_yolo_label(
    label_path: Path,
    num_classes: int,
    result: CheckResult
) -> list[list[float]]:
    """Проверяет файл аннотации YOLO. Возвращает список валидных боксов."""
    labels = []

    if not label_path.exists():
        # Файл аннотации отсутствует — это не обязательно ошибка
        # (может быть отрицательный пример без дефектов)
        return labels

    content = label_path.read_text(encoding="utf-8")
    if not content.strip():
        return labels  # пустой файл = отрицательный пример

    for line_no, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            result.add_error(
                f"{label_path.name}:{line_no}: ожидалось 5 значений, получено {len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as exc:
            result.add_error(
                f"{label_path.name}:{line_no}: нечисловые значения ({exc})"
            )
            continue

        # Проверка class_id
        if not (0 <= class_id < num_classes):
            result.add_error(
                f"{label_path.name}:{line_no}: class_id={class_id} вне диапазона [0, {num_classes - 1}]"
            )

        # Проверка координат
        if not (0.0 <= x_center <= 1.0):
            result.add_error(
                f"{label_path.name}:{line_no}: x_center={x_center} вне [0, 1]"
            )
        if not (0.0 <= y_center <= 1.0):
            result.add_error(
                f"{label_path.name}:{line_no}: y_center={y_center} вне [0, 1]"
            )
        if not (0.0 < width <= 1.0):
            result.add_error(
                f"{label_path.name}:{line_no}: width={width} должен быть в (0, 1]"
            )
        if not (0.0 < height <= 1.0):
            result.add_error(
                f"{label_path.name}:{line_no}: height={height} должен быть в (0, 1]"
            )

        # Проверка что бокс не выходит за границы изображения
        eps = 1e-6
        if x_center - width / 2 < -eps or x_center + width / 2 > 1.0 + eps:
            result.add_error(
                f"{label_path.name}:{line_no}: бокс выходит за границы по X"
            )
        if y_center - height / 2 < -eps or y_center + height / 2 > 1.0 + eps:
            result.add_error(
                f"{label_path.name}:{line_no}: бокс выходит за границы по Y"
            )

        labels.append([class_id, x_center, y_center, width, height])

    return labels


def check_dataset(
    root: str,
    labels_folder: Optional[str] = None,
    dataset_type: str = "unified",
    num_classes: int = NUM_CLASSES,
) -> CheckResult:
    """
    Основная функция проверки целостности датасета.
    """
    result = CheckResult()
    root_path = Path(root)

    if not root_path.is_dir():
        result.add_error(f"Корневая папка не найдена: {root}")
        return result

    # Определяем папку с изображениями
    images_dir = root_path / "images"
    if not images_dir.is_dir():
        # Возможно, это плоская структура (GC10-DET без split)
        images_dir = root_path

    # Определяем папку с метками
    if labels_folder is None:
        labels_dir = root_path / "labels"
    else:
        labels_dir = Path(labels_folder)

    # Собираем все изображения
    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_files.extend(sorted(images_dir.rglob(f"*{ext}")))

    # Также ищем в подпапках split (train/val/test)
    if not image_files:
        for split in ["train", "val", "test"]:
            split_dir = images_dir / split
            if split_dir.is_dir():
                for ext in SUPPORTED_IMAGE_EXTENSIONS:
                    image_files.extend(sorted(split_dir.rglob(f"*{ext}")))

    if not image_files:
        result.add_error(f"Изображения не найдены в {images_dir}")
        return result

    result.add_info(f"Найдено изображений: {len(image_files)}")

    # Проверяем каждое изображение
    label_stems = set()
    if labels_dir.is_dir():
        for f in labels_dir.rglob("*.txt"):
            label_stems.add(f.stem)

    seen_stems = Counter()
    class_counter = Counter()
    images_with_labels = 0
    images_without_labels = 0
    total_boxes = 0

    for img_path in image_files:
        stem = img_path.stem
        seen_stems[stem] += 1

        # 1. Проверка читаемости
        readable, img, err = check_image_readable(img_path)
        if not readable:
            result.add_error(f"{img_path.name}: не читается — {err}")
            continue

        # 2. Проверка размеров
        check_image_dimensions(img, img_path, dataset_type, result)

        # 3. Проверка пары с аннотацией
        if labels_dir.is_dir():
            # Ищем label в той же split-подпапке, если есть
            label_file = None
            for split in ["train", "val", "test"]:
                candidate = labels_dir / split / f"{stem}.txt"
                if candidate.is_file():
                    label_file = candidate
                    break
            if label_file is None:
                candidate = labels_dir / f"{stem}.txt"
                if candidate.is_file():
                    label_file = candidate

            if label_file is not None:
                images_with_labels += 1
                labels = check_yolo_label(label_file, num_classes, result)
                total_boxes += len(labels)
                for l in labels:
                    class_counter[int(l[0])] += 1
            else:
                images_without_labels += 1
                result.add_warning(
                    f"{img_path.name}: нет файла аннотации в {labels_dir}"
                )
        else:
            images_without_labels += 1

    # 4. Проверка дублей имён
    duplicates = {s: c for s, c in seen_stems.items() if c > 1}
    if duplicates:
        result.add_error(f"Дубликаты имён файлов: {len(duplicates)}")
        for stem, count in duplicates.items():
            result.add_error(f"  {stem}: {count} раз")

    # 5. Проверка annotation files без изображений
    if labels_dir.is_dir():
        image_stems = {f.stem for f in image_files}
        orphan_labels = label_stems - image_stems
        if orphan_labels:
            result.add_warning(
                f"Аннотаций без изображений: {len(orphan_labels)}"
            )
            for stem in sorted(orphan_labels)[:10]:
                result.add_warning(f"  {stem}.txt")

    # 6. Итоговая статистика
    result.add_info(f"Изображений с аннотациями: {images_with_labels}")
    result.add_info(f"Изображений без аннотаций: {images_without_labels}")
    result.add_info(f"Всего боксов: {total_boxes}")

    if class_counter:
        result.add_info("Распределение по классам:")
        for cid in sorted(class_counter.keys()):
            name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"unknown_{cid}"
            result.add_info(f"  {name} ({cid}): {class_counter[cid]}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Проверка целостности и корректности изображений и аннотаций"
    )
    parser.add_argument("--root", required=True, help="Корневая папка датасета")
    parser.add_argument(
        "--labels", default=None,
        help="Папка с аннотациями (по умолчанию <root>/labels)"
    )
    parser.add_argument(
        "--type", default="unified",
        choices=["gc10", "severstal", "unified"],
        help="Тип датасета для проверки размеров изображений"
    )
    parser.add_argument(
        "--num-classes", type=int, default=NUM_CLASSES,
        help="Количество классов для валидации class_id"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Завершаться с ошибкой также при наличии предупреждений"
    )
    args = parser.parse_args()

    result = check_dataset(
        root=args.root,
        labels_folder=args.labels,
        dataset_type=args.type,
        num_classes=args.num_classes,
    )

    print(result.summary())

    if result.has_errors:
        print("\n❌ Проверка не пройдена (есть ошибки)")
        sys.exit(1)
    elif result.has_warnings and args.strict:
        print("\n⚠️  Проверка пройдена с предупреждениями (режим --strict)")
        sys.exit(1)
    else:
        print("\n✅ Проверка пройдена успешно")
        sys.exit(0)


if __name__ == "__main__":
    main()
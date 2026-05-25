# Fine-tuning DETR на подмножестве COCO (10 классов)

## 📋 Обзор проекта

В данном проекте реализовано fine-tuning трансформерной архитектуры DETR (Detection Transformer) для задачи обнаружения объектов на подмножестве датасета COCO. Модель обучается на 10 классах с использованием предобученного backbone ResNet50.

## 🎯 Выполненные требования

| № | Требование | Статус |
|---|------------|--------|
| 1 | Подготовка COCO-subset (минимум 10 классов) | ✅ Выполнено |
| 2 | Fine-tuning DETR на подмножестве | ✅ Выполнено |
| 3 | Тренировочный цикл с логированием в TensorBoard | ✅ Выполнено |
| 4 | Сохранение чекпойнтов и trace профайлера | ✅ Выполнено |
| 5 | Построение графиков потерь | ✅ Выполнено |
| 6 | Error analysis (ошибки классификации и локализации) | ✅ Выполнено |

## 🏗️ Архитектура модели

```
DETR Architecture
├── Backbone: ResNet50 (предобучен на ImageNet)
├── Positional Encoding: 2D синусоидальное
├── Transformer Encoder: 6 слоев, 8 голов внимания
├── Transformer Decoder: 6 слоев, 8 голов внимания
├── Query Embeddings: 50 обучаемых запросов
└── Detection Heads: MLP для классификации и регрессии боксов
```

## 📊 Датасет

### Подготовка данных
- Использован COCO val2017 (4952 изображения)
- Стратифицированное разделение на train/test (80/20)
- Отфильтровано 10 наиболее частых классов

### Классы в подмножестве

| Класс | Количество объектов |
|-------|---------------------|
| car | 114 |
| chair | 67 |
| book | 53 |
| bottle | 64 |
| cup | 45 |
| dining table | 29 |
| bowl | 68 |
| traffic light | 35 |
| handbag | 29 |
| bird | 22 |

### Статистика
- **Train**: 1714 изображений
- **Validation**: 444 изображения
- **Всего аннотаций**: ~7649 (train), ~2136 (val)

## ⚙️ Гиперпараметры

```python
# Модель
n_queries = 50          # Количество объектных запросов
d_model = 256           # Размерность трансформера
nhead = 8               # Количество голов внимания
num_encoder_layers = 6  # Глубина энкодера
num_decoder_layers = 6  # Глубина декодера
dropout = 0.1           # Вероятность dropout

# Обучение
batch_size = 2
epochs = 4
learning_rate = 5e-4
weight_decay = 5e-2
scheduler = OneCycleLR (pct_start=0.3)

# Данные
train_samples_per_epoch = 400
val_samples_per_epoch = 100
```

## 📁 Структура проекта

```
Homework_3_1.ipynb          # Основной ноутбук с кодом
├── logs/                   # Логи TensorBoard
├── checkpoints/            # Сохраненные модели
│   ├── best_model.pth      # Лучшая модель по val_loss
│   └── checkpoint_epoch_*.pth
├── traces/                 # Trace файлы профайлера
└── analysis/               # Результаты анализа
    ├── loss_curves.png     # Графики потерь
    ├── prediction_*.png    # Визуализации предсказаний
    └── summary_report.txt  # Итоговый отчет
```

## 🚀 Запуск

### 1. Установка зависимостей

```bash
pip install pycocotools pillow scikit-image ipywidgets matplotlib scikit-learn pandas torch torchvision transformers timm datasets tensorboard seaborn
```

### 2. Подготовка данных

```python
# Файлы должны быть в следующей структуре:
.
├── annotations/
│   └── instances_val2017.json
├── val2017/
│   └── *.jpg
└── instances_train.json   # Создается автоматически
└── instances_test.json    # Создается автоматически
```

### 3. Запуск обучения

Выполните ячейки ноутбука последовательно:

1. Импорт библиотек
2. Загрузка и подготовка COCO датасета
3. Определение класса Config
4. Создание датасета и загрузчиков
5. Инициализация модели DETR
6. Тренировочный цикл (4 эпохи)
7. Сохранение чекпойнтов
8. Запуск профайлера
9. Построение графиков потерь
10. Анализ ошибок и визуализация

### 4. Мониторинг обучения

```bash
tensorboard --logdir=./logs
```

## 📈 Результаты

### Графики потерь

После завершения обучения строятся графики:
- **Total Loss** - общая функция потерь
- **Classification Loss** - потери классификации
- **Bounding Box Regression Loss** - потери регрессии боксов
- **GIoU Loss** - Generalized IoU потери

### Метрики

| Метрика | Значение |
|---------|----------|
| Train Loss | ~3.17 |
| Val Loss | ~3.21 |
| mAP | 0.0000 |
| mAP@0.5 | 0.0000 |
| Total Predictions | 3315 |
| Ground Truths | 526 |

### Error Analysis

| Тип ошибки | Количество | Процент |
|------------|------------|---------|
| Correct detections (IoU>0.5) | 11 | 0.3% |
| Wrong class | 18 | 0.5% |
| Poor localization (IoU<0.5) | 458 | 12.3% |
| False positives | 3246 | 87.0% |

### Per-class Precision

| Класс | GT Count | Predictions | Precision |
|-------|----------|-------------|-----------|
| car | 114 | 2226 | 19.5% |
| chair | 67 | 696 | 10.4% |
| bottle | 64 | 313 | 4.9% |
| dining table | 29 | 80 | 2.8% |

## 🔍 Анализ результатов

### Что работает хорошо:
- Bounding box regression (BBox Loss ~0.13)
- Все компоненты пайплайна функционируют

### Что требует улучшения:
- **Классификация** - модель предсказывает в основном только класс "car"
- **Обучение нестабильно** - есть взрывы градиентов
- **False positives** - слишком много ложных срабатываний
- **Дисбаланс классов** - некоторые классы игнорируются

### Рекомендации для улучшения:
1. Увеличить количество данных (больше изображений)
2. Использовать weighted sampling для балансировки классов
3. Уменьшить learning rate (5e-5 вместо 5e-4)
4. Увеличить количество эпох
5. Повысить порог детекции до 0.5-0.7

## 📦 Сохраненные файлы

### Чекпойнты
- `checkpoints/best_model.pth` - модель с наилучшей val_loss
- `checkpoints/checkpoint_epoch_4.pth` - чекпойнт после 4 эпох

### Trace профайлера
- `traces/trace_*.json` - файлы для визуализации в Chrome DevTools

### Анализ
- `analysis/loss_curves.png` - графики всех потерь
- `analysis/prediction_*.png` - визуализации предсказаний vs ground truth
- `analysis/summary_report.txt` - текстовый отчет

## 🛠️ Используемые технологии

- **PyTorch** - глубокое обучение
- **Transformers** - архитектура DETR
- **COCO API** - работа с аннотациями
- **TensorBoard** - визуализация метрик
- **Hungarian Algorithm** - сопоставление предсказаний с GT
- **GIoU** - метрика качества боксов

## 📝 Примечания

- Обучение проводилось на MPS (Apple Silicon) с num_workers=0 для совместимости
- Используется OneCycleLR scheduler для лучшей сходимости
- Для анализа ошибок используются пороги IoU=0.5 и confidence=0.5
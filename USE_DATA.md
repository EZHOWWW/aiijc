# Specification for Data Loading and Usage (torch.dataset)

## 1. Overview

This document outlines the specification for a unified, flexible, and high-performance data handling pipeline. The primary goal is to create a standardized structure for using multimodal data, ensuring it is convenient for model training and evaluation.

The data pipeline must support:

- **CAD Models**: `.stl` files, point clouds.
- **Images**: Renders or photographs of models (`.png`/`.jpeg` files).
- **Text**: Descriptions of the models.

A key requirement is to handle complex relationships between data modalities, such as one-to-many, many-to-one, and many-to-many mappings (e.g., a single CAD model may have multiple corresponding images and text descriptions). These relationships must be clearly documented and easily managed within the dataset implementation. All data will be processed into a standardized format suitable for model consumption.

## 2. Data Structure

All datasets are stored on a dedicated hard disk drive (HDD) with the following root directory:
`root_dir="/mnt/new_disk/aiijc_data"`

The directory structure is organized as follows:

- **Models**: `/aiijc_data/<dataset_name>/[category]/models/<model_name>.stl`
- **Images (Renders/Photos)**: `/aiijc_data/<dataset_name>/[category]/images/<model_name>_[number].<jpeg/png>`
- **Text Descriptions**: `/aiijc_data/<dataset_name>/[category]/captions.json`

**Placeholders:**

- `<dataset_name>`: The name of the dataset (e.g., `CustomDataset`, `Thingi10k`).
- `[category]`: An optional subdirectory for the object category.
- `<model_name>`: The unique identifier for the model.
- `[number]`: An optional suffix for multiple images of the same model.

The `captions.json` file contains a dictionary mapping model names to their descriptions:

```json
{
    "model_name_1": "This is a description of the first model.",
    "model_name_2": [
        "This is the first description for the second model.",
        "This is another description for the second model."
    ]
}
```

## 3. Augmentations and Transformations

A flexible and explicit system for applying data augmentations must be implemented for all three modalities (3D models, images, and text). This allows for robust training and experimentation with different data transformations.

## 4. Dataloader

Given that the data is stored on a potentially slow HDD, the `Dataloader` implementation must be optimized for performance. It should provide a convenient and flexible batching mechanism to ensure fast data throughput during model training, minimizing I/O bottlenecks.

## 5. Dataset Splitting

The data should be partitioned into standard `train`, `validation`, and `test` sets to ensure proper model evaluation and hyperparameter tuning.

### Test Set Structure

The test set is organized to facilitate inference for specific cross-modal retrieval tasks. The expected directory format is as follows:

```
test/
├── queries_text_to_mesh/    # Text queries for retrieving meshes
├── gallery_mesh_for_text/   # The collection of meshes to search through

├── queries_mesh_to_text/    # Mesh queries for retrieving text
├── gallery_text_for_mesh/   # The collection of text descriptions to search through

# (Similar structures for image-based retrieval can be added as needed)
```

This structure ensures a convenient and standardized way to use the test data for model inference and evaluation.

## 6. Руководство по использованию (`use_data.py`)

Этот раздел объясняет, как использовать модуль `use_data.py` для загрузки мультимодальных данных и их использования в цикле обучения PyTorch.

### Основные компоненты

- **`MultimodalDataset`**: Основной класс `torch.utils.data.Dataset`, который сканирует директории, находит соответствующие модели (`.stl`), изображения и текстовые описания.
- **`create_dataloader`**: Вспомогательная функция, которая создает оптимизированный `torch.utils.data.DataLoader` для быстрой подачи данных в модель.

### Шаг 1: Определение трансформаций

Перед созданием датасета необходимо определить функции для аугментации и предобработки данных. `MultimodalDataset` принимает отдельные функции для каждой модальности.

```python
from torchvision import transforms
import torch

# 1. Трансформация для изображений (изменение размера, преобразование в тензор, нормализация)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. Трансформация для 3D-моделей (преобразование облака точек в тензор)
def point_cloud_transform(pc):
    return torch.from_numpy(pc).float()

# 3. Трансформация для текста (например, токенизация)
# text_transform = lambda text: tokenizer(text, ...)
```

### Шаг 2: Создание `Dataset` и `DataLoader`

Используйте `MultimodalDataset` для создания экземпляров датасета для обучающего, валидационного и тестового наборов данных. Затем оберните их в `DataLoader` с помощью функции `create_dataloader`.

```python
# Путь к корневой директории с данными
DATA_ROOT = "/mnt/new_disk/aiijc_data"

# Создание датасета для обучающего набора
train_dataset = MultimodalDataset(
    root_dir=DATA_ROOT,
    dataset_name="CustomDataset",  # или другое имя датасета
    transform_image=image_transform,
    transform_3d=point_cloud_transform
)

# Создание DataLoader'а
train_loader = create_dataloader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

### Шаг 3: Использование в цикле обучения

`DataLoader` возвращает словарь (`batch`), содержащий данные для каждой модальности. Вы можете использовать его непосредственно в вашем цикле обучения.

```python
# Предполагается, что у вас есть три энкодера: image_encoder, text_encoder, mesh_encoder
# и они уже перенесены на нужное устройство (device)

for batch in train_loader:
    # Перенос данных на GPU
    models = batch['model'].to(device)
    images = batch['images'].to(device)

    # Текстовые данные остаются в виде списка строк для токенизации
    texts = batch['texts']

    # Получение эмбеддингов от моделей
    # (Обратите внимание, что обработка текста может потребовать отдельного шага токенизации)
    model_embeddings = mesh_encoder(models)
    image_embeddings = image_encoder(images)
    # text_embeddings = text_encoder(tokenizer(texts, ...))

    # ... здесь ваша логика вычисления функции потерь (loss) ...
```

### Структура данных в батче

Словарь `batch`, возвращаемый `DataLoader`, имеет следующую структуру:

- `model_name` (`List[str]`): Список имен моделей в батче.
- `model` (`torch.Tensor`): Тензор с облаками точек.
    - **Форма**: `(B, N, 3)`, где `B` - размер батча, `N` - количество точек (по умолчанию 1024).
- `images` (`torch.Tensor`): Тензор с изображениями.
    - **Форма**: `(M, C, H, W)`, где `M` - _общее количество_ изображений для всех моделей в батче, `C` - каналы, `H` - высота, `W` - ширина.
- `texts` (`List[str]`): Плоский список всех текстовых описаний для моделей в батче.

> **Важно**: Так как одна модель может иметь несколько изображений или описаний, `collate_fn` объединяет их. Количество изображений (`M`) и текстов в батче может быть больше, чем размер батча (`B`). Это необходимо учитывать при вычислении contrastive loss.

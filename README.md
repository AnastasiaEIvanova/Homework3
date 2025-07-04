# Домашнее задание к уроку 3: Полносвязные сети

## Задание 1: Эксперименты с глубиной сети

### 1.1 Сравнение моделей разной глубины
__Создание модели с различным количеством слоев__

_Описание структуры:_
-  `input_size`: размер входных данных
-  `num_classes`: количество выходных классов
-  `layers`: список слоев, каждый из которых описан как словарь с типом слоя и его размером
```python
def create_configs():
    configs = {
        '1_layer': {
            'input_size': 784,
            'num_classes': 10,
            'layers': []
        },
        '2_layers': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 256},
                {'type': 'relu'}
            ]
        },
        '3_layers': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'}
            ]
        },
        '5_layers': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'}
            ]
        },
        '7_layers': {
            'input_size': 784,
            'num_classes': 10,
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'}
            ]
        }
    }
```
_Конфигурации включают:_
- `1_layer`: сеть без скрытых слоев.
- `2_layers`: один скрытый слой с 256 нейронами и активацией ReLU.
- `3_layers`: два скрытых слоя (первый с 256 нейронами и второй с 128 нейронами), оба с активацией ReLU.
- `5_layers`: четыре скрытых слоя с разными размерами (512, 256, 128, 64) и активацией ReLU.
- `7_layers`: шесть скрытых слоев с различными размерами (512, 512, 256, 256, 128, 64) и активацией ReLU.

__График точности на обучающей выборке:__
```python
plt.subplot(2, 2, 1)
for name, res in results.items():
    plt.plot(res['train_accs'], label=f'{name} (train)')
plt.title('Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
```
- `plt.subplot(2, 2, 1)`: создает первую ячейку в сетке 2x2
- Цикл for перебирает результаты (`results`), где для каждой модели (`name`) отображаются значения точности на обучающей выборке (`res['train_accs']`)

__График точности на тестовой выборке:__
```python
plt.subplot(2, 2, 2)
for name, res in results.items():
    plt.plot(res['test_accs'], label=f'{name} (test)')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
```
- `plt.subplot(2, 2, 2)`: создает вторую ячейку
- Аналогично первому графику, здесь отображаются значения точности на тестовой выборке (`res['test_accs']`)

__График потерь на обучающей выборке__
```python
plt.subplot(2, 2, 3)
for name, res in results.items():
    plt.plot(res['train_losses'], label=name)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
```
- `plt.subplot(2, 2, 3)`: создает третью ячейку
- Здесь отображаются значения потерь (`res['train_losses']`) для каждой модели

__График времени обучения:__
```python
plt.subplot(2, 2, 4)
avg_times = [res['avg_time_per_epoch'] for res in results.values()]
plt.bar(results.keys(), avg_times)
plt.title('Average Time per Epoch')
plt.ylabel('Seconds')
plt.xticks(rotation=45)
```
- `plt.subplot(2, 2, 4)`: создает четвертую ячейку
- Сначала вычисляется среднее время обучения на эпоху (`avg_time_per_epoch`) для каждой модели
- Затем строится столбчатая диаграмма с названиями моделей по оси X и временем по оси Y

### 1.2 Анализ переобучения
__Сравнение test accuracy__
```python
plt.subplot(1, 2, 1)
for name in ['3_layers', '3_layers_dropout']:
    if name in results:
        plt.plot(results[name]['test_accs'], label=name)
    elif name in reg_results:
        plt.plot(reg_results[name]['test_accs'], label=name)
plt.title('3 Layers vs 3 Layers with Dropout (Test Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
for name in ['5_layers', '5_layers_batchnorm']:
    if name in results:
        plt.plot(results[name]['test_accs'], label=name)
    elif name in reg_results:
        plt.plot(reg_results[name]['test_accs'], label=name)
plt.title('5 Layers vs 5 Layers with BatchNorm (Test Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
```
1. Создание подграфиков:
- `plt.subplot(1, 2, 1)` и `plt.subplot(1, 2, 2)` создают два подграфика в одной строке (1 ряд и 2 столбца). Первый подграфик будет для моделей с 3 слоями, а второй — для моделей с 5 слоями.

_Первый подграфик (3 слоя против 3 слоев с Dropout)_
2. Цикл для первой модели:
- `for name in ['3_layers', '3_layers_dropout']:`: перебирает имена моделей, которые будут отображаться на графике.
- `if name in results:`: проверяет, есть ли результаты для данной модели в словаре `results`.
- Если есть, `plt.plot(results[name]['test_accs'], label=name)` строит график точности тестирования для этой модели.
- `elif name in reg_results:`: если модель не найдена в results, проверяет наличие в `reg_results`.
- Если модель найдена, также строит график.

3. Настройка графика:
- `plt.title('3 Layers vs 3 Layers with Dropout (Test Accuracy)')`: устанавливает заголовок графика.
- `plt.xlabel('Epoch')` и `plt.ylabel('Accuracy')`: добавляют метки осей.
- `plt.legend()`: отображает легенду, показывающую, какая линия соответствует какой модели.

_Второй подграфик (5 слоев против 5 слоев с Batch Normalization)_
4. Цикл для второй модели:
- `for name in ['5_layers', '5_layers_batchnorm']:`: аналогично первому подграфику, перебирает имена моделей для второго графика.
- Логика проверки наличия результатов и построения графиков остается такой же, как и в первом подграфике.

5. Настройка графика:
- `plt.title('5 Layers vs 5 Layers with BatchNorm (Test Accuracy)')`: устанавливает заголовок для второго графика.
- Метки осей и легенда добавляются так же, как и в первом подграфике.

__Анализ переобучения__
```python
print("\nOverfitting Analysis:")
```
Эта строка выводит заголовок "Overfitting Analysis" в консоль, предваряя его пустой строкой для лучшей читаемости.

_Цикл по архитектурам:Цикл по архитектурам:_
```python
for name in ['3_layers', '5_layers', '7_layers']:
```
Здесь мы перебираем три архитектуры нейронных сетей: с 3, 5 и 7 слоями.

```python
    if name in results:
```
Проверяем, есть ли результаты для текущей модели в словаре `results`.

```python
        train_acc = results[name]['train_accs'][-1]
        test_acc = results[name]['test_accs'][-1]
```
Извлечение точности:
- `train_acc` — это точность на обучающей выборке, взятая из последнего элемента списка `train_accs`.
- `test_acc` — это точность на тестовой выборке, взятая из последнего элемента списка `test_accs`.

```python
        gap = train_acc - test_acc
```
Здесь вычисляется разница между точностью на обучающей и тестовой выборках, которая позволяет оценить степень переобучения. Большое значение `gap` указывает на то, что модель хорошо подстраивается под обучающие данные, но плохо обобщает на тестовых данных.

```python
        print(f"{name}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, Gap {gap:.4f}")
```
Строка формирует и выводит информацию о текущей модели: ее название, точность на обучающей выборке, точность на тестовой выборке и разницу между ними. Форматирование `:.4f` означает, что числа будут округлены до четырех знаков после запятой.

_Цикл по моделям с регуляризацией:_
```python
for name in reg_results.keys():
    train_acc = reg_results[name]['train_accs'][-1]
    test_acc = reg_results[name]['test_accs'][-1]
    gap = train_acc - test_acc
    print(f"{name}: Train Acc {train_acc:.4f}, Test Acc {test_acc:.4f}, Gap {gap:.4f}")
```
Здесь мы перебираем все модели, которые содержатся в словаре `reg_results`, который, как предполагается, содержит результаты моделей с применением различных техник регуляризации.


## Задание 2: Эксперименты с шириной сети

### 2.1 Сравнение моделей разной ширины
__Конфигурации моделей__
```python
configs = {
        'Узкие': [64, 32, 16],
        'Средние': [256, 128, 64],
        'Широкие': [1024, 512, 256],
        'Очень широкие': [2048, 1024, 512]
    }
```

__Генерация данных__
```python
 X, y = generate_data()
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    results = []
```

__Визуализация кривых обучения__
```python
plt.plot(val_accuracies, label=f"{name} ({total_params:,} параметров)")
```

__Вывод таблицы результатов__
```python
import pandas as pd
    df = pd.DataFrame(results)
    print("\nРезультаты сравнения моделей:")
    print(df[['Название', 'Архитектура', 'Параметры', 'Время обучения', 'Лучшая точность']])
```

### 2.2 Оптимизация архитектуры
```python
def optimize_architecture():
    # Параметры для grid search
    param_grid = {
        'first_layer': [64, 128, 256, 512],
        'second_layer': [64, 128, 256, 512],
        'third_layer': [64, 128, 256, 512],
        'scheme': ['расширение', 'сужение', 'постоянная']
    }

    # Фильтр для схем изменения ширины
    def is_valid_combination(params):
        if params['scheme'] == 'расширение':
            return params['first_layer'] < params['second_layer'] < params['third_layer']
        elif params['scheme'] == 'сужение':
            return params['first_layer'] > params['second_layer'] > params['third_layer']
        else:  # постоянная
            return params['first_layer'] == params['second_layer'] == params['third_layer']

    # Генерация данных
    X, y = generate_data()
    dataset = TensorDataset(X, y)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    results = []
    valid_params = [p for p in ParameterGrid(param_grid) if is_valid_combination(p)]

    for params in tqdm.tqdm(valid_params, desc="Grid Search"):
        hidden_sizes = [params['first_layer'], params['second_layer'], params['third_layer']]
        model = MLP(hidden_sizes=hidden_sizes)

        # Обучение (уменьшим количество эпох для ускорения)
        _, val_accuracies, training_time = train_model(model, train_loader, val_loader, epochs=10)

        results.append({
            'first_layer': params['first_layer'],
            'second_layer': params['second_layer'],
            'third_layer': params['third_layer'],
            'scheme': params['scheme'],
            'accuracy': max(val_accuracies),
            'time': training_time
        })

    # Анализ результатов
    import pandas as pd
    df = pd.DataFrame(results)

    # Визуализация heatmap для точности
    for scheme in ['расширение', 'сужение', 'постоянная']:
        scheme_df = df[df['scheme'] == scheme]
        if len(scheme_df) > 0:
            # Для постоянной схемы нужен особый подход
            if scheme == 'постоянная':
                plt.figure()
                plt.plot(scheme_df['first_layer'], scheme_df['accuracy'])
                plt.title(f'Точность для схемы "{scheme}"')
                plt.xlabel('Размер слоя')
                plt.ylabel('Точность')
            else:
                # Создаем pivot таблицу для heatmap
                pivot_df = scheme_df.pivot_table(index='first_layer',
                                                 columns='second_layer',
                                                 values='accuracy',
                                                 aggfunc='mean')
                plt.figure()
                sns.heatmap(pivot_df, annot=True, fmt=".3f")
                plt.title(f'Точность для схемы "{scheme}"')

    plt.show()

    # Вывод лучших комбинаций
    print("\nЛучшие комбинации параметров:")
    print(df.sort_values('accuracy', ascending=False).head(10))
```
_Оптимизация архитектуры:_
Проводится grid search по:
- Размерам каждого из 3 слоев (64, 128, 256, 512)
- Схемам изменения ширины:
  - Расширение (размеры слоев увеличиваются)
  - Сужение (размеры слоев уменьшаются)
  - Постоянная (все слои одинакового размера)

Результаты визуализируются с помощью heatmap (для расширяющихся/сужающихся схем) и графиков (для постоянных схем).
Выводятся 10 лучших комбинаций параметров.

_Визуализация:_
- Используются matplotlib и seaborn для графиков и heatmap.
- Результаты представлены в табличном формате с помощью pandas.

## Задание 3: Эксперименты с регуляризацией

### 3.1 Сравнение техник регуляризации
```python
def compare_regularization_techniques():
    techniques = [
        ("No regularization", {'dropout_rate': 0.0, 'use_batchnorm': False, 'weight_decay': 0.0}),
        ("Dropout 0.1", {'dropout_rate': 0.1, 'use_batchnorm': False, 'weight_decay': 0.0}),
        ("Dropout 0.3", {'dropout_rate': 0.3, 'use_batchnorm': False, 'weight_decay': 0.0}),
        ("Dropout 0.5", {'dropout_rate': 0.5, 'use_batchnorm': False, 'weight_decay': 0.0}),
        ("BatchNorm only", {'dropout_rate': 0.0, 'use_batchnorm': True, 'weight_decay': 0.0}),
        ("Dropout 0.3 + BatchNorm", {'dropout_rate': 0.3, 'use_batchnorm': True, 'weight_decay': 0.0}),
        ("L2 regularization", {'dropout_rate': 0.0, 'use_batchnorm': False, 'weight_decay': 0.001}),
    ]

    results = {}
    models = []

    for name, params in techniques:
        print(f"\nTraining {name}")
        model = BaseNet(**params).to(device)
        train_losses, test_accuracies = train_and_evaluate(
            model, train_loader, test_loader, num_epochs, learning_rate, params['weight_decay']
        )
        results[name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1]
        }
        models.append(model)

    # Визуализация результатов
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['train_losses'], label=name)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, res in results.items():
        plt.plot(res['test_accuracies'], label=name)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Визуализация распределения весов
    plot_weight_distributions(models, [name for name, _ in techniques])

    # Вывод финальных точностей
    print("\nFinal Accuracies:")
    for name, res in results.items():
        print(f"{name}: {res['final_accuracy']:.2f}%")
```

1. Определение техник регуляризации
```python
techniques = [
    ("No regularization", {'dropout_rate': 0.0, 'use_batchnorm': False, 'weight_decay': 0.0}),
    ("Dropout 0.1", {'dropout_rate': 0.1, 'use_batchnorm': False, 'weight_decay': 0.0}),
    ("Dropout 0.3", {'dropout_rate': 0.3, 'use_batchnorm': False, 'weight_decay': 0.0}),
    ("Dropout 0.5", {'dropout_rate': 0.5, 'use_batchnorm': False, 'weight_decay': 0.0}),
    ("BatchNorm only", {'dropout_rate': 0.0, 'use_batchnorm': True, 'weight_decay': 0.0}),
    ("Dropout 0.3 + BatchNorm", {'dropout_rate': 0.3, 'use_batchnorm': True, 'weight_decay': 0.0}),
    ("L2 regularization", {'dropout_rate': 0.0, 'use_batchnorm': False, 'weight_decay': 0.001}),
]
```
Здесь определяется список техник регуляризации, каждая из которых представлена кортежем с именем техники и параметрами, которые будут использоваться для создания модели.\
Параметры включают уровень dropout, использование нормализации по батчам (Batch Normalization) и значение веса для L2-регуляризации (`weight_decay`).

2. Инициализация контейнеров для результатов
```python
results = {}
models = []
```
- `results` — это словарь для хранения результатов обучения и тестирования для каждой техники.
- `models` — список для хранения экземпляров моделей, созданных для каждой техники.

3. Обучение и оценка моделей
```python
for name, params in techniques:
    print(f"\nTraining {name}")
    model = BaseNet(**params).to(device)
    train_losses, test_accuracies = train_and_evaluate(
        model, train_loader, test_loader, num_epochs, learning_rate, params['weight_decay']
    )
    results[name] = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'final_accuracy': test_accuracies[-1]
    }
    models.append(model)
```
Для каждой техники:
- Выводится сообщение о начале обучения.
- Создается модель `BaseNet` с заданными параметрами и переносится на устройство (например, GPU).
- Вызывается функция `train_and_evaluate`, которая обучает модель и возвращает потери на обучающем наборе и точности на тестовом наборе.
- Результаты сохраняются в словаре `results`.
- Модель добавляется в список `models`.
  
4. Визуализация результатов
```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, res in results.items():
    plt.plot(res['train_losses'], label=name)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
for name, res in results.items():
    plt.plot(res['test_accuracies'], label=name)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
```
- Создается фигура для визуализации результатов.
- В первом подграфике отображаются потери на обучающем наборе в зависимости от эпохи.
- Во втором подграфике отображается точность на тестовом наборе.
- Используется `plt.legend()` для добавления легенды к графикам.

5. Визуализация распределения весов
```python
plot_weight_distributions(models, [name for name, _ in techniques])
```
Вызывается функция `plot_weight_distributions`, которая визуализирует распределение весов для всех обученных моделей.

6. Вывод финальных точностей   
```python
print("\nFinal Accuracies:")
for name, res in results.items():
    print(f"{name}: {res['final_accuracy']:.2f}%")
```
Выводятся финальные точности для каждой техники регуляризации с форматированием до двух знаков после запятой.

### 3.2 Адаптивная регуляризация
```python
class AdaptiveNet(nn.Module):
    def __init__(self, initial_dropout=0.1, final_dropout=0.5,
                 batchnorm_momentum=0.1, layer_specific_reg=False):
        super(AdaptiveNet, self).__init__()
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.batchnorm_momentum = batchnorm_momentum
        self.layer_specific_reg = layer_specific_reg

        # Конволюционные слои
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=batchnorm_momentum)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=batchnorm_momentum)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.bn3 = nn.BatchNorm1d(512, momentum=batchnorm_momentum)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(512, 10)

        # Инициализация dropout
        self.current_dropout_rate = initial_dropout
        self.dropout1 = nn.Dropout2d(initial_dropout)
        self.dropout2 = nn.Dropout2d(initial_dropout)
        self.dropout3 = nn.Dropout(initial_dropout)

    def update_dropout(self, epoch, total_epochs):
        # Линейное увеличение dropout rate
        self.current_dropout_rate = self.initial_dropout + (self.final_dropout - self.initial_dropout) * (
                epoch / total_epochs)
        self.dropout1.p = self.current_dropout_rate
        self.dropout2.p = self.current_dropout_rate
        self.dropout3.p = self.current_dropout_rate

        if self.layer_specific_reg:
            # Разные dropout rates для разных слоев
            self.dropout1.p = min(0.2, self.current_dropout_rate)
            self.dropout2.p = min(0.4, self.current_dropout_rate)
            self.dropout3.p = min(0.3, self.current_dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc2(x)
        return x


def evaluate_adaptive_techniques():
    techniques = [
        ("Fixed Dropout 0.3", {'initial_dropout': 0.3, 'final_dropout': 0.3,
                               'batchnorm_momentum': 0.1, 'layer_specific_reg': False}),
        ("Increasing Dropout 0.1-0.5", {'initial_dropout': 0.1, 'final_dropout': 0.5,
                                        'batchnorm_momentum': 0.1, 'layer_specific_reg': False}),
        ("High BatchNorm Momentum 0.9", {'initial_dropout': 0.3, 'final_dropout': 0.3,
                                         'batchnorm_momentum': 0.9, 'layer_specific_reg': False}),
        ("Layer-specific Dropout", {'initial_dropout': 0.1, 'final_dropout': 0.5,
                                    'batchnorm_momentum': 0.1, 'layer_specific_reg': True}),
        ("Combined Adaptive", {'initial_dropout': 0.1, 'final_dropout': 0.4,
                               'batchnorm_momentum': 0.5, 'layer_specific_reg': True}),
    ]

    results = {}

    for name, params in techniques:
        print(f"\nTraining {name}")
        model = AdaptiveNet(**params).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_losses = []
        test_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            model.update_dropout(epoch, num_epochs)

            running_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            train_losses.append(epoch_loss)

            # Оценка на тестовом наборе
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            test_accuracies.append(accuracy)
            print(
                f"Epoch {epoch + 1}, Dropout: {model.current_dropout_rate:.2f}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

        results[name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'final_accuracy': test_accuracies[-1]
        }

    # Визуализация результатов
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, res in results.items():
        plt.plot(res['train_losses'], label=name)
    plt.title('Training Loss (Adaptive Techniques)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, res in results.items():
        plt.plot(res['test_accuracies'], label=name)
    plt.title('Test Accuracy (Adaptive Techniques)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Вывод финальных точностей
    print("\nFinal Accuracies (Adaptive Techniques):")
    for name, res in results.items():
        print(f"{name}: {res['final_accuracy']:.2f}%")
```
__Класс AdaptiveNet__
1. Инициализация
```python
def __init__(self, initial_dropout=0.1, final_dropout=0.5,
             batchnorm_momentum=0.1, layer_specific_reg=False):
```
Этот метод инициализирует параметры сети, включая начальный и конечный уровни dropout, моментум для нормализации по батчам и флаг для включения специфичной для слоя регуляризации.

2. Конволюционные слои
```python
self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
self.bn1 = nn.BatchNorm2d(32, momentum=batchnorm_momentum)
self.relu1 = nn.ReLU()

self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
self.bn2 = nn.BatchNorm2d(64, momentum=batchnorm_momentum)
self.relu2 = nn.ReLU()
```
Определяются два свёрточных слоя с использованием `nn.Conv2d`, каждый из которых сопровождается слоем нормализации по батчам (`nn.BatchNorm2d`) и активацией ReLU (`nn.ReLU`).

3. Пулинг и полносвязные слои
```python
self.pool = nn.MaxPool2d(2, 2)

self.fc1 = nn.Linear(64 * 16 * 16, 512)
self.bn3 = nn.BatchNorm1d(512, momentum=batchnorm_momentum)
self.relu3 = nn.ReLU()

self.fc2 = nn.Linear(512, 10)
```
Добавляется слой максимального пулинга (`MaxPool2d`) для уменьшения размерности. Определяются два полносвязных слоя (`nn.Linear`), также с использованием нормализации по батчам и активации ReLU.

4. Dropout
```python
self.current_dropout_rate = initial_dropout
self.dropout1 = nn.Dropout2d(initial_dropout)
self.dropout2 = nn.Dropout2d(initial_dropout)
self.dropout3 = nn.Dropout(initial_dropout)
```
- Инициализируются слои dropout с начальным значением `initial_dropout`.

__Метод `update_dropout`__
```python
def update_dropout(self, epoch, total_epochs):
```
- Этот метод обновляет значение dropout в зависимости от текущей эпохи.
- Оно линейно увеличивается от `initial_dropout` до `final_dropout`.
- Если включена специфичная для слоя регуляризация, устанавливаются разные значения dropout для каждого слоя.

__Метод `forward`__
```python
def forward(self, x):
```
- Этот метод описывает прямое распространение данных через сеть.
- Данные проходят через свёрточные слои, нормализацию, активацию и dropout.
- После свёрточных слоев данные преобразуются в вектор и проходят через полносвязные слои.

__Функция `evaluate_adaptive_techniques`__
```python
def evaluate_adaptive_techniques():
```
- Эта функция предназначена для оценки различных техник адаптивной регуляризации.
- Определяется список техник с соответствующими параметрами.

__Цикл обучения__
```python
for name, params in techniques:
    model = AdaptiveNet(**params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```
Для каждой техники создается экземпляр модели `AdaptiveNet`, настраиваются функции потерь и оптимизаторы.

__Процесс обучения__
```python
for epoch in range(num_epochs):
    model.train()
    model.update_dropout(epoch, num_epochs)
```
В каждой эпохе модель переводится в режим обучения и обновляет значение dropout.

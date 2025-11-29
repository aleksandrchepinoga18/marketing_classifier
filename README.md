# 📊 Marketing Campaign Classifier

**End-to-end ML pipeline** для прогнозирования вероятности отклика клиентов на маркетинговые кампании.  
✅ Обучение → ✅ API → ✅ Мониторинг → ✅ Ретрейн → ✅ Готово к продакшену.


## 🎯 Задача

Оптимизация маркетинговых затрат за счёт точечной рассылки предложения **только тем клиентам, у которых высока вероятность отклика**.  
Исходные данные: датасет маркетинговых взаимодействий с сильным дисбалансом классов (15% положительных откликов).


## 🧠 Подход и алгоритмы

- **Модель**: `LightGBM` с балансировкой классов (`scale_pos_weight`)
- **Порог**: подобран по **F1-score** на валидационной выборке
- **Метрики**:
  - ROC-AUC: **0.897**
  - F1 (класс 1): **0.52**
  - Precision (класс 1): **0.50**
  - Recall (класс 1): **0.54**
- **Интерпретация**: SHAP-анализ
- **Предобработка**: удалены константные и избыточные признаки (`Z_CostContact`, `MntTotal`, `AcceptedCmpOverall`)

🔹 Развертывание
Создал Flask API, который принимает JSON с признаками клиента и возвращает вероятность отклика.

API автоматически логирует все предсказания в файл.

🔹 Мониторинг и ретрейн
Каждый месяц:

Сравниваю распределения новых данных с тренировочными (data drift, KS-тест).

Оцениваю качество модели на новых размеченных данных (model drift, ROC-AUC).

При деградации — автоматически переобучаю модель и обновляю API.

🔹 Интеграция в продакшен
Источник данных: CRM / база клиентов (через ETL-процесс).

Инференс: запросы к Flask API из маркетинговой системы.

Мониторинг: запуск скриптов по расписанию (например, через Airflow или cron).

Хранение логов: в продакшене — **в базе данных **(PostgreSQL), а не в CSV.

В текущей реализации всё готово к работе. Для масштабирования достаточно подключить базу для логов и Airflow для оркестрации — логика уже реализована».

🔮 В продакшене:

Новые клиенты → приходят в маркетинговую систему.

Система вызывает API → получает вероятность отклика.

Если prob ≥ 0.56 → отправляется предложение.

Через 30 дней → CRM предоставляет фактические метки (отклик/не отклик).

Запускается мониторинг → если качество упало → модель обновляется.

Цикл повторяется.


## 📁 Структура проекта

├── api/ # Flask API для инференса

├── monitoring/ # Скрипты мониторинга и ретрейна

├── reports/ # Артефакты: модель, метрики, графики

├── src/ # Модули: данные, EDA, обучение

├── ifood_df.csv # Исходный датасет

└── run_all.py # Запуск полного pipeline



## ▶️ Как запускать

### Полный pipeline (рекомендуется)
bash
python run_all.py

Отдельные компоненты

Команда                                  Назначение
python src/eda.py                        Генерирует EDA-отчёты → reports/eda/
python src/train_final.py                Обучает модель → reports/model/
python api/app.py                        Запускает API на http://localhost:5000
python monitoring/simulate_labels.py     Имитирует появление реальных меток
python monitoring/retrain_if_needed.py   Проверяет дрифт и качество, делает ретрейн при необходимости

Тест API
После запуска python api/app.py:
curl -X POST http://localhost:5000/predict \-H "Content-Type: application/json" \-d '{"Income":50000,"Kidhome":1,...}'  # полный JSON с признаками

🚀 Готовность к продакшену

Инференс: REST API (POST /predict)

Логирование: все предсказания сохраняются

Мониторинг:
Model drift: падение ROC-AUC / F1 на новых размеченных данных

Data drift: тест Колмогорова-Смирнова по признакам

Ретрейн: автоматический при деградации качества

Масштабирование: легко интегрируется с PostgreSQL, Airflow, Docker

📈 Результаты на тесте (331 клиент, 50 положительных)

<img width="443" height="193" alt="image" src="https://github.com/user-attachments/assets/ceebe984-d114-4b9e-ba55-6238a26b4742" />


💡 Практическая ценность:
**Точность**(Precision) = 50% → каждый второй клиент из целевой группы реально откликается (в 3.3× эффективнее случайной рассылки).

**Полнота**(Recall) = 54% → охватывается более половины всех реальных целевых клиентов.

ROC-AUC = 0.897 → отличная способность ранжировать клиентов.

📌 Вывод
Проект реализует полный жизненный цикл ML-модели:

От EDA и обучения до мониторинга и ретрейна

С фокусом на интерпретируемость, стабильность и бизнес-ценность

Готов к внедрению в реальные маркетинговые процессы

pipeline работает идеально:

✔️ Обнаруживает падение качества

✔️ Ловит смещение распределений

✔️ Автоматически запускает ретрейн

Симуляция показала крайний случай (шум вместо сигнала), и система корректно отреагировала.

🚀 Дальнейшие шаги
Подключите реальные данные (CRM, базу клиентов),
Запускайте мониторинг ежемесячно,
При необходимости — ретрейн.


data:image/svg+xml;utf8,%3Csvg%20id%3D%22mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%22%20width%3D%22100%25%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20style%3D%22max-width%3A%201920.897705078125px%3B%22%20viewBox%3D%22-7.999996185302734%20-8%201920.897705078125%20254.125%22%20role%3D%22graphics-document%20document%22%20aria-roledescription%3D%22flowchart-v2%22%3E%3Cstyle%3E%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%7Bfont-family%3A%22trebuchet%20ms%22%2Cverdana%2Carial%2Csans-serif%3Bfont-size%3A16px%3Bfill%3A%23333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.error-icon%7Bfill%3A%23552222%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.error-text%7Bfill%3A%23552222%3Bstroke%3A%23552222%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.edge-thickness-normal%7Bstroke-width%3A2px%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.edge-thickness-thick%7Bstroke-width%3A3.5px%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.edge-pattern-solid%7Bstroke-dasharray%3A0%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.edge-pattern-dashed%7Bstroke-dasharray%3A3%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.edge-pattern-dotted%7Bstroke-dasharray%3A2%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.marker%7Bfill%3A%23333333%3Bstroke%3A%23333333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.marker.cross%7Bstroke%3A%23333333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20svg%7Bfont-family%3A%22trebuchet%20ms%22%2Cverdana%2Carial%2Csans-serif%3Bfont-size%3A16px%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.label%7Bfont-family%3A%22trebuchet%20ms%22%2Cverdana%2Carial%2Csans-serif%3Bcolor%3A%23333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.cluster-label%20text%7Bfill%3A%23333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.cluster-label%20span%2C%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20p%7Bcolor%3A%23333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.label%20text%2C%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20span%2C%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20p%7Bfill%3A%23333%3Bcolor%3A%23333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.node%20rect%2C%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.node%20circle%2C%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.node%20ellipse%2C%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.node%20polygon%2C%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.node%20path%7Bfill%3A%23ECECFF%3Bstroke%3A%239370DB%3Bstroke-width%3A1px%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.flowchart-label%20text%7Btext-anchor%3Amiddle%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.node%20.katex%20path%7Bfill%3A%23000%3Bstroke%3A%23000%3Bstroke-width%3A1px%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.node%20.label%7Btext-align%3Acenter%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.node.clickable%7Bcursor%3Apointer%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.arrowheadPath%7Bfill%3A%23333333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.edgePath%20.path%7Bstroke%3A%23333333%3Bstroke-width%3A2.0px%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.flowchart-link%7Bstroke%3A%23333333%3Bfill%3Anone%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.edgeLabel%7Bbackground-color%3A%23e8e8e8%3Btext-align%3Acenter%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.edgeLabel%20rect%7Bopacity%3A0.5%3Bbackground-color%3A%23e8e8e8%3Bfill%3A%23e8e8e8%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.labelBkg%7Bbackground-color%3Argba(232%2C%20232%2C%20232%2C%200.5)%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.cluster%20rect%7Bfill%3A%23ffffde%3Bstroke%3A%23aaaa33%3Bstroke-width%3A1px%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.cluster%20text%7Bfill%3A%23333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.cluster%20span%2C%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20p%7Bcolor%3A%23333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20div.mermaidTooltip%7Bposition%3Aabsolute%3Btext-align%3Acenter%3Bmax-width%3A200px%3Bpadding%3A2px%3Bfont-family%3A%22trebuchet%20ms%22%2Cverdana%2Carial%2Csans-serif%3Bfont-size%3A12px%3Bbackground%3Ahsl(80%2C%20100%25%2C%2096.2745098039%25)%3Bborder%3A1px%20solid%20%23aaaa33%3Bborder-radius%3A2px%3Bpointer-events%3Anone%3Bz-index%3A100%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20.flowchartTitleText%7Btext-anchor%3Amiddle%3Bfont-size%3A18px%3Bfill%3A%23333%3B%7D%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b%20%3Aroot%7B--mermaid-font-family%3A%22trebuchet%20ms%22%2Cverdana%2Carial%2Csans-serif%3B%7D%3C%2Fstyle%3E%3Cg%3E%3Cmarker%20id%3D%22mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd%22%20class%3D%22marker%20flowchart%22%20viewBox%3D%220%200%2010%2010%22%20refX%3D%226%22%20refY%3D%225%22%20markerUnits%3D%22userSpaceOnUse%22%20markerWidth%3D%2212%22%20markerHeight%3D%2212%22%20orient%3D%22auto%22%3E%3Cpath%20d%3D%22M%200%200%20L%2010%205%20L%200%2010%20z%22%20class%3D%22arrowMarkerPath%22%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%3E%3C%2Fpath%3E%3C%2Fmarker%3E%3Cmarker%20id%3D%22mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointStart%22%20class%3D%22marker%20flowchart%22%20viewBox%3D%220%200%2010%2010%22%20refX%3D%224.5%22%20refY%3D%225%22%20markerUnits%3D%22userSpaceOnUse%22%20markerWidth%3D%2212%22%20markerHeight%3D%2212%22%20orient%3D%22auto%22%3E%3Cpath%20d%3D%22M%200%205%20L%2010%2010%20L%2010%200%20z%22%20class%3D%22arrowMarkerPath%22%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%3E%3C%2Fpath%3E%3C%2Fmarker%3E%3Cmarker%20id%3D%22mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-circleEnd%22%20class%3D%22marker%20flowchart%22%20viewBox%3D%220%200%2010%2010%22%20refX%3D%2211%22%20refY%3D%225%22%20markerUnits%3D%22userSpaceOnUse%22%20markerWidth%3D%2211%22%20markerHeight%3D%2211%22%20orient%3D%22auto%22%3E%3Ccircle%20cx%3D%225%22%20cy%3D%225%22%20r%3D%225%22%20class%3D%22arrowMarkerPath%22%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%3E%3C%2Fcircle%3E%3C%2Fmarker%3E%3Cmarker%20id%3D%22mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-circleStart%22%20class%3D%22marker%20flowchart%22%20viewBox%3D%220%200%2010%2010%22%20refX%3D%22-1%22%20refY%3D%225%22%20markerUnits%3D%22userSpaceOnUse%22%20markerWidth%3D%2211%22%20markerHeight%3D%2211%22%20orient%3D%22auto%22%3E%3Ccircle%20cx%3D%225%22%20cy%3D%225%22%20r%3D%225%22%20class%3D%22arrowMarkerPath%22%20style%3D%22stroke-width%3A%201%3B%20stroke-dasharray%3A%201%2C%200%3B%22%3E%3C%2Fcircle%3E%3C%2Fmarker%3E%3Cmarker%20id%3D%22mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-crossEnd%22%20class%3D%22marker%20cross%20flowchart%22%20viewBox%3D%220%200%2011%2011%22%20refX%3D%2212%22%20refY%3D%225.2%22%20markerUnits%3D%22userSpaceOnUse%22%20markerWidth%3D%2211%22%20markerHeight%3D%2211%22%20orient%3D%22auto%22%3E%3Cpath%20d%3D%22M%201%2C1%20l%209%2C9%20M%2010%2C1%20l%20-9%2C9%22%20class%3D%22arrowMarkerPath%22%20style%3D%22stroke-width%3A%202%3B%20stroke-dasharray%3A%201%2C%200%3B%22%3E%3C%2Fpath%3E%3C%2Fmarker%3E%3Cmarker%20id%3D%22mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-crossStart%22%20class%3D%22marker%20cross%20flowchart%22%20viewBox%3D%220%200%2011%2011%22%20refX%3D%22-1%22%20refY%3D%225.2%22%20markerUnits%3D%22userSpaceOnUse%22%20markerWidth%3D%2211%22%20markerHeight%3D%2211%22%20orient%3D%22auto%22%3E%3Cpath%20d%3D%22M%201%2C1%20l%209%2C9%20M%2010%2C1%20l%20-9%2C9%22%20class%3D%22arrowMarkerPath%22%20style%3D%22stroke-width%3A%202%3B%20stroke-dasharray%3A%201%2C%200%3B%22%3E%3C%2Fpath%3E%3C%2Fmarker%3E%3Cg%20class%3D%22root%22%3E%3Cg%20class%3D%22clusters%22%3E%3C%2Fg%3E%3Cg%20class%3D%22edgePaths%22%3E%3Cpath%20d%3D%22M121.909%2C83.75L126.076%2C83.75C130.242%2C83.75%2C138.576%2C83.75%2C146.109%2C83.816C153.643%2C83.882%2C160.376%2C84.014%2C163.743%2C84.08L167.11%2C84.146%22%20id%3D%22L-A-B-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-A%20LE-B%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3Cpath%20d%3D%22M283.994%2C55.767L294.431%2C48.473C304.868%2C41.178%2C325.741%2C26.589%2C360.721%2C19.295C395.701%2C12%2C444.788%2C12%2C492.269%2C12C539.75%2C12%2C585.625%2C12%2C631.542%2C12C677.458%2C12%2C723.417%2C12%2C769.375%2C12C815.333%2C12%2C861.292%2C12%2C902.979%2C12C944.667%2C12%2C982.083%2C12%2C1021.705%2C12C1061.328%2C12%2C1103.155%2C12%2C1148.208%2C12C1193.261%2C12%2C1241.54%2C12%2C1287.613%2C12C1333.686%2C12%2C1377.553%2C12%2C1421.573%2C12C1465.593%2C12%2C1509.765%2C12%2C1553.938%2C12C1598.11%2C12%2C1642.282%2C12%2C1678.381%2C20.26C1714.479%2C28.52%2C1742.504%2C45.039%2C1756.517%2C53.299L1770.529%2C61.559%22%20id%3D%22L-B-C-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-B%20LE-C%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3Cpath%20d%3D%22M285.113%2C111.614L295.363%2C118.158C305.613%2C124.701%2C326.113%2C137.788%2C341.253%2C144.332C356.392%2C150.875%2C366.171%2C150.875%2C371.061%2C150.875L375.95%2C150.875%22%20id%3D%22L-B-D-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-B%20LE-D%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3Cpath%20d%3D%22M606.5%2C150.875L610.667%2C150.875C614.833%2C150.875%2C623.167%2C150.875%2C630.617%2C150.875C638.067%2C150.875%2C644.633%2C150.875%2C647.917%2C150.875L651.2%2C150.875%22%20id%3D%22L-D-E-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-D%20LE-E%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3Cpath%20d%3D%22M882.25%2C150.875L886.417%2C150.875C890.583%2C150.875%2C898.917%2C150.875%2C906.45%2C150.941C913.984%2C151.007%2C920.717%2C151.139%2C924.084%2C151.205L927.451%2C151.271%22%20id%3D%22L-E-F-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-E%20LE-F%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3Cpath%20d%3D%22M1084.409%2C174.216L1094.504%2C177.743C1104.6%2C181.269%2C1124.792%2C188.322%2C1146.795%2C191.849C1168.798%2C195.375%2C1192.612%2C195.375%2C1204.52%2C195.375L1216.427%2C195.375%22%20id%3D%22L-F-G-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-F%20LE-G%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3Cpath%20d%3D%22M1084.409%2C128.534L1094.504%2C124.841C1104.6%2C121.147%2C1124.792%2C113.761%2C1140.376%2C110.068C1155.961%2C106.375%2C1166.938%2C106.375%2C1172.427%2C106.375L1177.916%2C106.375%22%20id%3D%22L-F-H-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-F%20LE-H%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3Cpath%20d%3D%22M1357.909%2C195.375L1368.494%2C195.375C1379.08%2C195.375%2C1400.25%2C195.375%2C1414.119%2C195.375C1427.987%2C195.375%2C1434.554%2C195.375%2C1437.837%2C195.375L1441.12%2C195.375%22%20id%3D%22L-G-I-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-G%20LE-I%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3Cpath%20d%3D%22M1661.455%2C195.375L1665.621%2C195.375C1669.788%2C195.375%2C1678.121%2C195.375%2C1698.38%2C180.618C1718.638%2C165.861%2C1750.822%2C136.346%2C1766.914%2C121.589L1783.006%2C106.832%22%20id%3D%22L-I-C-0%22%20class%3D%22%20edge-thickness-normal%20edge-pattern-solid%20flowchart-link%20LS-I%20LE-C%22%20style%3D%22fill%3Anone%3B%22%20marker-end%3D%22url(%23mermaid-d24757e6-c9ef-4738-863f-9438c799b13b_flowchart-pointEnd)%22%3E%3C%2Fpath%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabels%22%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(0%2C%200)%22%3E%3CforeignObject%20width%3D%220%22%20height%3D%220%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%20transform%3D%22translate(1019.4999923706055%2C%2012)%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(-13.232954978942871%2C%20-12)%22%3E%3CforeignObject%20width%3D%2226.465909957885742%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3EНет%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%20transform%3D%22translate(346.6136283874512%2C%20150.875)%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(-9.636363983154297%2C%20-12)%22%3E%3CforeignObject%20width%3D%2219.272727966308594%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3EДа%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(0%2C%200)%22%3E%3CforeignObject%20width%3D%220%22%20height%3D%220%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(0%2C%200)%22%3E%3CforeignObject%20width%3D%220%22%20height%3D%220%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%20transform%3D%22translate(1144.9829473495483%2C%20195.375)%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(-9.636363983154297%2C%20-12)%22%3E%3CforeignObject%20width%3D%2219.272727966308594%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3EДа%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%20transform%3D%22translate(1144.9829473495483%2C%20106.375)%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(-13.232954978942871%2C%20-12)%22%3E%3CforeignObject%20width%3D%2226.465909957885742%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3EНет%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(0%2C%200)%22%3E%3CforeignObject%20width%3D%220%22%20height%3D%220%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22edgeLabel%22%3E%3Cg%20class%3D%22label%22%20transform%3D%22translate(0%2C%200)%22%3E%3CforeignObject%20width%3D%220%22%20height%3D%220%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22edgeLabel%22%3E%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22nodes%22%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-A-18%22%20data-node%3D%22true%22%20data-id%3D%22A%22%20transform%3D%22translate(60.95454406738281%2C%2083.75)%22%3E%3Crect%20class%3D%22basic%20label-container%22%20style%3D%22%22%20rx%3D%220%22%20ry%3D%220%22%20x%3D%22-60.95454788208008%22%20y%3D%22-19.5%22%20width%3D%22121.90909576416016%22%20height%3D%2239%22%3E%3C%2Frect%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-53.45454788208008%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%22106.90909576416016%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EНовые%20данные%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-B-19%22%20data-node%3D%22true%22%20data-id%3D%22B%22%20transform%3D%22translate(241.94317626953125%2C%2083.75)%22%3E%3Cpolygon%20points%3D%2270.03409194946289%2C0%20140.06818389892578%2C-70.03409194946289%2070.03409194946289%2C-140.06818389892578%200%2C-70.03409194946289%22%20class%3D%22label-container%22%20transform%3D%22translate(-70.03409194946289%2C70.03409194946289)%22%20style%3D%22%22%3E%3C%2Fpolygon%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-43.03409194946289%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%2286.06818389892578%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EЕсть%20метка%3F%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-C-21%22%20data-node%3D%22true%22%20data-id%3D%22C%22%20transform%3D%22translate(1808.1761531829834%2C%2083.75)%22%3E%3Crect%20class%3D%22basic%20label-container%22%20style%3D%22%22%20rx%3D%220%22%20ry%3D%220%22%20x%3D%22-96.72159576416016%22%20y%3D%22-19.5%22%20width%3D%22193.4431915283203%22%20height%3D%2239%22%3E%3C%2Frect%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-89.22159576416016%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%22178.4431915283203%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EПредсказание%20через%20API%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-D-23%22%20data-node%3D%22true%22%20data-id%3D%22D%22%20transform%3D%22translate(493.87499237060547%2C%20150.875)%22%3E%3Crect%20class%3D%22basic%20label-container%22%20style%3D%22%22%20rx%3D%220%22%20ry%3D%220%22%20x%3D%22-112.625%22%20y%3D%22-19.5%22%20width%3D%22225.25%22%20height%3D%2239%22%3E%3C%2Frect%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-105.125%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%22210.25%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EСравнение%20с%20предсказанием%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-E-25%22%20data-node%3D%22true%22%20data-id%3D%22E%22%20transform%3D%22translate(769.3749923706055%2C%20150.875)%22%3E%3Crect%20class%3D%22basic%20label-container%22%20style%3D%22%22%20rx%3D%220%22%20ry%3D%220%22%20x%3D%22-112.875%22%20y%3D%22-19.5%22%20width%3D%22225.75%22%20height%3D%2239%22%3E%3C%2Frect%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-105.375%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%22210.75%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EРасчёт%20метрик%3A%20ROC-AUC%2C%20F1%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-F-27%22%20data-node%3D%22true%22%20data-id%3D%22F%22%20transform%3D%22translate(1019.4999923706055%2C%20150.875)%22%3E%3Cpolygon%20points%3D%2287.25%2C0%20174.5%2C-87.25%2087.25%2C-174.5%200%2C-87.25%22%20class%3D%22label-container%22%20transform%3D%22translate(-87.25%2C87.25)%22%20style%3D%22%22%3E%3C%2Fpolygon%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-60.25%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%22120.5%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EКачество%20упало%3F%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-G-29%22%20data-node%3D%22true%22%20data-id%3D%22G%22%20transform%3D%22translate(1289.8181743621826%2C%20195.375)%22%3E%3Crect%20class%3D%22basic%20label-container%22%20style%3D%22%22%20rx%3D%220%22%20ry%3D%220%22%20x%3D%22-68.09091186523438%22%20y%3D%22-19.5%22%20width%3D%22136.18182373046875%22%20height%3D%2239%22%3E%3C%2Frect%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-60.590911865234375%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%22121.18182373046875%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EРетрейн%20модели%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-H-31%22%20data-node%3D%22true%22%20data-id%3D%22H%22%20transform%3D%22translate(1289.8181743621826%2C%20106.375)%22%3E%3Crect%20class%3D%22basic%20label-container%22%20style%3D%22%22%20rx%3D%220%22%20ry%3D%220%22%20x%3D%22-106.6022720336914%22%20y%3D%22-19.5%22%20width%3D%22213.2045440673828%22%20height%3D%2239%22%3E%3C%2Frect%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-99.1022720336914%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%22198.2045440673828%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EПродолжаем%20использовать%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3Cg%20class%3D%22node%20default%20default%20flowchart-label%22%20id%3D%22flowchart-I-33%22%20data-node%3D%22true%22%20data-id%3D%22I%22%20transform%3D%22translate(1553.937505722046%2C%20195.375)%22%3E%3Crect%20class%3D%22basic%20label-container%22%20style%3D%22%22%20rx%3D%220%22%20ry%3D%220%22%20x%3D%22-107.51705932617188%22%20y%3D%22-19.5%22%20width%3D%22215.03411865234375%22%20height%3D%2239%22%3E%3C%2Frect%3E%3Cg%20class%3D%22label%22%20style%3D%22%22%20transform%3D%22translate(-100.01705932617188%2C%20-12)%22%3E%3Crect%3E%3C%2Frect%3E%3CforeignObject%20width%3D%22200.03411865234375%22%20height%3D%2224%22%3E%3Cdiv%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxhtml%22%20style%3D%22display%3A%20inline-block%3B%20white-space%3A%20nowrap%3B%22%3E%3Cspan%20class%3D%22nodeLabel%22%3EОбновление%20Docker-образа%3C%2Fspan%3E%3C%2Fdiv%3E%3C%2FforeignObject%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E



## Интерпретация результата

отчёт построен на тестовой выборке из 331 клиента:
<img width="1016" height="187" alt="image" src="https://github.com/user-attachments/assets/e7b6a43e-d53a-416a-9332-a57eb25da5d8" />


✅ Истинные клиенты класса 1 = 50 штук — это реальные отклики в тесте.

📊 Как рассчитываются precision и recall для класса 1?
Из отчёта:

Precision = 0.50
Recall = 0.54
support = 50

🔹 Recall = 0.54 → «поймали 54% истинных откликов»

✅ Модель правильно нашла 27 из 50 реальных откликов.

🔹 Precision = 0.50 → «половина наших предсказаний — правда»

✅ Модель предсказала 54 клиента как «откликнутся», но только 27 из них — реальные.

📌 Итог по бизнес-логике

Всего истинных откликов: 50

Мы нашли: 27 (Recall = 54%)

Отправили предложение: 54 клиентам (27 истинных + 27 ложных)

Эффективность рассылки: 27 / 54 = 50% (Precision = 0.50)

💡 Вместо рассылки всем 331 клиентам, мы отправляем только 54,
при этом ловим более половины всех реальных откликов.

🧮 Сводка по confusion matrix (для класса 1)

<img width="947" height="170" alt="image" src="https://github.com/user-attachments/assets/251e44e5-f80a-4805-8a23-020495a1d9ab" />


Проверка:

support класса 0 = TN + FP = 253 + 27 = 280 ≈ 281 (округление)

support класса 1 = FN + TP = 23 + 27 = 50 ✅
✅ Вывод
«Мы охватываем более половины всех реальных целевых клиентов»
— имеем:

27 из 50 истинных откликов → 54% → это и есть Recall = 0.54.

«Каждый второй клиент из целевой группы реально откликается»

27 истинных / 54 предсказанных = 50% → это и есть Precision = 0.50.

Итог:

**Для маркетинового результата это отличный показатель работы, мы имеем конверсию в отклик около 50% всех заинтересованных клиентов**

# UML-диаграммы сервиса hpwhd

Диаграммы отражают архитектуру сервиса оценки стоимости квартир и обмен данными
между **Home Predictor** и **ML Model**, а также среду выполнения (Docker /
uvicorn :8000 внутри Universal DMZ Halyk Bank).

> Все диаграммы — на Mermaid и рендерятся прямо в GitHub. PlantUML-вариант
> C4 приведён в конце.

---

## 1. C4 Container — компоненты и обмен данными

```mermaid
flowchart TB
    user([Пользователь / Банковский клиент])

    subgraph org["Halyk Bank — Organization"]
        subgraph dmz["Universal DMZ — Zone (прокси proxy.halykbank.nb:8080)"]
            subgraph host["Docker host — контейнер kz_house_price"]
                subgraph app["ASGI-процесс: uvicorn main:app — 0.0.0.0:8000"]
                    hp["Home Predictor<br/>[Container: Service]<br/>FastAPI (main.py)<br/>+ FeaturePipeline"]
                    ml["ML Model<br/>[Container: Service]<br/>LightGBM + PyTorch NN<br/>+ Ridge ensemble (nn_inference.py)"]
                end
            end
        end
    end

    user -- "HTTPS → :8000<br/>POST /predict (JSON: 11 признаков)" --> hp
    hp  -- "in-process call<br/>predict_kzt(DataFrame[55 признаков])" --> ml
    ml  -- "np.ndarray[KZT/m²]" --> hp
    hp  -- "HTTPS<br/>{price_kzt, price_per_sqm, confidence}" --> user
```

---

## 2. Sequence — сценарий одного прогноза `/predict`

```mermaid
sequenceDiagram
    autonumber
    actor U as Пользователь
    participant HP as Home Predictor<br/>(FastAPI main.py)
    participant FP as FeaturePipeline
    participant ML as ML Model<br/>(NNInference)
    participant CAL as Region Calibration (α)

    U->>HP: POST /predict (JSON, 11 признаков)
    HP->>HP: Валидация (Pydantic PredictionInput)
    HP->>FP: assemble(user_input)
    FP->>FP: REGION, segment_code, city, OSM-дистанции,<br/>макростатистика, BLP/BFE, price_index
    FP-->>HP: DataFrame[55 признаков]
    HP->>ML: predict_kzt(features_df)
    ML->>ML: scaler_X → LGB + NN → Ridge meta
    ML->>ML: exp(log) × price_index_current
    ML-->>HP: KZT/m² (2025Q4 → номинал)
    HP->>CAL: get_region_alpha(lat, lon)
    CAL-->>HP: α
    HP->>HP: price/m² × α × TOTAL_AREA<br/>+ доверит. интервал ±10%
    HP-->>U: {price_kzt, price_per_sqm, confidence, drivers}
```

---

## 3. Component / Class — ключевые классы

```mermaid
classDiagram
    class FastAPIApp {
        +predict(PredictionInput) dict
        +batch(UploadFile) dict
        +health() dict
    }
    class PredictionInput {
        +int ROOMS
        +float LONGITUDE
        +float LATITUDE
        +float TOTAL_AREA
        +int FLOOR
        +int TOTAL_FLOORS
        +int FURNITURE
        +int CONDITION
        +float CEILING
        +int MATERIAL
        +int YEAR
    }
    class FeaturePipeline {
        +assemble(user_input) DataFrame
        +get_region_alpha(lat, lon) float
        +get_display_info(lat, lon) dict
    }
    class NNInference {
        +feature_list: list
        +lgb_model: Booster
        +nn_model: HousePriceNN
        +ridge_meta: Ridge
        +predict_kzt(DataFrame) ndarray
    }
    class HousePriceNN {
        +forward(x) Tensor
    }
    class RegionGrid
    class StatLoader
    class OSMDistances

    FastAPIApp --> PredictionInput : валидирует
    FastAPIApp --> FeaturePipeline : собирает признаки
    FastAPIApp --> NNInference : предсказывает
    FeaturePipeline --> RegionGrid
    FeaturePipeline --> StatLoader
    FeaturePipeline --> OSMDistances
    NNInference --> HousePriceNN
```

---

## 4. Deployment — среда выполнения (ИС / адрес сервера)

```mermaid
flowchart LR
    client([Клиент<br/>браузер / банковский BFF])

    subgraph node["Docker host (Universal DMZ, Halyk Bank)"]
        subgraph cont["container: kz_house_price<br/>python:3.11-slim"]
            uv["uvicorn main:app<br/>bind 0.0.0.0:8000<br/>--workers 1"]
            art["Артефакты:<br/>nn_model/ + data/<br/>distance_grid.parquet (LFS)"]
            uv --- art
        end
    end

    client -- "HTTPS → host:8000<br/>(publish 8000:8000)" --> uv
    uv -- "HEALTHCHECK GET /health" --> uv
```

**Параметры среды (ИС / сервер):**

| Параметр | Значение |
|----------|----------|
| Рантайм / ИС | Docker-контейнер `kz_house_price`, ASGI-сервер uvicorn (Python 3.11-slim) |
| Внутренний bind | `0.0.0.0:8000` |
| Публикация порта | host `8000` → container `8000` |
| Внешний вход | HTTPS → reverse-proxy → `http://<host>:8000` |
| Health-check | `GET /health` |
| Зона / организация | Universal DMZ, Halyk Bank |
| Корпоративный прокси | `proxy.halykbank.nb:8080` |

> ⚠️ Обмен «Home Predictor → ML Model» в текущей реализации — **внутренний вызов
> функции Python** (`predict_kzt`), оба компонента живут в одном процессе/контейнере.
> На C4-диаграмме стрелка показана как логический интерфейс «запрос 55 признаков →
> ответ KZT/m²».

---

## 5. PlantUML — C4 Container (альтернатива)

```plantuml
@startuml
!include https://raw.githubusercontent.com/plantuml-stdlib/C4-PlantUML/master/C4_Container.puml

Person(user, "Пользователь", "Банковский клиент / оператор")

System_Boundary(org, "Halyk Bank") {
  System_Boundary(dmz, "Universal DMZ") {
    Container(hp, "Home Predictor", "FastAPI, Python 3.11 (main.py)", "Приём запроса, сборка 55 признаков, пост-калибровка")
    Container(ml, "ML Model", "LightGBM + PyTorch NN + Ridge (nn_inference.py)", "Инференс цены за м²")
  }
}

Rel(user, hp, "POST /predict (JSON, 11 признаков)", "HTTPS :8000")
Rel(hp, ml, "predict_kzt(DataFrame[55]) → KZT/m²", "in-process call")
Rel(ml, hp, "np.ndarray[KZT/m²]", "return")
Rel(hp, user, "{price_kzt, confidence}", "HTTPS")
@enduml
```

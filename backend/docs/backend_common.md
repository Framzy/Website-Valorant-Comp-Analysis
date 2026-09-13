# Valorant Predictor V2 — Backend Architecture & API Common Specification

## 1. Tujuan Dokumen

Dokumen ini berisi aturan umum backend yang berlaku untuk General Analysis API dan Team Prediction API.

Dokumen khusus endpoint dipisahkan menjadi:

- `api_general.md`
- `api_team.md`

---

# 2. Arsitektur

```text
Frontend / Postman
        |
        | HTTP
        v
+-----------------------+
|       app.py          |
|       API Layer       |
+-----------+-----------+
            |
            v
+-----------------------+
|       Services        |
|                       |
| GeneralAnalysisService|
| TeamPredictionService |
+-----------+-----------+
            |
            +----------------------+
            |                      |
            v                      v
   General Dataset          Team Dataset
                              |
                              +--> Feature Builder
                              |
                              +--> Feature Encoder
                              |
                              +--> Model Artifacts
```

---

# 3. Target Structure

```text
backend/
│
├── app.py
├── config.py
├── constants.py
│
├── data/
│   ├── valorant_dataset_general_v2.csv
│   └── valorant_dataset_team_v2.csv
│
├── ml/
│   └── team/
│       ├── feature_encoder.py
│       └── inference_feature_builder.py
│
├── models/
│   └── team_prediction_v2/
│       ├── team_model_v2.joblib
│       ├── encoders.joblib
│       ├── feature_names.joblib
│       └── metadata.json
│
└── services/
    ├── general_analysis_service.py
    └── team_prediction_service.py
```

---

# 4. Responsibility Boundary

## `app.py`

Bertanggung jawab untuk:

- menerima HTTP request,
- membaca JSON/query parameter,
- validasi dasar request structure,
- memanggil service,
- mengubah hasil menjadi HTTP response,
- memetakan error ke HTTP status.

`app.py` tidak melakukan business logic.

## `GeneralAnalysisService`

Bertanggung jawab atas seluruh business logic General Analysis.

Dataset di-load dan diprepare sekali saat service dibuat.

## `TeamPredictionService`

Bertanggung jawab atas:

- Team options,
- input validation,
- historical lookup,
- inference orchestration,
- prediction.

Dataset, model, encoder, dan feature names di-load sekali saat service dibuat.

## `ml/team/`

Berisi logic ML yang diperlukan saat inference:

- feature construction,
- feature encoding.

## `models/`

Berisi trained artifacts yang digunakan runtime.

## `training_v2/`

Berisi development/training pipeline.

Backend tidak boleh bergantung pada `training_v2`.

---

# 5. Runtime Lifecycle

Saat aplikasi dimulai:

```text
Application Start
       |
       +--> GeneralAnalysisService()
       |       |
       |       +--> Load dataset
       |       +--> Prepare dataset
       |
       +--> TeamPredictionService()
               |
               +--> Load dataset
               +--> Load model
               +--> Load encoders
               +--> Load feature names
               +--> Validate artifacts
```

Resource statis tidak di-load ulang pada setiap request.

---

# 6. HTTP Convention

## Success

```http
200 OK
```

## Client Error

```http
400 Bad Request
```

Digunakan untuk request/input yang tidak valid.

## Server Error

```http
500 Internal Server Error
```

Digunakan untuk unexpected internal exception.

Format error:

```json
{
  "error": "Descriptive error message"
}
```

Untuk unexpected production error:

```json
{
  "error": "Internal server error."
}
```

Detail internal tidak diekspos ke client.

---

# 7. Route Convention

General:

```http
POST /api/general/analyze
```

Team:

```http
GET  /api/team/options/years
GET  /api/team/options/maps
GET  /api/team/options/teams
POST /api/team/predict
```

Semua API menggunakan prefix:

```text
/api
```

---

# 8. `app.py` Design

Route harus sesederhana:

```text
HTTP Request
    ↓
Basic request parsing
    ↓
Service method
    ↓
Result
    ↓
JSON Response
```

Tidak boleh menjadi tempat untuk:

- `pandas` data processing,
- `joblib.load`,
- model prediction langsung,
- feature engineering,
- Composition Strength calculation,
- recommendation logic,
- historical lookup logic.

---

# 9. Frontend Serving

Jika Flask tetap melayani frontend, root route dapat digunakan:

```http
GET /
```

untuk mengirim frontend application.

API tetap berada di bawah:

```text
/api/...
```

---

# 10. Development vs Production

Development dapat menggunakan Flask development server dan debug mode.

Production tidak menggunakan debug mode.

Production backend tidak menjalankan:

- training,
- preprocessing training,
- evaluation,
- test block,
- development-only code.

---

# 11. Postman Strategy

Postman digunakan sebagai integration test untuk HTTP layer.

Urutan testing:

```text
1. Start Flask
       ↓
2. Test General endpoint
       ↓
3. Test Team options
       ↓
4. Test Team prediction
       ↓
5. Test negative cases
       ↓
6. Connect frontend
```

Testing harus mencakup success dan error cases.

---

# 12. Definition of Done

## Architecture

- [x] General service selesai.
- [x] Team service selesai.
- [x] Backend tidak bergantung pada `training_v2`.
- [x] Resource statis di-load sekali.

## API

- [ ] `app.py` selesai.
- [ ] General endpoint selesai.
- [ ] Team options endpoint selesai.
- [ ] Team prediction endpoint selesai.
- [ ] HTTP error handling selesai.

## Validation

- [ ] Positive tests selesai.
- [ ] Negative tests selesai.
- [ ] Team 116-feature contract tervalidasi melalui API.

## Integration

- [ ] Postman testing selesai.
- [ ] Frontend terhubung kembali.

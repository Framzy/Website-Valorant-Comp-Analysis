# Valorant Predictor V2 — Team Prediction API

## 1. Tujuan

API Team Prediction V2 digunakan untuk mengevaluasi composition dalam konteks:

```text
Team + Map + Year + Agents
```

Fungsi utama:
- Menyediakan options Year, Map, dan Team.
- Mencari exact historical composition.
- Menangani unseen composition.
- Membentuk 116 feature.
- Menggunakan trained Team Prediction V2 model.
- Mengembalikan prediction dan historical metadata.

Backend tidak melakukan training.

---

# 2. Team Options API

Options digunakan frontend dengan pola:

```text
Year
 ↓
Map
 ↓
Team
```

## 2.1 Available Years

```http
GET /api/team/options/years
```

Response:

```json
{
  "years": [2024, 2025, 2026]
}
```

---

## 2.2 Available Maps

```http
GET /api/team/options/maps?year=2025
```

Query:

| Parameter | Type | Required |
|---|---|---:|
| `year` | integer | Yes |

Response:

```json
{
  "year": 2025,
  "maps": [
    "Abyss",
    "Ascent"
  ]
}
```

---

## 2.3 Available Teams

```http
GET /api/team/options/teams?year=2025&map=Abyss
```

Query:

| Parameter | Type | Required |
|---|---|---:|
| `year` | integer | Yes |
| `map` | string | Yes |

Response:

```json
{
  "year": 2025,
  "map": "Abyss",
  "teams": [
    "Fnatic",
    "Gen.G"
  ]
}
```

Options berasal dari Team V2 dataset yang sudah di-load oleh service.

---

# 3. Team Prediction API

## Endpoint

```http
POST /api/team/predict
```

Content-Type:

```http
application/json
```

---

## 4. Request

```json
{
  "team": "Fnatic",
  "map": "Abyss",
  "year": 2025,
  "agents": [
    "jett",
    "sova",
    "omen",
    "cypher",
    "kayo"
  ]
}
```

### Fields

| Field | Type | Required |
|---|---|---:|
| `team` | string | Yes |
| `map` | string | Yes |
| `year` | integer | Yes |
| `agents` | array[string] | Yes |

---

# 5. Validation

Service memastikan:

- Team wajib ada.
- Map wajib ada.
- Year wajib berupa integer.
- Agents wajib ada.
- Composition valid.
- Tidak ada unknown agent.
- Team + Map + Year tersedia dalam dataset.

---

# 6. Prediction Flow

## Historical Composition

Jika exact composition tersedia:

```text
Team + Map + Year + Agents
          ↓
Exact Historical Row
          ↓
Feature Encoding
          ↓
116 Features
          ↓
XGBoost
          ↓
Prediction
```

Historical Winrate hanya menjadi metadata dan tidak digunakan sebagai input untuk Composition Strength.

## Unseen Composition

Jika exact composition tidak tersedia:

```text
Team + Map + Year + Agents
          ↓
Inference Feature Builder
          ↓
Feature Encoding
          ↓
116 Features
          ↓
XGBoost
          ↓
Prediction
```

---

# 7. Feature Contract

Team Prediction V2 menggunakan tepat:

```text
116 features
```

| Group | Count |
|---|---:|
| Team | 57 |
| Map | 12 |
| Year | 3 |
| Agent | 29 |
| Role Count | 4 |
| Role Pattern | 8 |
| Numeric | 3 |
| **Total** | **116** |

Backend harus memastikan:

```text
model feature count = 116
feature_names count = 116
encoded feature count = 116
```

---

# 8. Response

```json
{
  "input": {
    "team": "Fnatic",
    "map": "Abyss",
    "year": 2025,
    "agents": [
      "jett",
      "sova",
      "omen",
      "cypher",
      "kayo"
    ]
  },
  "prediction": {
    "winrate": 0.61,
    "percentage": 61.0
  },
  "composition": {
    "role_pattern": "1D-2I-1C-1S",
    "duelist_count": 1,
    "initiator_count": 2,
    "controller_count": 1,
    "sentinel_count": 1
  },
  "historical": {
    "found": true,
    "winrate": 0.6,
    "composition_strength": 72.5
  },
  "inference": {
    "feature_count": 116
  }
}
```

Nilai prediction di atas hanya contoh struktur response.

---

# 9. Error

### 400 Bad Request

Untuk:
- missing field,
- invalid year,
- invalid map,
- invalid team/context,
- unknown agent,
- duplicate/invalid composition.

Response:

```json
{
  "error": "Descriptive error message"
}
```

### 500 Internal Server Error

Untuk unexpected server exception.

```json
{
  "error": "Internal server error."
}
```

---

# 10. Postman Testing

### Options

- [ ] GET years.
- [ ] GET maps berdasarkan year.
- [ ] GET teams berdasarkan year + map.

### Prediction

- [ ] Historical composition.
- [ ] Unseen composition.
- [ ] Prediction response.
- [ ] Feature count = 116.
- [ ] Historical metadata.

### Negative

- [ ] Missing team.
- [ ] Missing map.
- [ ] Missing year.
- [ ] Missing agents.
- [ ] Invalid year.
- [ ] Unknown agent.
- [ ] Duplicate agent.
- [ ] Kurang dari 5 agent.
- [ ] Lebih dari 5 agent.
- [ ] Team + Map + Year tidak tersedia.
- [ ] Malformed JSON.

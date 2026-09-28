# Team Prediction V2 — API Documentation

## 1. Overview

API ini menjadi kontrak antara frontend dan backend untuk fitur **Team Prediction V2**.

Alur utama:

```text
Frontend
   │
   ├── GET /api/team/options/years
   ├── GET /api/team/options/maps?year=2025
   ├── GET /api/team/options/teams?year=2025&map=Abyss
   └── POST /api/team/predict
                 │
                 ▼
        Team Prediction Service
                 │
                 ├── Validate input
                 ├── Build inference features
                 ├── Feature Engineering
                 └── Model inference
```

### Prinsip

- `valorant_dataset_all.csv` digunakan untuk build/training, bukan request runtime biasa.
- Team V2 hanya memuat team yang telah lolos eligibility filter.
- Eligibility global: **Tournament >= 40** dan **Active Years >= 2**.
- Runtime tetap memvalidasi konteks `Year + Map + Team`.
- Frontend menggunakan dropdown cascading: **Year → Map → Team**.
- Agent dipilih melalui card dan harus tepat 5 agent.
- Backend tetap menjadi source of truth untuk validasi.

---

## 2. Base URL

```text
/api/team
```

Contoh development:

```text
http://localhost:5000/api/team
```

Port mengikuti konfigurasi backend.

---

## 3. Endpoint Summary

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/api/team/options/years` | Daftar year tersedia |
| GET | `/api/team/options/maps` | Map berdasarkan year |
| GET | `/api/team/options/teams` | Team berdasarkan year + map |
| POST | `/api/team/predict` | Team Prediction |

---

## 4. GET `/api/team/options/years`

Mengambil daftar year yang tersedia pada Team V2.

### Request

```http
GET /api/team/options/years
```

Tidak membutuhkan parameter.

### Response

```json
{
  "years": [2023, 2024, 2025],
  "latest_year": 2025
}
```

| Field | Type | Keterangan |
|---|---|---|
| `years` | integer[] | Year tersedia |
| `latest_year` | integer | Year terbaru yang tersedia |

Frontend dapat menggunakan `latest_year` sebagai default.

---

## 5. GET `/api/team/options/maps`

Mengambil map yang tersedia untuk year tertentu.

### Request

```http
GET /api/team/options/maps?year=2025
```

### Query Parameters

| Parameter | Type | Required | Keterangan |
|---|---|---|---|
| `year` | integer | Yes | Year pilihan user |

### Response

```json
{
  "year": 2025,
  "maps": [
    "Abyss",
    "Ascent",
    "Bind",
    "Haven",
    "Icebox"
  ]
}
```

Hanya map yang mempunyai data pada year tersebut yang dikembalikan.

Jika year tidak valid:

```json
{
  "error": "Invalid year"
}
```

HTTP status: `400 Bad Request`.

---

## 6. GET `/api/team/options/teams`

Mengambil team yang:

1. telah lolos eligibility Team V2;
2. memiliki data pada year;
3. memiliki data pada map.

### Request

```http
GET /api/team/options/teams?year=2025&map=Abyss
```

### Query Parameters

| Parameter | Type | Required | Keterangan |
|---|---|---|---|
| `year` | integer | Yes | Year |
| `map` | string | Yes | Map |

### Response

```json
{
  "year": 2025,
  "map": "Abyss",
  "teams": [
    "100 Thieves",
    "FNATIC",
    "Gen.G",
    "Paper Rex"
  ]
}
```

Eligibility global:

```text
Tournament >= 40
AND
Active Years >= 2
```

Kemudian team harus tersedia pada konteks:

```text
Year + Map
```

Jika tidak ada:

```json
{
  "year": 2025,
  "map": "Abyss",
  "teams": []
}
```

---

## 7. POST `/api/team/predict`

Melakukan prediction berdasarkan Team, Map, Year, dan 5 Agent.

### Request

```http
POST /api/team/predict
Content-Type: application/json
```

### Body

```json
{
  "team": "100 Thieves",
  "map": "Abyss",
  "year": 2024,
  "agents": [
    "cypher",
    "gekko",
    "jett",
    "omen",
    "sova"
  ]
}
```

---

## 8. Request Validation

### Team

- Wajib diisi.
- Harus merupakan eligible team.
- Harus tersedia pada `Year + Map`.

### Map

- Wajib diisi.
- Harus tersedia pada Team V2.

### Year

- Wajib diisi.
- Harus tersedia pada Team V2.

### Agents

Harus tepat 5 agent:

```text
exactly 5 agents
```

Tidak boleh ada duplikasi.

Agent harus dikenal oleh pipeline.

**Composition baru tetap diperbolehkan.** Backend tidak mewajibkan composition pernah muncul secara historis.

---

## 9. Prediction Processing

```text
Request
   │
   ▼
Validate
   │
   ▼
Build application input
   │
   ▼
Historical/context feature lookup
   │
   ▼
Feature Engineering
   │
   ├── Team encoding
   ├── Map encoding
   ├── Year encoding
   ├── Agent encoding
   ├── Role counts
   ├── Role pattern
   └── Numeric features
   │
   ▼
116 features
   │
   ▼
Team Prediction Model
   │
   ▼
Prediction
```

### Current Feature Contract

| Feature Group | Count |
|---|---:|
| Team | 57 |
| Map | 12 |
| Year | 3 |
| Agent | 29 |
| Role Count | 4 |
| Role Pattern | 8 |
| Numeric | 3 |
| **Total** | **116** |

Urutan:

```text
Team
Map
Year
Agent
Role Count
Role Pattern
Numeric
```

---

## 10. Successful Response

Kontrak response awal:

```json
{
  "input": {
    "team": "100 Thieves",
    "map": "Abyss",
    "year": 2024,
    "agents": [
      "cypher",
      "gekko",
      "jett",
      "omen",
      "sova"
    ]
  },
  "prediction": {
    "winrate": 0.0365,
    "percentage": 3.65
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
    "winrate": 0.0,
    "composition_strength": 60.9
  }
}
```

> Struktur response ini adalah kontrak awal dan dapat disesuaikan setelah audit backend/service yang sudah ada.

---

## 11. HTTP Status

| Status | Kondisi |
|---|---|
| `200` | Request berhasil |
| `400` | Input tidak valid |
| `404` | Resource/context tidak ditemukan |
| `422` | Struktur request tidak dapat diproses |
| `500` | Internal server error |

Contoh:

```json
{
  "error": "Exactly 5 agents are required"
}
```

---

## 12. Frontend Flow

Frontend menggunakan cascading selection.

```text
Page Load
    │
    ▼
GET /options/years
    │
    ▼
Set latest_year as default
    │
    ▼
GET /options/maps?year={year}
    │
    ▼
User selects Map
    │
    ▼
GET /options/teams?year={year}&map={map}
    │
    ▼
User selects Team
    │
    ▼
User selects 5 Agent Cards
    │
    ▼
POST /predict
    │
    ▼
Display prediction
```

Jika Year berubah:

```text
Year changed
    │
    ├── reset Map
    ├── reset Team
    └── reload Maps
```

Jika Map berubah:

```text
Map changed
    │
    ├── reset Team
    └── reload Teams
```

---

## 13. All Years

UI dapat menyediakan:

```text
Year
├── 2025
├── 2024
├── 2023
└── All Years
```

Namun `All Years` **belum menjadi bagian dari prediction contract utama**.

Kita perlu menentukan cara agregasi lintas tahun untuk:

- Team Overall WR
- Team Map WR
- Composition Strength
- fitur historis lainnya

Mode utama V2:

```text
Specific Year → Map → Team → Agents → Prediction
```

---

## 14. Source of Truth

```text
RAW DATA
   │
   ▼
Build / Training
   │
   ▼
Eligibility Filter
   │
   ├── Rejected
   │
   └── Eligible Team
           │
           ▼
      Team V2 Dataset
           │
           ├──► Options API
           │
           └──► Prediction Service
                       │
                       ▼
                     Model
```

Frontend tidak menentukan eligibility.

Frontend hanya menggunakan data dari backend untuk UX. Backend wajib melakukan validasi kembali.

---

## 15. Non-Goals

API V2 tidak bertujuan untuk:

- mengakses raw dataset pada setiap request;
- melakukan training model saat prediction;
- menentukan eligibility Team di frontend;
- membatasi user hanya pada composition historis;
- melakukan prediction berdasarkan opponent;
- menggabungkan Team Dataset dan General Dataset.

---

## 16. Status Implementasi

| Komponen | Status |
|---|---|
| Team V2 Dataset | Ready |
| 116 Feature Contract | Ready |
| Model Artifact | Ready |
| Encoder Artifact | Ready |
| Feature Names Artifact | Ready |
| Inference Consistency | Validated |
| Eligibility Filter | **Belum diterapkan ke build pipeline** |
| Options API | Belum dibuat |
| Team Prediction Service | Belum dibuat |
| Flask Routes | Menunggu audit backend |
| Frontend Integration | Belum dibuat |
| All Years Prediction | Belum ditentukan |

Dokumen ini merupakan **API contract awal**. Implementasi endpoint dan response final dilakukan setelah audit backend yang sudah ada agar kita mengintegrasikannya tanpa merombak arsitektur secara tidak perlu.

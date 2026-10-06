# Valorant Predictor V2 — General Analysis API

## 1. Tujuan

API General Analysis V2 digunakan untuk menganalisis composition berdasarkan data historis.

Fungsi utama:
- Menentukan role pattern.
- Menentukan playstyle.
- Mencari exact historical composition.
- Menampilkan historical winrate dan pick rate jika tersedia.
- Memberikan recommendation berdasarkan popularity/pick rate.
- Menggunakan fallback jika composition tidak ditemukan.

API tidak melakukan ML prediction.

---

## 2. Endpoint

```http
POST /api/general/analyze
```

Content-Type:

```http
application/json
```

---

## 3. Shared Options

Year dan Map digunakan oleh General Analysis dan Team Prediction.

Endpoint bersama:

```http
GET /api/options/years
GET /api/options/maps?year=2025
```

Options tersebut berasal dari ketersediaan bersama dataset General V2 dan Team V2.

---

## 4. Request

```json
{
  "map": "Abyss",
  "year": 2024,
  "agents": [
    "cypher",
    "jett",
    "kayo",
    "omen",
    "sova"
  ]
}
```

### Fields

| Field | Type | Required | Keterangan |
|---|---|---:|---|
| `map` | string | Yes | Nama map |
| `year` | integer | Yes | Tahun |
| `agents` | array[string] | Yes | Tepat 5 agent |

### Validation

- Composition harus memiliki tepat 5 agent.
- Tidak boleh ada duplicate agent.
- Semua agent harus memiliki role yang dikenal.
- Agent dinormalisasi sebelum diproses.

---

## 5. Processing Flow

```text
Request
  ↓
Normalize Agents
  ↓
Determine Role Pattern
  ↓
Determine Playstyle
  ↓
Exact Historical Lookup
  ↓
Found ────────────────→ Historical Result
  │
  └─ Not Found ────────→ Fallback Recommendation
                              ↓
                         Recommendation
```

---

## 6. Playstyle

Playstyle ditentukan berdasarkan role pattern.

Playstyle yang digunakan:

- `STANDARD`
- `CONTROL`
- `AGGRESSIVE`
- `UTILITY_HEAVY`
- `UNCLASSIFIED`

`UNCLASSIFIED` tidak dipaksa menjadi playstyle tertentu.

---

## 7. Recommendation

Ranking recommendation:

1. `Pick Rate` descending.
2. `Total Maps Played` descending.

Winrate menjadi informasi pendukung, bukan primary ranking criterion.

---

## 8. Fallback

Jika exact composition tidak ditemukan:

```text
MAP + YEAR + PLAYSTYLE
        ↓
MAP + YEAR
        ↓
MAP
        ↓
GLOBAL
```

Jika playstyle `UNCLASSIFIED`, level `MAP + YEAR + PLAYSTYLE` dilewati.

---

## 9. Response

Contoh struktur:

```json
{
  "input": {
    "map": "Abyss",
    "year": 2024,
    "agents": [
      "cypher",
      "jett",
      "kayo",
      "omen",
      "sova"
    ]
  },
  "playstyle": {
    "name": "STANDARD",
    "pattern": "1D-2I-1C-1S"
  },
  "historical": {
    "found": true,
    "map": "Abyss",
    "year": 2024,
    "agents": [
      "cypher",
      "jett",
      "kayo",
      "omen",
      "sova"
    ],
    "role_pattern": "1D-2I-1C-1S",
    "playstyle": "STANDARD",
    "total_maps": 10,
    "winrate": 0.6,
    "pick_rate": 0.22727
  },
  "recommendations": [],
  "fallback": false,
  "fallback_source": null
}
```

Nilai statistik di atas hanya contoh struktur response.

---

## 10. Error

### 400 Bad Request

Untuk:
- field required tidak ada,
- composition tidak berisi tepat 5 agent,
- duplicate agent,
- unknown agent,
- input tidak valid.

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

## 11. API Test Checklist

### Positive
- [ ] Exact composition ditemukan.
- [ ] Exact composition tidak ditemukan.
- [ ] Fallback `MAP_YEAR_PLAYSTYLE`.
- [ ] Fallback `MAP_YEAR`.
- [ ] Fallback `MAP`.
- [ ] Fallback `GLOBAL`.
- [ ] `UNCLASSIFIED` playstyle.

### Negative
- [ ] Missing map.
- [ ] Missing year.
- [ ] Missing agents.
- [ ] Kurang dari 5 agent.
- [ ] Lebih dari 5 agent.
- [ ] Duplicate agent.
- [ ] Unknown agent.
- [ ] Malformed JSON.

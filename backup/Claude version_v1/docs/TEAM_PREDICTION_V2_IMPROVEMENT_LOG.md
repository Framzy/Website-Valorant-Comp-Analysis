# Team Prediction V2 — Log Perbaikan

> Dokumen ringkas, diperbarui setiap ada perubahan. Entri terbaru di bagian bawah tabel & bagian "Detail".

## Tentang Proyek

Memprediksi **win-rate sebuah tim pro Valorant di satu map** berdasarkan `Team + Map + Year + 5 Agent`
(termasuk composition yang belum pernah dimainkan — mode "what if"). Prediksi bersifat **berbasis tim**:
tiap tim punya cara menang sendiri. Data: hasil turnamen kompetitif (median hanya 2 map per composition,
sehingga data sangat tipis).

Alur: `valorant_dataset_all.csv` → `build_team_dataset.py` → `valorant_dataset_team_v2.csv` →
`train_team_v2.py` → `sync_to_backend.py` → `TeamPredictionService` (API) → frontend.

## Ringkasan Masalah & Perbaikan

| Tahap | Masalah | Perbaikan | File utama |
|---|---|---|---|
| 1 | **Target leakage**: Team WR dihitung dari Winrate baris itu sendiri, R² tampak bagus tapi palsu | WR berbobot map + shrinkage; saat training dihitung out-of-fold | `build_team_dataset.py`, `wr_features.py` |
| 2 | Model terlalu sensitif (satu ganti agent: 42% → 92%) | Model diregularisasi + blend `prior + λ·clip(model−prior, ±cap)` | `config.py` (`XGB_PARAMS`, `BLEND`) |
| 3 | Output satu angka tanpa tingkat keyakinan | Field `confidence` (level, jumlah map) dan `prediction.range` | `team_prediction_service.py` |
| 4 | Composition Strength = popularitas, bukan kualitas | Diganti win-rate hierarkis (exact → team+map+pattern → team+pattern → pattern global → prior tim) | `comp_strength_hierarchy.py` |
| 5 | Reconstruction menebak 1 composition untuk turnamen >5 agent; 98% campur ≥2 composition asli, 31% map hasil tebakan | Pakai composition per Stage+Match Type; tebakan hanya untuk sisa yang benar-benar ambigu (map hasil tebakan 31,2% → 0,5%) | `build_team_dataset.py` |
| 6 | Role pattern tidak wajar/tak pernah dipakai **tidak menurunkan prediksi** (5 Sentinel 60,3% > composition historis 59,4%) | Penyesuaian eksplisit, terbatas, transparan berdasarkan seberapa umum pattern di scene pro dan di tim itu | `config.py` (`PATTERN_ADJUST`), `team_prediction_service.py` |

## Detail Tahap 6 — Penyesuaian Role Pattern

**Alasan.** Tim pro tidak memilih role pattern acak: 6 pattern menutup 97% dari 6.386 map; 13 dari 19
pattern punya share <1%. Model yang diregularisasi hampir tidak bereaksi pada role pattern, jadi
pola langka tidak berdampak pada angka. Kalibrasi dari data (sampel kecil, arah konsisten):
pattern baru bagi tim ≈ −2 sampai −5 poin, pattern <1% ≈ −6 poin, baru + langka ≈ −11 poin.

**Aturan** (parameter di `PATTERN_ADJUST`, disimpan ke `metadata.json`, tidak perlu ubah backend manual):

| Situasi | Penalti (poin win-rate) | Tier |
|---|---|---|
| Tim rutin memakai pattern itu | ≈ 0 | `team_regular_pattern` |
| Pattern umum (≥10% map pro), tapi baru bagi tim | −2,0 | `team_new_common_pattern` |
| Pattern jarang/langka, baru bagi tim | −2 s/d −7 (skala log terhadap share) | `team_new_rare_pattern` |
| Tidak pernah dimainkan tim manapun | −8,0 | `never_played_by_any_team` |

Penalti mengecil bila tim sudah sering memakai pattern itu (`K/(K+team_maps)`) dan **dibatasi maksimum 10 poin**
(tidak radikal). Angka sebelum penyesuaian tetap dikembalikan terpisah.

**Contoh (G2 / Ascent / 2026, proxy sandbox):** pattern rutin −0,1 · pattern langka −5,9 ·
5 Sentinel dan 5 Duelist −8,0. Ganti agent dalam role yang sama tidak mengubah penalti (pattern sama); kasus G2 Vyse→Deadlock bergeser 0,6 poin (sweep 300 sampel: rata-rata ≈1,2 poin, maks 11,4 poin, batas teoretis 15).

**Field API baru (hanya menambah, field lama tidak berubah):**
- `prediction.winrate_before_adjustment`
- `prediction.pattern_adjustment` → `points`, `tier`, `reason`
- `confidence.meta_prevalence` → `role_pattern_share_pct`, `label`, `rank`, `n_patterns`, `team_maps_with_pattern`
- Catatan: `prediction.winrate` / `percentage` sekarang sudah termasuk penyesuaian.

**Perbaikan tambahan.** `inference_feature_builder.py` sudah meminta key `pattern_adoption` yang belum dihasilkan
`strength_for_new_row` (menyebabkan `KeyError` saat prediksi). Sekarang dihasilkan di kedua salinan
`comp_strength_hierarchy.py` (training dan backend). Service juga punya fallback: bila `inference_feature_builder.py` versi lama tidak meneruskan `pattern_adoption`, nilainya dihitung langsung dari hierarki yang sama (sebelumnya `KeyError` di `_evidence`).

## Perbaikan Lanjutan (setelah Tahap 6)

| Masalah | Perbaikan | File |
|---|---|---|
| Backend meng-import `training.team.comp_strength_hierarchy` (karena sync menyalin builder apa adanya) → backend tidak jalan tanpa folder `training/` | `sync_to_backend.py` menulis ulang import ke `backend.ml.team...` dan memberi peringatan bila ada import `training` di `backend/` | `sync_to_backend.py`, `backend/ml/team/inference_feature_builder.py` |
| Tiap prediksi mengulang semua groupby (≈24 ms/panggilan, 2× per request) → `check_team_service` lambat | Agregat dihitung sekali per dataset lalu dipakai ulang (≈0,03 ms/panggilan); hasil identik dengan versi lama (300/300 kasus diuji) | `comp_strength_hierarchy.py` (training + backend) |

## Cara Verifikasi

```
python -m training.team.preprocessing.build_team_dataset
python -m training.team.train_team_v2
python -m training.team.sync_to_backend
python -m backend.check_team_service     # kontrak API, stabilitas, urutan penalti role pattern
```

## Batasan yang Perlu Diketahui

- Kemampuan prediksi model tetap kecil (R² ≈ 0; MAE hanya sedikit di bawah baseline). Sinyal utama = kekuatan tim.
  Data yang lebih banyak berdampak lebih besar daripada tuning tambahan.
- Penalti role pattern adalah **prior produk berbasis pola scene pro**, bukan hasil belajar model dari win-rate;
  sampel pattern langka kecil (puluhan map). Karena itu dibatasi dan dilaporkan terpisah.
- Rekonstruksi role-based tersisa hanya untuk 16 record yang datanya memang tidak bisa dipecah lebih halus.

## Template Entri Berikutnya

`| N | Masalah | Perbaikan | File |` lalu tambahkan bagian "Detail Tahap N" bila perlu.

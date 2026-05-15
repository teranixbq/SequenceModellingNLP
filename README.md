# Sequence Modelling in NLP - 1D CNN

Repository ini berisi implementasi:
- Word-level CNN
- Character-level CNN
- Hierarchical CNN
- Eksperimen berbagai pooling:
  - Max Pooling
  - Average Pooling
  - Adaptive Pooling
  - Without Pooling

Dataset yang digunakan:
- [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
---

# How to Run
### 1. Buka Google Colab

Buka: https://colab.research.google.com

---

## 2. Open Notebook Via Link GitHub

Klik:
```bash
File → Open Notebook → GitHub → Salin link repo ini
```
<img width="989" height="768" alt="image" src="https://github.com/user-attachments/assets/17a9e4bf-c4be-4e88-8f0c-19c1fa4827a4" />

## 3. Enable Step Pertama
```bash
!mkdir -p src
!wget -N -P src https://raw.githubusercontent.com/teranixbq/SequenceModellingNLP/refs/heads/main/src/char_level.py
!wget -N -P src https://raw.githubusercontent.com/teranixbq/SequenceModellingNLP/refs/heads/main/src/word_level.py
```

# Summary Best Model

## 1. Best Top Model Pooling - Word-level CNN

| Rank | Model Type | Pooling Type | Kernel Sizes | Accuracy | Precision | Recall | F1-score | Training Time |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | word_cnn | Average Pooling | [3, 4, 5, 6] | 0.8790 | 0.864685 | 0.900735 | 0.882342 | 67.186841s |
| 2 | word_cnn | Average Pooling | [3, 4] | 0.8782 | 0.870130 | 0.891205 | 0.880541 | 40.780969s |
| 3 | word_cnn | Adaptive Pooling | [3, 4, 5, 6] | 0.8661 | 0.860008 | 0.876911 | 0.868377 | 65.764987s |
| 4 | word_cnn | Max Pooling | [3, 4] | 0.8658 | 0.892834 | 0.833631 | 0.862218 | 39.266199s |
| 5 | word_cnn | Max Pooling | [3, 4, 5, 6] | 0.8621 | 0.837454 | 0.901132 | 0.868127 | 65.108639s |

**Best Word-level CNN:**  
Model terbaik pada Word-level CNN adalah **word_cnn dengan Average Pooling dan kernel size [3, 4, 5, 6]**. Model ini memperoleh accuracy **0.8790** dan F1-score **0.882342**.

---

## 2. Best Top Model Pooling - Character-level CNN

| Rank | Model Type | Pooling Type | Kernel Sizes | Accuracy | Precision | Recall | F1-score | Training Time |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | char_cnn | Max Pooling | [3, 5] | 0.7722 | 0.747889 | 0.826285 | 0.785135 | 17.106889s |
| 2 | hierarchical_char_cnn | Max Pooling | [3, 5, 7] | 0.7599 | 0.708347 | 0.889617 | 0.788700 | 27.866008s |
| 3 | char_cnn | Adaptive Pooling | [3, 5] | 0.7418 | 0.870734 | 0.572365 | 0.690704 | 16.200598s |
| 4 | char_cnn | Average Pooling | [3, 5] | 0.7185 | 0.704453 | 0.759976 | 0.731162 | 19.208938s |
| 5 | char_cnn | Without Pooling | [3, 5] | 0.7082 | 0.695300 | 0.748858 | 0.721086 | 17.538617s |

**Best Character-level CNN:**  
Model terbaik pada Character-level CNN berdasarkan accuracy adalah **char_cnn dengan Max Pooling dan kernel size [3, 5]**. Model ini memperoleh accuracy **0.7722** dan F1-score **0.785135**.

---

## 3. Overall Best Model

| Model | Pooling Type | Kernel Sizes | Accuracy | F1-score |
|---|---|---|---:|---:|
| word_cnn | Average Pooling | [3, 4, 5, 6] | 0.8790 | 0.882342 |

Model terbaik secara keseluruhan adalah **Word-level CNN dengan Average Pooling dan kernel size [3, 4, 5, 6]**, karena menghasilkan accuracy dan F1-score tertinggi dibanding model Character-level CNN.

---

## 4. Word CNN vs Character CNN on Slang, Typo, Singkatan, dan Rare Text

| No | Review | Word Prediction | Character Prediction | Analisis |
|---|---|---|---|---|
| 1 | omg this movie was sickkkk, def 10/10 | Positive | Positive | Kedua model berhasil memahami slang dan typo seperti `sickkkk`. |
| 2 | the ending made me cry fr fr, so emotional | Positive | Positive | Kedua model mampu memahami singkatan informal seperti `fr fr`. |
| 3 | worst cinemax experience ever, rly bad acting | Negative | Negative | Kedua model tetap mampu mengenali typo/singkatan seperti `rly`. |
| 4 | ngl the cgi was kinda sus and cheap | Negative | Negative | Kedua model memahami slang seperti `ngl` dan `sus`. |
| 5 | totally mindblown! dat plot twist was cray | Negative | Negative | Kedua model masih dapat memahami bahasa informal seperti `dat` dan `cray`. |
| 6 | idk why ppl hate this, it was p cool | Negative | Positive | Character CNN lebih baik pada contoh ini karena mampu membaca pola singkatan seperti `idk`, `ppl`, dan `p cool`. |
| 7 | just watched it and wowww, absolute masterpiece | Positive | Positive | Kedua model berhasil membaca pola positif, termasuk pengulangan karakter seperti `wowww`. |
| 8 | boring af, i almost fell asleep in d theater | Negative | Negative | Kedua model berhasil memahami slang seperti `af` dan singkatan `d`. |
| 9 | the soundtrack was a whole vibe, luvvv it | Positive | Positive | Kedua model berhasil memprediksi positif meskipun terdapat typo/pengulangan seperti `luvvv`. |
| 10 | istg this is the best film of 2026, no cap | Positive | Negative | Character CNN salah memprediksi pada contoh ini, sedangkan Word CNN berhasil memprediksi dengan benar. |

---

## 5. Final Summary

Model terbaik akhir adalah:

| Best Model | Pooling Type | Kernel Sizes | Accuracy | F1-score |
|---|---|---|---:|---:|
| word_cnn | Average Pooling | [3, 4, 5, 6] | 0.8790 | 0.882342 |

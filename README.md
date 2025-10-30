# EE623: Speech Processing Assignment 1

This repository contains the recorded voice samples and analysis code for the EE623 (Speech Processing) assignment. The project involves recording various vowels and consonants in Hindi and performing acoustic analysis to determine their pitch and formant structures.

The analysis is divided into two main objectives:
1.  **Objective 1:** Pitch and formant estimation using AMDF and spectrograms (narrow-band and wide-band).
2.  **Objective 2:** Pitch and formant estimation using cepstral analysis over 6 consecutive frames.

## Recorded Audio Samples

As required by the assignment, all samples were recorded at a **44.1 kHz sampling frequency** and **16-bit resolution**.

The language chosen for all recordings is **Hindi**. All samples are located in the `/recordings` directory.

### Vowel Samples

| Filename | Target Vowel (Hindi) | Phonetic Symbol |
| :--- | :--- | :--- |
| `hindi_vowel_a.wav` | अ (a) | /ə/ |
| `hindi_vowel_aa.wav` | आ (aa) | /a:/ |
| `hindi_vowel_aha.wav` | अः (aha) | /əh/ |
| `hindi_vowel_ai.wav` | ऐ (ai) | /ɛ/ |
| `hindi_vowel_am.wav` | अं (am) | /ə̃/ |
| `hindi_vowel_aou.wav` | औ (aou) | /ɔ/ |
| `hindi_vowel_eh.wav` | ए (eh) | /e/ |
| `hindi_vowel_i.wav` | इ (i) | /ɪ/ |
| `hindi_vowel_ii.wav` | ई (ii) | /i:/ |
| `hindi_vowel_oh.wav` | ओ (oh) | /o/ |
| `hindi_vowel_ru.wav` | ऋ (ru) | /ɻ/ |
| `hindi_vowel_u.wav` | उ (u) | /ʊ/ |
| `hindi_vowel_uu.wav` | ऊ (uu) | /u:/ |

### Consonant Samples

| Filename | Target Consonant (Hindi) |
| :--- | :--- |
| `hindi_cons_ba.wav` | ब (ba) |
| `hindi_cons_bha.wav` | भ (bha) |
| `hindi_cons_cha.wav` | च (cha) |
| `hindi_cons_chha.wav` | छ (chha) |
| `hindi_cons_da.wav` | द (da) |
| `hindi_cons_dda.wav` | ड (dda) |
| `hindi_cons_dha.wav` | ध (dha) |
| `hindi_cons_dhha.wav` | ढ (dhha) |
| `hindi_cons_dna.wav` | ण (dna) |
| `hindi_cons_ga.wav` | ग (ga) |
| `hindi_cons_gha.wav` | घ (gha) |
| `hindi_cons_gya.wav` | ज्ञ (gya) |
| `hindi_cons_ha.wav` | ह (ha) |
| `hindi_cons_ja.wav` | ज (ja) |
| `hindi_cons_jha.wav` | झ (jha) |
| `hindi_cons_ka.wav` | क (ka) |
| `hindi_cons_kha.wav` | ख (kha) |
| `hindi_cons_ksha.wav` | क्ष (ksha) |
| `hindi_cons_la.wav` | ल (la) |
| `hindi_cons_ma.wav` | म (ma) |
| `hindi_cons_na.wav` | न (na) |
| `hindi_cons_nda.wav` | ङ (nda) |
| `hindi_cons_nya.wav` | ञ (nya) |
| `hindi_cons_pa.wav` | प (pa) |
| `hindi_cons_pha.wav` | फ (pha) |
| `hindi_cons_ra.wav` | र (ra) |
| `hindi_cons_sa.wav` | स (sa) |
| `hindi_cons_sha.wav` | श (sha) |
| `hindi_cons_shha.wav` | ष (shha) |
| `hindi_cons_ta.wav` | त (ta) |
| `hindi_cons_tha.wav` | थ (tha) |
| `hindi_cons_thha.wav` | ठ (thha) |
| `hindi_cons_tra.wav` | त्र (tra) |
| `hindi_cons_tta.wav` | ट (tta) |
| `hindi_cons_va.wav` | व (va) |
| `hindi_cons_ya.wav` | य (ya) |

## Analysis Code





This repository contains the Python scripts used to generate the results for t

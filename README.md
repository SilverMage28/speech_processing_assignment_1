# EE623: Speech Processing Assignment 1

This repository contains the recorded voice samples and analysis code for the EE623 (Speech Processing) assignment. The project involves recording various vowels and consonants in **Hindi** and performing acoustic analysis to determine their pitch and formant structures.

All audio samples were recorded at a **44.1 kHz sampling frequency** and **16-bit resolution**, as specified by the assignment.

The analysis scripts for both objectives are included in their respective folders.

## Recorded Audio Samples

All recorded `.wav` files are located in the `recordings/` directory.

### Vowels (स्वर)

The following Hindi vowels were recorded:

| Hindi Vowel | Phonetic Symbol | Filename |
| :--- | :--- | :--- |
| अ | /a/ | `hindi_vowel_a.wav` |
| आ | /a:/ | `hindi_vowel_aa.wav` |
| इ | /ɪ/ | `hindi_vowel_i.wav` |
| ई | /i:/ | `hindi_vowel_ii.wav` |
| उ | /ʊ/ | `hindi_vowel_u.wav` |
| ऊ | /u:/ | `hindi_vowel_uu.wav` |
| ऋ | /rɪ/ | `hindi_vowel_ru.wav` |
| ए | /e/ | `hindi_vowel_eh.wav` |
| ऐ | /ai/ | `hindi_vowel_ai.wav` |
| ओ | /o/ | `hindi_vowel_oh.wav` |
| औ | /au/ | `hindi_vowel_aou.wav` |
| अं | /aM/ | `hindi_vowel_am.wav` |
| अः | /aH/ | `hindi_vowel_aha.wav` |

### Consonants (व्यंजन)

The following Hindi consonants (in their default /a/ vowel form) were recorded:

| Hindi Consonant | Phonetic Symbol | Filename |
| :--- | :--- | :--- |
| क | /ka/ | `hindi_cons_ka.wav` |
| ख | /kʰa/ | `hindi_cons_kha.wav` |
| ग | /ɡa/ | `hindi_cons_ga.wav` |
| घ | /ɡʱa/ | `hindi_cons_gha.wav` |
| ङ | /ŋa/ | `hindi_cons_dna.wav` |
| च | /tʃa/ | `hindi_cons_cha.wav` |
| छ | /tʃʰa/ | `hindi_cons_chha.wav` |
| ज | /dʒa/ | `hindi_cons_ja.wav` |
| झ | /dʒʱa/ | `hindi_cons_jha.wav` |
| ञ | /ɲa/ | `hindi_cons_nya.wav` |
| ट | /ʈa/ | `hindi_cons_tta.wav` |
| ठ | /ʈʰa/ | `hindi_cons_thha.wav` |
| ड | /ɖa/ | `hindi_cons_dda.wav` |
| ढ | /ɖʱa/ | `hindi_cons_dhha.wav` |
| ण | /ɳa/ | `hindi_cons_nda.wav` |
| त | /t̪a/ | `hindi_cons_ta.wav` |
| थ | /t̪ʰa/ | `hindi_cons_tha.wav` |
| द | /d̪a/ | `hindi_cons_da.wav` |
| ध | /d̪ʱa/ | `hindi_cons_dha.wav` |
| न | /na/ | `hindi_cons_na.wav` |
| प | /pa/ | `hindi_cons_pa.wav` |
| फ | /pʰa/ | `hindi_cons_pha.wav` |
| ब | /ba/ | `hindi_cons_ba.wav` |
| भ | /bʱa/ | `hindi_cons_bha.wav` |
| म | /ma/ | `hindi_cons_ma.wav` |
| य | /ja/ | `hindi_cons_ya.wav` |
| र | /ra/ | `hindi_cons_ra.wav` |
| ल | /la/ | `hindi_cons_la.wav` |
| व | /ʋa/ | `hindi_cons_va.wav` |
| श | /ʃa/ | `hindi_cons_sha.wav` |
| ष | /ʂa/ | `hindi_cons_shha.wav` |
| स | /sa/ | `hindi_cons_sa.wav` |
| ह | /ha/ | `hindi_cons_ha.wav` |
| क्ष | /kʂa/ | `hindi_cons_ksha.wav` |
| त्र | /t̪ra/ | `hindi_cons_tra.wav` |
| ज्ञ | /ɡja/ | `hindi_cons_gya.wav` |

### Script Descriptions

* **`Objective_1/amd_analysis.py`**: Implements the Average Magnitude Difference Function (AMDF) for Objective 1. It loads a specified vowel file and given a stable frame, calculates the pitch in Hz by finding the first significant minimum in the AMDF.
* **`Objective_2/formant_analysis_pyceps.py`**: Implements the cepstral analysis for Objective 2. It uses the included `pyceps` library to analyze 6 consecutive frames of a vowel, calculating the average F0, F1, F2, and F3. It also generates the stacked plots of the cepstral sequence and the smoothed spectral envelope.

### Dependencies

The analysis scripts require the following Python libraries:
* `numpy`
* `librosa`
* `matplotlib`
* `scipy`

You can install them using pip:
```bash
pip install numpy librosa matplotlib scipy

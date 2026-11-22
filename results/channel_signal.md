# Channel signal diagnostic

Run date: 2026-11-22T13:38:57.408392+05:30
Git commit: `822c708f67b8e11562f9649aed52dad9e3d26af2`

Scores use one-second bins. AUC/AP assess ranking; mean positive percentile is the average percentile of labeled seconds.

## human verified

| Match | Channel | ROC-AUC | Average precision | Mean positive percentile |
|---|---|---:|---:|---:|
| cricket_match_1 | visual | 0.4785 | 0.0202 | 0.4790 |
| cricket_match_1 | audio | 0.6576 | 0.0596 | 0.6459 |
| cricket_match_1 | speech | 0.4938 | 0.0252 | 0.4165 |
| cricket_match_1 | current_fused | 0.4902 | 0.0208 | 0.4895 |

## pseudo label

| Match | Channel | ROC-AUC | Average precision | Mean positive percentile |
|---|---|---:|---:|---:|
| cricket_match_1 | visual | 0.5657 | 0.0588 | 0.5627 |
| cricket_match_1 | audio | 0.7674 | 0.1577 | 0.7557 |
| cricket_match_1 | speech | 0.5802 | 0.0637 | 0.5829 |
| cricket_match_1 | current_fused | 0.6235 | 0.0738 | 0.6178 |
| cricket_match_2 | visual | 0.6453 | 0.0702 | 0.6386 |
| cricket_match_2 | audio | 0.7809 | 0.1749 | 0.7666 |
| cricket_match_2 | speech | 0.6043 | 0.0977 | 0.5821 |
| cricket_match_2 | current_fused | 0.7387 | 0.1323 | 0.7276 |
| football_match_1 | visual | 0.5820 | 0.0588 | 0.5784 |
| football_match_1 | audio | 0.6842 | 0.1213 | 0.6795 |
| football_match_1 | speech | 0.4772 | 0.0502 | 0.4869 |
| football_match_1 | current_fused | 0.6922 | 0.1151 | 0.6838 |

## Top ten six-second windows: cricket_match_1 human labels

| Channel | Start | End | Score | Overlaps human interval |
|---|---:|---:|---:|---|
| visual | 9004 | 9010 | 19.425517 | False |
| visual | 9005 | 9011 | 18.827150 | False |
| visual | 9006 | 9012 | 18.599433 | False |
| visual | 4606 | 4612 | 18.513183 | False |
| visual | 9003 | 9009 | 18.271633 | False |
| visual | 4607 | 4613 | 18.006217 | False |
| visual | 2465 | 2471 | 17.944433 | False |
| visual | 2169 | 2175 | 17.814750 | False |
| visual | 4608 | 4614 | 17.675000 | False |
| visual | 9007 | 9013 | 17.546433 | False |
| audio | 1925 | 1931 | 0.026717 | False |
| audio | 1923 | 1929 | 0.026317 | False |
| audio | 1924 | 1930 | 0.026317 | False |
| audio | 1922 | 1928 | 0.025950 | False |
| audio | 1921 | 1927 | 0.025017 | False |
| audio | 9621 | 9627 | 0.021800 | False |
| audio | 9620 | 9626 | 0.021000 | False |
| audio | 9622 | 9628 | 0.020767 | False |
| audio | 9619 | 9625 | 0.020117 | False |
| audio | 9618 | 9624 | 0.019033 | False |
| speech | 2055 | 2061 | 1.459300 | True |
| speech | 6853 | 6859 | 1.455800 | False |
| speech | 6852 | 6858 | 1.455800 | False |
| speech | 6851 | 6857 | 1.455800 | False |
| speech | 6850 | 6856 | 1.455800 | False |
| speech | 6849 | 6855 | 1.455800 | False |
| speech | 6848 | 6854 | 1.455800 | False |
| speech | 6847 | 6853 | 1.455800 | False |
| speech | 6846 | 6852 | 1.455800 | False |
| speech | 11704 | 11710 | 1.452100 | False |
| current_fused | 9006 | 9012 | 0.377050 | False |
| current_fused | 9005 | 9011 | 0.366150 | False |
| current_fused | 9007 | 9013 | 0.358300 | False |
| current_fused | 9004 | 9010 | 0.357667 | False |
| current_fused | 9008 | 9014 | 0.330633 | False |
| current_fused | 3000 | 3006 | 0.325867 | False |
| current_fused | 2999 | 3005 | 0.318333 | False |
| current_fused | 9003 | 9009 | 0.305667 | False |
| current_fused | 3001 | 3007 | 0.302833 | False |
| current_fused | 2998 | 3004 | 0.294750 | False |

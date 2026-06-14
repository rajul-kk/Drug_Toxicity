# Model Metrics

Ensemble of 2 models, scaffold split (80/10/10), CPU training.
Metrics on held-out test set. Calibrated = temperature-scaled (T fitted on val set).

## GNN (ensemble=2, epochs=60)

Checkpoint: `checkpoints/gnn_tox-cli-sid-ame-cyp-her-dil_20260612`  
Temperature: T=0.6020

### Per-task (calibrated)

| Task | Dataset | AUC | AUPRC | Brier | ECE |
|------|---------|-----|-------|-------|-----|
| NR-AR | tox21 | 0.7675 | 0.4518 | 0.1039 | 0.2402 |
| NR-AR-LBD | tox21 | 0.7917 | 0.3599 | 0.0751 | 0.1679 |
| NR-AhR | tox21 | 0.8131 | 0.4673 | 0.1762 | 0.2370 |
| NR-Aromatase | tox21 | 0.7043 | 0.2121 | 0.1584 | 0.2148 |
| NR-ER | tox21 | 0.6354 | 0.3031 | 0.2016 | 0.2736 |
| NR-ER-LBD | tox21 | 0.6825 | 0.2701 | 0.0857 | 0.1625 |
| NR-PPAR-gamma | tox21 | 0.7743 | 0.1814 | 0.0788 | 0.1231 |
| SR-ARE | tox21 | 0.7644 | 0.4789 | 0.2005 | 0.2212 |
| SR-ATAD5 | tox21 | 0.7072 | 0.1222 | 0.0901 | 0.1321 |
| SR-HSE | tox21 | 0.7095 | 0.1820 | 0.1393 | 0.2098 |
| SR-MMP | tox21 | 0.8190 | 0.5463 | 0.1495 | 0.1245 |
| SR-p53 | tox21 | 0.6729 | 0.2013 | 0.1624 | 0.1781 |
| FDA_APPROVED | clintox | 0.6772 | 0.9610 | 0.2108 | 0.3495 |
| CT_TOX | clintox | 0.7594 | 0.3730 | 0.1072 | 0.1036 |
| Hepatobiliary disorders | sider | 0.6736 | 0.7035 | 0.2521 | 0.1640 |
| Metabolism and nutrition disorders | sider | 0.6571 | 0.8086 | 0.2066 | 0.0633 |
| Product issues | sider | 0.5280 | 0.0214 | 0.0205 | 0.0287 |
| Eye disorders | sider | 0.6091 | 0.6136 | 0.2559 | 0.1359 |
| Investigations | sider | 0.6715 | 0.9071 | 0.1409 | 0.0955 |
| Musculoskeletal and connective tissue disorders | sider | 0.6792 | 0.8690 | 0.1876 | 0.0876 |
| Gastrointestinal disorders | sider | 0.6646 | 0.9403 | 0.1100 | 0.0731 |
| Social circumstances | sider | 0.6360 | 0.2253 | 0.2087 | 0.2376 |
| Immune system disorders | sider | 0.5352 | 0.6686 | 0.2344 | 0.1198 |
| Reproductive system and breast disorders | sider | 0.6967 | 0.6200 | 0.2438 | 0.1638 |
| Neoplasms benign, malignant and unspecified (incl cysts and polyps) | sider | 0.6226 | 0.3531 | 0.2729 | 0.2701 |
| General disorders and administration site conditions | sider | 0.5148 | 0.9328 | 0.1383 | 0.2342 |
| Endocrine disorders | sider | 0.6349 | 0.2280 | 0.2299 | 0.2734 |
| Surgical and medical procedures | sider | 0.5331 | 0.1251 | 0.2274 | 0.2867 |
| Vascular disorders | sider | 0.5300 | 0.7521 | 0.2183 | 0.1495 |
| Blood and lymphatic system disorders | sider | 0.6827 | 0.7744 | 0.2211 | 0.1117 |
| Skin and subcutaneous tissue disorders | sider | 0.4824 | 0.9022 | 0.1166 | 0.1735 |
| Congenital, familial and genetic disorders | sider | 0.5160 | 0.1446 | 0.2084 | 0.2325 |
| Infections and infestations | sider | 0.6824 | 0.8461 | 0.2003 | 0.0662 |
| Respiratory, thoracic and mediastinal disorders | sider | 0.6338 | 0.8063 | 0.1893 | 0.1107 |
| Psychiatric disorders | sider | 0.5951 | 0.7402 | 0.2231 | 0.1010 |
| Renal and urinary disorders | sider | 0.6688 | 0.7909 | 0.2062 | 0.0821 |
| Pregnancy, puerperium and perinatal conditions | sider | 0.6706 | 0.1012 | 0.1175 | 0.1461 |
| Ear and labyrinth disorders | sider | 0.6073 | 0.4850 | 0.2864 | 0.2404 |
| Cardiac disorders | sider | 0.5955 | 0.7179 | 0.2287 | 0.1447 |
| Nervous system disorders | sider | 0.5670 | 0.9415 | 0.0997 | 0.1493 |
| Injury, poisoning and procedural complications | sider | 0.5548 | 0.6601 | 0.2604 | 0.1381 |
| mutagenic | ames | 0.7321 | 0.7039 | 0.2230 | 0.1203 |
| cyp1a2_inhibitor | cyp450 | 0.8526 | 0.8434 | 0.1569 | 0.0234 |
| cyp2c9_inhibitor | cyp450 | 0.8086 | 0.6985 | 0.1771 | 0.0644 |
| cyp2d6_inhibitor | cyp450 | 0.8012 | 0.6108 | 0.1547 | 0.1160 |
| cyp3a4_inhibitor | cyp450 | 0.8597 | 0.7951 | 0.1496 | 0.0349 |
| cyp2c19_inhibitor | cyp450 | 0.8167 | 0.8056 | 0.1749 | 0.0436 |
| herg_blocker | herg | 0.7809 | 0.6319 | 0.1813 | 0.1016 |
| dili_concern | dili | 0.8295 | 0.8612 | 0.1694 | 0.1365 |
| **Mean** | | **0.6776** | **0.5661** | **0.1762** | **0.1522** |

---

## DMPNN (ensemble=2, epochs=60)

Checkpoint: `checkpoints/dmpnn_tox-cli-sid-ame-cyp-her-dil_20260612`  
Temperature: T=0.6363

### Per-task (calibrated)

| Task | Dataset | AUC | AUPRC | Brier | ECE |
|------|---------|-----|-------|-------|-----|
| NR-AR | tox21 | 0.7752 | 0.4081 | 0.1363 | 0.2891 |
| NR-AR-LBD | tox21 | 0.7817 | 0.2649 | 0.0704 | 0.1482 |
| NR-AhR | tox21 | 0.8007 | 0.4840 | 0.1871 | 0.2520 |
| NR-Aromatase | tox21 | 0.6609 | 0.1879 | 0.1756 | 0.2353 |
| NR-ER | tox21 | 0.6218 | 0.2882 | 0.2477 | 0.3408 |
| NR-ER-LBD | tox21 | 0.6859 | 0.2324 | 0.0790 | 0.1471 |
| NR-PPAR-gamma | tox21 | 0.7549 | 0.1550 | 0.1006 | 0.1672 |
| SR-ARE | tox21 | 0.6817 | 0.4112 | 0.2445 | 0.2648 |
| SR-ATAD5 | tox21 | 0.7094 | 0.1425 | 0.0974 | 0.1543 |
| SR-HSE | tox21 | 0.7424 | 0.2323 | 0.1462 | 0.2349 |
| SR-MMP | tox21 | 0.7901 | 0.4987 | 0.1796 | 0.1789 |
| SR-p53 | tox21 | 0.6900 | 0.1985 | 0.1635 | 0.1884 |
| FDA_APPROVED | clintox | 0.7419 | 0.9695 | 0.2021 | 0.3265 |
| CT_TOX | clintox | 0.7767 | 0.2794 | 0.1229 | 0.1215 |
| Hepatobiliary disorders | sider | 0.6835 | 0.6692 | 0.2351 | 0.1189 |
| Metabolism and nutrition disorders | sider | 0.6856 | 0.8319 | 0.1937 | 0.0758 |
| Product issues | sider | 0.6608 | 0.0298 | 0.0242 | 0.0364 |
| Eye disorders | sider | 0.6196 | 0.6393 | 0.2532 | 0.1175 |
| Investigations | sider | 0.7313 | 0.9315 | 0.1339 | 0.1085 |
| Musculoskeletal and connective tissue disorders | sider | 0.6802 | 0.8670 | 0.1841 | 0.0495 |
| Gastrointestinal disorders | sider | 0.6692 | 0.9509 | 0.1151 | 0.0922 |
| Social circumstances | sider | 0.6508 | 0.2847 | 0.2232 | 0.2733 |
| Immune system disorders | sider | 0.6040 | 0.7526 | 0.2248 | 0.0973 |
| Reproductive system and breast disorders | sider | 0.6925 | 0.6378 | 0.2425 | 0.1727 |
| Neoplasms benign, malignant and unspecified (incl cysts and polyps) | sider | 0.6572 | 0.3747 | 0.2320 | 0.1865 |
| General disorders and administration site conditions | sider | 0.6259 | 0.9618 | 0.1242 | 0.2185 |
| Endocrine disorders | sider | 0.6770 | 0.2571 | 0.2089 | 0.2494 |
| Surgical and medical procedures | sider | 0.5708 | 0.1360 | 0.2026 | 0.2330 |
| Vascular disorders | sider | 0.6438 | 0.8323 | 0.1923 | 0.0880 |
| Blood and lymphatic system disorders | sider | 0.6744 | 0.7594 | 0.2235 | 0.0967 |
| Skin and subcutaneous tissue disorders | sider | 0.4817 | 0.9161 | 0.1252 | 0.1894 |
| Congenital, familial and genetic disorders | sider | 0.5348 | 0.2061 | 0.2056 | 0.2340 |
| Infections and infestations | sider | 0.7106 | 0.8607 | 0.1929 | 0.0796 |
| Respiratory, thoracic and mediastinal disorders | sider | 0.6783 | 0.8704 | 0.1816 | 0.0904 |
| Psychiatric disorders | sider | 0.5830 | 0.7521 | 0.2271 | 0.1100 |
| Renal and urinary disorders | sider | 0.6476 | 0.7933 | 0.2122 | 0.1114 |
| Pregnancy, puerperium and perinatal conditions | sider | 0.6533 | 0.0979 | 0.1321 | 0.1785 |
| Ear and labyrinth disorders | sider | 0.6224 | 0.5264 | 0.2674 | 0.1903 |
| Cardiac disorders | sider | 0.5889 | 0.7279 | 0.2289 | 0.1318 |
| Nervous system disorders | sider | 0.6127 | 0.9594 | 0.0953 | 0.1531 |
| Injury, poisoning and procedural complications | sider | 0.5949 | 0.7194 | 0.2444 | 0.1194 |
| mutagenic | ames | 0.7049 | 0.6675 | 0.2345 | 0.1268 |
| cyp1a2_inhibitor | cyp450 | 0.8500 | 0.8448 | 0.1584 | 0.0255 |
| cyp2c9_inhibitor | cyp450 | 0.8034 | 0.6825 | 0.1859 | 0.0977 |
| cyp2d6_inhibitor | cyp450 | 0.8124 | 0.6289 | 0.1568 | 0.1368 |
| cyp3a4_inhibitor | cyp450 | 0.8613 | 0.7999 | 0.1559 | 0.0800 |
| cyp2c19_inhibitor | cyp450 | 0.8108 | 0.8041 | 0.1773 | 0.0282 |
| herg_blocker | herg | 0.7882 | 0.6553 | 0.1923 | 0.1403 |
| dili_concern | dili | 0.8509 | 0.8719 | 0.1554 | 0.1003 |
| **Mean** | | **0.6925** | **0.5725** | **0.1775** | **0.1548** |

---

## Summary

| Model | Mean AUC | Mean AUPRC | Mean Brier | Mean ECE |
|-------|----------|------------|------------|----------|
| GNN (calibrated) | 0.6776 | 0.5661 | 0.1762 | 0.1522 |
| DMPNN (calibrated) | 0.6925 | 0.5725 | 0.1775 | 0.1548 |

DMPNN edges out GNN on AUC (+0.015) and AUPRC (+0.006). GNN has slightly better Brier and ECE. Both benefit from temperature scaling — CYP450 tasks are the strongest performers (AUC 0.80–0.86) while some SIDER endpoints remain difficult (AUC ~0.48–0.53) due to sparse labels and class imbalance.

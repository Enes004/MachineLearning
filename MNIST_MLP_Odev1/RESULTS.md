# MNIST MLP Deney Sonuclari

Bu rapor `train.py` tarafindan egitim sonunda olusturulmustur.
El yazisi degerlendirmesini ve genel yorumunuzu `REPORT.md` dosyasina ekleyin.

## Veri bolumu

| Train | Validation | Test |
|---:|---:|---:|
| 55000 | 5000 | 10000 |

Test kumesi yalnizca validation ile model secimi tamamlandiktan sonra degerlendirilmistir.

## Hiperparametreler ve son olcumler

| Parametre/olcum | Deger |
|---|---:|
| Gizli katman boyutlari | 512, 256, 128 |
| Aktivasyon | ReLU |
| Kayip | CrossEntropyLoss |
| Optimizer | adam |
| Ogrenme orani | 0.001 |
| Batch size | 128 |
| Epoch sayisi | 8 |
| Seed | 42 |
| Cihaz | cuda |
| En iyi validation epoch'u | 7 |
| Training loss (secilen epoch) | 0.0261 |
| Validation loss (secilen epoch) | 0.0802 |
| Training accuracy (secilen epoch) | 99.14% |
| Validation accuracy (secilen epoch) | 97.86% |
| Test accuracy | 97.98% |
| Secilen modelin egitim suresi | 43.5 saniye |

## Yapilandirma karsilastirmasi

| Gizli katman boyutlari | En iyi epoch | Validation accuracy | Egitim suresi (saniye) |
|---|---:|---:|---:|
| 128, 64 | 7 | 97.08% | 41.0 |
| 256, 128 | 7 | 97.60% | 36.7 |
| 512, 256, 128 | 7 | 97.86% | 43.5 |

## Grafikler

![Train ve validation loss/accuracy](artifacts/training_curves.png)

![Yapilandirma karsilastirmasi](artifacts/configuration_comparison.png)

El yazisi tahminlerini ve kendi degerlendirmenizi `REPORT.md` dosyasinda tamamlayin.

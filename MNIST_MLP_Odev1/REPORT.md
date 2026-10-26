# MNIST MLP Deney Raporu

> Egitimden sonra sayisal sonuclar `RESULTS.md` dosyasina otomatik yazilir. Bu dosyayi kendi el yazisi deneyiniz ve yorumunuzla birlikte teslim edin; olcum uydurmayin.

## 1. Amac ve yontem

MNIST'teki 0-9 rakamlarini CNN kullanmadan, sifirdan egitilen bir MLP ile siniflandirdim. Girdi her biri 28x28 piksel olan gri tonlu goruntulerdir. Modelde tam baglantili katmanlar ve ReLU kullanilmistir.

## 2. Veri ve bolme

- Veri kumesi: MNIST
- Train: 55.000 ornek
- Validation: 5.000 ornek
- Test: 10.000 ornek
- Train/validation ayrimi seed 42 ile tekrar uretilebilir.
- Test kumesi model egitimi veya hiperparametre secimi icin kullanilmamistir; yalnizca son secilen model icin degerlendirilmistir.

## 3. Hiperparametreler ve sonuclar

`RESULTS.md` dosyasi hiperparametreleri, training/validation loss ve accuracy degerlerini, test accuracy ve egitim suresini egitimden sonra otomatik listeler. Egitim grafigi `artifacts/training_curves.png` dosyasindadir.

## 4. Yapilandirma karsilastirmasi (istege bagli)

`python train.py --compare` calistirildiysa `artifacts/metrics.json` icindeki `experiments` sonuclarini ve `artifacts/configuration_comparison.png` grafigini buraya ekleyin. Yapilandirma secimini yalnizca validation sonuclarina gore aciklayin; test sonucuna bakarak yeni model secmeyin.

## 5. Kendi el yazimla yazdigim okul numarasi

Kendi el yazimla olusturdugum rakam gorselleri: `data/handwriting/`

Tahminler ve etiketle karsilastirma: `artifacts/handwriting_predictions.csv`

- Dogru tahmin edilen rakam sayisi / toplam rakam sayisi:
- MNIST test basarisi ile el yazisi basarisi arasindaki fark:
- Muhtemel nedenler (yazim bicimi, kalinlik, hizalama, olcekleme, veri dagilimi):

## 6. Genel degerlendirme

Egitim/validation egilimlerini, modelin guclu ve zayif yonlerini ve gercek el yazisi goruntulerinin MNIST'ten nasil farkli olabilecegini kendi sonuclarima dayanarak aciklayin.

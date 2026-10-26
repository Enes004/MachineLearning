# MNIST ile MLP El Yazisi Rakam Tanima

Bu proje MNIST rakamlarini **CNN kullanmadan**, PyTorch ile sifirdan egitilen cok katmanli bir perceptron (MLP) ile siniflandirir. Herhangi bir hazir model veya onceden egitilmis agirlik kullanilmaz. Veri, ilk calistirmada torchvision tarafindan indirilir.

## Kurulum

Python 3.10 veya ustu onerilir. PyTorch kurulumunu bilgisayarinizin CPU/GPU durumuna uygun sekilde [resmi kurulum secicisinden](https://pytorch.org/get-started/locally/) yapin. Ardindan proje klasorunde:

```bash
python -m pip install -r requirements.txt
```

## Egitim ve degerlendirme

Temel deney:

```bash
python train.py
```

Varsayilan yapi `784 -> 256 -> 128 -> 10`, aktivasyon ReLU, kayip CrossEntropyLoss, optimizer Adam'dir. Girdi gri tonlu 28x28 goruntudur; `ToTensor` pikselleri `[0, 1]` araligina donusturur. Varsayilan hiperparametreler: 8 epoch, batch size 128, learning rate 0.001, seed 42.

MNIST'in 60.000 egitim ornegi sabit seed ile 55.000 train ve 5.000 validation olarak ayrilir. 10.000 orneklik test kumesi egitim ve model seciminde kullanilmaz; yalnizca validation basarisi en iyi modeli sectikten sonra bir kez degerlendirilir.

Farkli MLP boyutlarini validation kumesinde karsilastirmak icin:

```bash
python train.py --compare
```

Bu secenek `[128, 64]`, `[256, 128]` ve `[512, 256, 128]` gizli katman yapilarini karsilastirir ve test kumesine yalnizca validation sonucu en iyi olan modeli uygular. Kendi yapinizi egitmek icin, ornegin:

```bash
python train.py --hidden-dims 256 128 --optimizer adam --learning-rate 0.001 --epochs 8 --batch-size 128
```

Egitim tamamlaninca kok klasorde `RESULTS.md` otomatik olusur; sayisal olcumleri, hiperparametreleri ve grafikleri iceren bu raporu ve `artifacts/` altindaki `metrics.json` ile grafikleri deponuza ekleyebilirsiniz. Model agirliklari (`best_mnist_mlp.pt`) yeniden olusturulabildigi icin `.gitignore` tarafindan dislanir. `--compare` seceneginde ayrica `configuration_comparison.png` uretilir. Sure bilgisi egitim dongulerini kapsar; veri indirme ve test degerlendirmesi bu olcume dahil degildir.

## Kendi el yazinizla okul numarasi

Grafik arayuzlu bir masaustu oturumunda:

```bash
python collect_handwriting.py
```

Okul numaranizi uygulama icinde girin ve her rakami ayri tuvalde **kendi el yazinizla** cizip kaydedin. Uygulama her goruntuyu gri tonlu 28x28 MNIST benzeri (siyah zemin, beyaz rakam) PNG olarak `data/handwriting/` altina kaydeder. Bos tuval kaydedilmez. Okul numaranizi terminale, kaynak koda veya rapora yazmaniz gerekmez.

Model egitildikten ve en az bir rakam kaydedildikten sonra:

```bash
python predict_handwriting.py
```

Tahminler, guven skorlar ve dosya adinda bulunan dogru etiketle karsilastirma `artifacts/handwriting_predictions.csv` dosyasina yazilir. Bu sonuc MNIST test basarisindan ayri raporlanmalidir; ogrenci el yazisi MNIST'ten farkli dagilim gosterebilir.

**Gizlilik:** Okul numarasi goruntuleri kisisel bilgi icerebilir. Depoyu herkese acik paylasmadan once gorselleri eklemek icin izin/kurum politikasini kontrol edin; gerekiyorsa depoyu ozel yapin. `data/handwriting/` otomatik olarak `.gitignore` listesinde degildir, bu nedenle teslimden once ekleyeceginiz dosyalari gozden gecirin.

## Teslim raporu

`RESULTS.md` egitimden sonra gercek olcumlerle otomatik olusur; egitimden once bu depo sonuc uydurulmus degerler icermez. `REPORT.md` sablonuna el yazisi tahminlerinizi ve kendi yorumunuzu ekleyin. Teslimden once okul numaranizla olusturdugunuz kendi goruntulerini, `RESULTS.md` dosyasini ve `artifacts/` altindaki olcum/grafikleri depoya ekleyin.

## Testler

```bash
python -m unittest discover -s tests -v
```

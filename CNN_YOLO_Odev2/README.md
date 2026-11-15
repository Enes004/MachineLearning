# Pascal VOC Nesne Tespiti - YOLO26n

Pascal VOC 2007 + 2012 veri seti üzerinde Ultralytics YOLO26n modelinin fine-tune edilmesi,
test değerlendirmesi, hız ölçümü ve Gradio uygulaması.

- Görev: nesne tespiti (object detection), 20 VOC sınıfı
- Model: `yolo26n.pt` (Ultralytics 8.4.214 içindeki en güncel nano model), COCO-pretrained
- Girdi boyutu: 640×640
- Split: VOC07+12 trainval → %90 train / %10 val (seed=42), VOC2007 test → test
- Donanım: NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)

Ayrıntılı rapor: [`report/REPORT.md`](report/REPORT.md)

## Sonuçlar (VOC2007 test)

- Test mAP@0.5: **0.801**, mAP@0.5:0.95: **0.593**
- Precision / Recall (conf=0.25, IoU=0.5): **0.740 / 0.779**, F1: **0.759**
- Eğitim: **01:46:21** wall-clock, 48 epoch (RTX 4050 Laptop, AdamW)
- Inference (640×640, batch 1, 200 test görüntüsü): GPU **8.63 ms** (115.8 FPS),
  CPU **36.0 ms** (27.8 FPS)
- Ağırlık: `weights/best.pt` (5.3 MB)

## Kurulum

Sistem Python'una paket kurulmaz; her şey `.venv` içindedir.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Bu makinede ROS 2 nedeniyle `PYTHONPATH` dolu olabilir. Komutları şu şekilde çalıştırmak
en güvenlisidir:

```bash
env -u PYTHONPATH .venv/bin/python <script.py>
```

GPU kontrolü:

```bash
nvidia-smi
env -u PYTHONPATH .venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

## Veri Hazırlama

VOC arşivlerini indirin (Ultralytics aynaları):

```bash
mkdir -p data/raw
cd data/raw
base=https://github.com/ultralytics/assets/releases/download/v0.0.0
curl -LO $base/VOCtrainval_06-Nov-2007.zip
curl -LO $base/VOCtest_06-Nov-2007.zip
curl -LO $base/VOCtrainval_11-May-2012.zip
cd ../..
```

Resmi kaynak: <http://host.robots.ox.ac.uk/pascal/VOC/>. Arşivler `data/raw/` altında
beklenir:

```text
data/raw/VOCtrainval_06-Nov-2007.zip
data/raw/VOCtest_06-Nov-2007.zip
data/raw/VOCtrainval_11-May-2012.zip
```

Ardından:

```bash
env -u PYTHONPATH .venv/bin/python src/prepare_data.py
```

Bu komut YOLO formatındaki görüntü/etiketleri, `configs/voc_split.yaml` dosyasını,
split listelerini ve `results/dataset_stats.json` ile sınıf dağılımı grafiğini üretir.

## Eğitim

Smoke test (1 epoch, verinin %5'i):

```bash
env -u PYTHONPATH .venv/bin/python src/train.py --smoke
```

Tam eğitim (arka planda):

```bash
nohup env -u PYTHONPATH .venv/bin/python -u src/train.py --epochs 48 > results/train.log 2>&1 &
```

Hiperparametreler `configs/train_config.yaml` içindedir. Eğitim sonunda en iyi ağırlık
`weights/best.pt` konumuna kopyalanır; süre `results/train_time.json` dosyasına yazılır.

## Test Değerlendirmesi (bir kez)

```bash
env -u PYTHONPATH .venv/bin/python src/evaluate.py
```

Çıktılar: `results/test_metrics.json`, `results/test_metrics_per_class.csv`,
`report/figures/test_pred_*.jpg`, `report/figures/test_predictions_grid.png`.

## Hız Ölçümü

```bash
env -u PYTHONPATH .venv/bin/python src/benchmark.py
```

GPU ve CPU için preprocess / inference / postprocess süreleri ve FPS:
`results/inference_time.json`.

## Gradio Uygulaması

```bash
env -u PYTHONPATH .venv/bin/python app.py
```

Tarayıcıda `http://127.0.0.1:7860` adresini açın. Uygulama JPG/PNG/WEBP/HEIC yüklemeyi,
EXIF yön düzeltmesini, confidence slider'ını, tespit tablosunu ve inference süresini destekler.

## Gerçek Fotoğraflar

1. 5 fotoğrafı `real_photos/` klasörüne koyun.
2. `real_photos/ground_truth.csv` dosyasını doldurun:

```csv
filename,ground_truth,notes
foto1.jpg,"car:3;person:2","gündüz, sokak"
```

3. Çalıştırın:

```bash
env -u PYTHONPATH .venv/bin/python src/real_photos.py
```

Çıktılar: `report/figures/real_photo_*.jpg`, `results/real_photos_comparison.csv`,
`results/real_photos_summary.json`, `results/real_photos_draft.md`.

## Rapor

Ölçülen değerleri rapor placeholder'larına yazın ve Word dosyasını üretin:

```bash
env -u PYTHONPATH .venv/bin/python src/fill_report.py
env -u PYTHONPATH .venv/bin/python src/build_docx.py
```

- Markdown: `report/REPORT.md`
- Word: `report/REPORT.docx`
- Uygulama ekran görüntüsü: `src/screenshot_app.py` → `report/figures/app_screenshot.png`

## Repo Yapısı

```text
app.py                     Gradio uygulaması
configs/                   veri ve eğitim yapılandırmaları
src/                       veri hazırlama, eğitim, değerlendirme, benchmark, gerçek fotoğraf
weights/best.pt            eğitilmiş en iyi ağırlık
results/                   tüm ölçülen metrikler ve JSON/CSV çıktıları
report/                    rapor ve figürler
real_photos/               gerçek fotoğraflar ve ground truth şablonu
```

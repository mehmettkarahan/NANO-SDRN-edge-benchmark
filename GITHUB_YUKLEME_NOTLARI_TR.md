# GitHub'a ekleme

Bu klasörün içeriğini GitHub deponuzun kök dizinine yükleyin. ZIP dosyasını tek başına GitHub'a koymak yerine açıp dosyaları ve klasörleri yükleyin.

- İlk sayfa `README.md` olacaktır.
- Önerilen model klasörü `NANO-SDRN/` olarak adlandırılmıştır.
- PyTorch FP32/FP16 batch 1, 4, 8 ve 16 E2E sonuçları `results/cross_platform_e2e_batch1_4_8_16.csv` içindedir. Her model/koşulun zamanlama örnekleri `reproducibility/deployment/pytorch_e2e/` altındadır.
- Jetson TensorRT ve Jetson/Raspberry Pi 5 ONNX Runtime CPU aynı CSV'de sunulmuştur.
- Dokuz koşullu TOPSIS: `results/topsis/`.
- Ablasyon: `NANO-SDRN/ablation/`.
- Makalede yer almayan tahmin kalibrasyonu, seçici sınıflandırma, bootstrap ve thread-ölçeklenmesi sonuçları ana veri sunumundan çıkarılmıştır.
- Sayılar ve model etiketleri metrik kapsamlarıyla ayrılmıştır: `GPU-forward` ve `pipeline E2E`; batch=1 için bu iki sayıyı birbirinin yerine kullanmayın.

Kontrol için bilgisayarınızda pandas ve numpy kuruluysa şu komutları çalıştırın:

```bash
python scripts/validate_tables.py
python scripts/verify_topsis.py
```

Bu depo GitHub'a otomatik olarak gönderilmedi; yayımlama işlemi hesabınızda yapılmalıdır.

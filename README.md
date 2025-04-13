# Flappy Bird AI - Evrimsel Algoritma ile Öğrenme

Bu proje, Flappy Bird oyununu Python ve Pygame kullanarak gerçekleştirir ve evrimsel algoritmalar ile yapay zeka eğitimini gösterir. Evrimsel öğrenme yöntemi kullanarak, kuşlar oyunu oynamayı nesiller boyunca gelişerek öğrenirler.

## Özellikler

- Basit ve optimize edilmiş Flappy Bird oyun motoru
- İnsan oyuncular için oynanabilir mod
- Eğitim için evrimsel algoritma uygulaması
- Headless (görsel olmayan) veya görsel eğitim modları
- Eğitim performansı takibi ve grafikleri
- Eğitilmiş modelleri kaydetme ve yükleme

## Kurulum

1. Python 3.6+ kurun
2. Sanal ortam oluşturun ve aktifleştirin:

```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. Gereksinimleri yükleyin:

```bash
pip install -r requirements.txt
```

## Kullanım

Proje, aşağıdaki modlarda çalıştırılabilir:

### İnsan Olarak Oyna

```bash
python train.py --mode play_human
```

### Headless Modda AI Eğitimi

```bash
python train.py --mode train_headless --generations 100 --population 100
```

### Görsel Modda AI Eğitimi (daha az kuşla)

```bash
python train.py --mode train_visual --generations 20 --population 50
```

### Eğitilmiş AI ile Oyna

```bash
python train.py --mode play_ai --model models/best_model_gen_99.pkl
```

## Evrimsel Algoritma Açıklaması

Bu proje, aşağıdaki ana bileşenleri kullanarak evrimsel bir yaklaşım uygular:

1. **Popülasyon**: 100 kuş (bireyler) her nesilde eş zamanlı olarak oluşturulur
2. **Sinir Ağı**: Her kuş, 4 girişli (kuşun konumu, hızı, boruların konumu), 8 gizli nörona sahip basit bir sinir ağı ile kontrol edilir
3. **Uygunluk Fonksiyonu**: Kuşun skoru ve hayatta kalma süresi esas alınarak ölçülür
4. **Seçilim**: En yüksek uygunluk değerine sahip bireyler bir sonraki nesil için ebeveyn olarak seçilir
5. **Mutasyon**: Yeni nesildeki kuşların sinir ağı ağırlıkları rastgele mutasyona uğratılır

## Proje Yapısı

- `game.py`: Flappy Bird oyun motoru
- `neural_network.py`: Sinir ağı modeli ve evrimsel algoritma uygulaması
- `train.py`: Eğitim ve oyun modlarını içeren ana dosya
- `requirements.txt`: Gerekli Python kütüphaneleri
- `models/`: Eğitilmiş modellerin kaydedildiği dizin
- `plots/`: Eğitim istatistiklerinin grafiklerinin kaydedildiği dizin

## Performans İyileştirmeleri

- Headless mod, görselleştirme olmadan hızlı eğitim sağlar
- Optimize edilmiş çarpışma tespiti ve oyun fiziği
- İlerleme çubuğu ve ayrıntılı eğitim kayıtları
- Nesiller arasında en iyi modelin korunması 
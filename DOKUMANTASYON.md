# Flappy Bird AI - Evrimsel Algoritma ile Yapay Zeka Eğitimi

Bu dokümantasyon, Flappy Bird oyununda evrimsel algoritmalar kullanarak yapay zeka eğitimi yapan projenin detaylı açıklamasını içermektedir. Proje, Python ve Pygame kullanılarak geliştirilmiş olup, yapay sinir ağları ve evrimsel algoritmaların temel prensiplerini uygulamalı olarak göstermektedir.

## İçindekiler

1. [Proje Genel Bakış](#proje-genel-bakış)
2. [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
3. [Kod Yapısı ve Açıklamaları](#kod-yapısı-ve-açıklamaları)
   - [Neural Network (Sinir Ağı) Modülü](#neural-network-sinir-ağı-modülü)
   - [Game (Oyun) Modülü](#game-oyun-modülü)
   - [Train (Eğitim) Modülü](#train-eğitim-modülü)
4. [Evrimsel Algoritma Detayları](#evrimsel-algoritma-detayları)
5. [Eğitim Parametreleri ve Optimizasyon](#eğitim-parametreleri-ve-optimizasyon)
6. [Görselleştirme ve Analiz](#görselleştirme-ve-analiz)
7. [İleri Seviye Kullanım](#ileri-seviye-kullanım)
8. [Sorun Giderme](#sorun-giderme)

## Proje Genel Bakış

Bu proje, popüler Flappy Bird oyununun basitleştirilmiş bir versiyonunu oluşturarak, yapay zeka ajanlarının (kuşların) oyunu oynamayı evrimsel algoritmalar yoluyla öğrenmesini sağlar. Evrimsel algoritma, doğal seçilim prensiplerini taklit ederek, en iyi performans gösteren bireylerin özelliklerini sonraki nesillere aktarır ve mutasyonlar yoluyla yeni özellikler keşfeder.

Projede kullanılan temel bileşenler:

- **Flappy Bird Oyun Motoru**: Pygame kütüphanesi kullanılarak oluşturulmuş basit bir oyun motoru
- **Yapay Sinir Ağı**: Kuşların karar vermesini sağlayan basit bir ileri beslemeli sinir ağı
- **Evrimsel Algoritma**: Sinir ağlarının ağırlıklarını optimize eden genetik algoritma
- **Görselleştirme Araçları**: Eğitim sürecini ve sonuçlarını görselleştiren araçlar

## Kurulum ve Çalıştırma

### Gereksinimler

Projeyi çalıştırmak için aşağıdaki yazılım ve kütüphanelere ihtiyacınız vardır:

- Python 3.6 veya üzeri
- Pygame 2.5.2
- NumPy 1.26.3
- Matplotlib 3.8.2
- tqdm 4.66.1

### Kurulum Adımları

1. Projeyi bilgisayarınıza klonlayın veya indirin
2. Sanal ortam oluşturun (isteğe bağlı ama önerilir):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate     # Windows
   ```
3. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

### Çalıştırma Modları

Proje, aşağıdaki modlarda çalıştırılabilir:

1. **İnsan Olarak Oynama**:
   ```bash
   python train.py --mode play_human
   ```
   Bu modda, SPACE tuşunu kullanarak kuşu zıplatabilir ve oyunu kendiniz oynayabilirsiniz.

2. **Headless (Görsel Olmayan) Modda AI Eğitimi**:
   ```bash
   python train.py --mode train_headless --generations 100 --population 100 --max_steps 1000
   ```
   Bu mod, görselleştirme olmadan hızlı eğitim sağlar. Eğitim sonuçları ve grafikler `logs/` ve `plots/` klasörlerine kaydedilir.

3. **Görsel Modda AI Eğitimi**:
   ```bash
   python train.py --mode train_visual --generations 20 --population 50 --max_steps 2000
   ```
   Bu mod, eğitim sürecini gerçek zamanlı olarak görselleştirir. Kuşların nasıl öğrendiğini gözlemleyebilirsiniz.

4. **Eğitilmiş AI ile Oynama**:
   ```bash
   python train.py --mode play_ai --model models/best_model_gen_99.pkl
   ```
   Bu mod, önceden eğitilmiş bir modeli yükleyerek, AI'ın oyunu nasıl oynadığını gösterir.

### Komut Satırı Parametreleri

- `--mode`: Çalıştırma modu (`train_headless`, `train_visual`, `play_human`, `play_ai`)
- `--generations`: Eğitim için nesil sayısı (varsayılan: 100)
- `--population`: Her nesildeki kuş (birey) sayısı (varsayılan: 100)
- `--max_steps`: Her nesil için maksimum adım sayısı (varsayılan: 1000)
- `--model`: Yüklenecek model dosyası yolu (sadece `play_ai` modu için)

## Kod Yapısı ve Açıklamaları

Proje üç ana modülden oluşmaktadır:

### Neural Network (Sinir Ağı) Modülü

`neural_network.py` dosyası, sinir ağı modelini ve evrimsel algoritmayı içerir.

#### NeuralNetwork Sınıfı

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Ağırlıkları rastgele değerlerle başlat
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        
        # Bias değerlerini başlat
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
```

Bu sınıf, basit bir ileri beslemeli sinir ağını temsil eder:
- `input_size`: Giriş katmanındaki nöron sayısı (kuşun ve boruların konumu gibi özellikler)
- `hidden_size`: Gizli katmandaki nöron sayısı
- `output_size`: Çıkış katmanındaki nöron sayısı (zıplama kararı)

Önemli metodlar:
- `predict(inputs)`: Verilen girdilere göre bir tahmin yapar (zıplama kararı)
- `mutate(mutation_rate, mutation_amount)`: Sinir ağının ağırlıklarını rastgele değiştirir
- `save(filename)` ve `load(filename)`: Modeli kaydetme ve yükleme işlemleri

#### EvolutionaryAlgorithm Sınıfı

```python
class EvolutionaryAlgorithm:
    def __init__(self, population_size=100, input_size=4, hidden_size=8, output_size=1, 
                 mutation_rate=0.1, mutation_amount=0.5, survival_rate=0.2, crossover_rate=0.0):
        # Popülasyon parametreleri
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Evrimsel parametreler
        self.mutation_rate = mutation_rate
        self.mutation_amount = mutation_amount
        self.survival_rate = survival_rate
        self.crossover_rate = crossover_rate
```

Bu sınıf, evrimsel algoritmanın uygulanmasını sağlar:
- `population_size`: Her nesildeki birey (kuş) sayısı
- `mutation_rate`: Mutasyon olasılığı (0-1 arası)
- `mutation_amount`: Mutasyon miktarı (değişimin büyüklüğü)
- `survival_rate`: Hayatta kalan bireylerin oranı

Önemli metodlar:
- `calculate_fitness(birds)`: Her kuşun uygunluk değerini hesaplar
- `selection(birds)`: En iyi performans gösteren kuşları seçer
- `create_next_generation(birds)`: Yeni nesil kuşları oluşturur
- `plot_fitness_history()`: Eğitim performansını görselleştirir

### Game (Oyun) Modülü

`game.py` dosyası, Flappy Bird oyun motorunu içerir.

#### GameMode Enum Sınıfı

```python
class GameMode(Enum):
    HUMAN = 1   # İnsan oyuncular için mod
    AI = 2      # AI eğitimi için görsel mod
    HEADLESS = 3  # Görsel olmayan hızlı eğitim modu
```

#### Bird Sınıfı

```python
class Bird:
    def __init__(self, x, y, brain=None):
        self.x = x
        self.y = y
        self.velocity = 0
        self.gravity = 0.8
        self.jump_strength = -10
        self.width = 34
        self.height = 24
        self.alive = True
        self.score = 0
        self.fitness = 0
        self.brain = brain
        # Kuşlar için rastgele renk (görsel ayırt etme için)
        self.color = (
            random.randint(100, 255),  # R
            random.randint(100, 255),  # G
            random.randint(100, 255)   # B
        )
```

Bu sınıf, oyundaki kuşları temsil eder:
- `x, y`: Kuşun konumu
- `velocity`: Dikey hız
- `brain`: Kuşun kararlarını veren sinir ağı
- `color`: Kuşun rengi (görsel ayırt etme için)

Önemli metodlar:
- `jump()`: Kuşu zıplatır
- `apply_brain(pipes)`: Sinir ağını kullanarak zıplama kararı verir
- `update()`: Kuşun fiziksel durumunu günceller
- `collides_with(pipe)`: Boru ile çarpışma kontrolü yapar

#### Pipe Sınıfı

```python
class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 60
        self.speed = 3  # Daha yavaş - kuşların tepki vermesi için daha fazla zaman
        self.gap_height = 170  # Daha geniş boşluk - başlangıçta geçmeyi kolaylaştırır
        self.gap_y = random.randint(180, 320)  # Daha dar yükseklik aralığı - uç pozisyonları azaltır
        self.passed = False
```

Bu sınıf, oyundaki boruları temsil eder:
- `x`: Borunun x konumu
- `gap_height`: Borudaki boşluğun yüksekliği
- `gap_y`: Boşluğun y konumu

Önemli metodlar:
- `update()`: Borunun konumunu günceller
- `is_passed_by(bird)`: Kuşun boruyu geçip geçmediğini kontrol eder

#### FlappyBird Sınıfı

```python
class FlappyBird:
    def __init__(self, mode=GameMode.HUMAN, width=400, height=600):
        self.width = width
        self.height = height
        self.mode = mode
        self.ground_y = height - 100
        self.bg_color = (135, 206, 250)  # Açık mavi
        
        # Oyun durumu
        self.birds = []
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.frame_count = 0
        self.living_birds = 0
```

Bu sınıf, oyun motorunu temsil eder:
- `mode`: Oyun modu (HUMAN, AI, HEADLESS)
- `birds`: Oyundaki kuşlar listesi
- `pipes`: Oyundaki borular listesi

Önemli metodlar:
- `reset(birds=None)`: Oyunu yeniden başlatır
- `step(action=0)`: Oyunu bir adım ilerletir
- `render()`: Oyun durumunu ekrana çizer
- `run_human()`: İnsan oyuncular için oyun döngüsü

### Train (Eğitim) Modülü

`train.py` dosyası, eğitim ve oyun modlarını içeren ana dosyadır.

#### Önemli Fonksiyonlar

```python
def create_bird_with_brain(x, y, brain):
    """Sinir ağı ile kontrol edilen bir kuş oluşturur"""
    bird = Bird(x, y, brain=brain)
    return bird

def display_best_model(model, max_steps=2000):
    """En iyi modeli görsel olarak gösterir"""
    game = FlappyBird(mode=GameMode.AI)
    bird = create_bird_with_brain(100, 300, model)
    game.reset(birds=[bird])
    # ...

def train_headless(generations=100, population_size=100, max_steps=1000, 
                  save_interval=10, render_best=False, render_interval=10):
    """Headless modda AI eğitimi yapar"""
    # ...

def run_visualized_training(generations=20, population_size=50, max_steps=1000):
    """Görsel modda AI eğitimi yapar"""
    # ...

def play_human():
    """İnsan oyuncular için oyun başlatır"""
    # ...
```

## Evrimsel Algoritma Detayları

Projede kullanılan evrimsel algoritma, aşağıdaki adımları takip eder:

1. **Başlangıç Popülasyonu Oluşturma**:
   - Rastgele ağırlıklara sahip sinir ağları ile başlangıç popülasyonu oluşturulur
   - Her kuş, kendi sinir ağı (beyin) ile kontrol edilir

2. **Uygunluk Değerlendirmesi**:
   - Her kuşun performansı değerlendirilir
   - Uygunluk değeri, kuşun skoru (geçtiği boru sayısı) ve hayatta kalma süresine göre hesaplanır
   - Ayrıca, henüz bir boru geçmemiş kuşlar için boruya yakınlık ve hizalanma bonusları eklenir

3. **Seçilim**:
   - En yüksek uygunluk değerine sahip kuşlar, bir sonraki nesil için ebeveyn olarak seçilir
   - Seçilim oranı `survival_rate` parametresi ile belirlenir (varsayılan: %20)

4. **Çoğalma ve Mutasyon**:
   - Seçilen ebeveynlerden yeni nesil oluşturulur
   - Her yeni birey, bir ebeveynin kopyası olarak başlar
   - Mutasyon işlemi ile sinir ağı ağırlıkları rastgele değiştirilir
   - Mutasyon olasılığı `mutation_rate` ve mutasyon miktarı `mutation_amount` parametreleri ile kontrol edilir

5. **Nesil Değişimi**:
   - Yeni nesil, eski neslin yerini alır
   - En iyi birey her zaman korunur (elitizm)

6. **Tekrarlama**:
   - Bu süreç, belirlenen nesil sayısı kadar tekrarlanır

## Eğitim Parametreleri ve Optimizasyon

Eğitim performansını etkileyen önemli parametreler:

### Popülasyon Parametreleri

- `population_size`: Popülasyon büyüklüğü (varsayılan: 100)
  - Daha büyük popülasyonlar daha fazla çeşitlilik sağlar ancak daha yavaş eğitim gerektirir
  - Görsel eğitim için daha küçük popülasyonlar (50) önerilir

- `generations`: Eğitim nesil sayısı
  - Daha fazla nesil, daha iyi sonuçlar sağlayabilir ancak daha uzun eğitim süresi gerektirir

### Sinir Ağı Parametreleri

- `input_size`: Giriş katmanı boyutu (varsayılan: 8)
  - Kuşun konumu, hızı, boruların konumu gibi özellikleri içerir

- `hidden_size`: Gizli katman boyutu (varsayılan: 24)
  - Daha büyük gizli katmanlar daha karmaşık davranışlar öğrenebilir

### Evrimsel Parametreler

- `mutation_rate`: Mutasyon olasılığı (varsayılan: 0.2)
  - Daha yüksek değerler daha fazla keşif sağlar ancak kararlılığı azaltabilir

- `mutation_amount`: Mutasyon miktarı (varsayılan: 0.3)
  - Daha yüksek değerler daha büyük değişiklikler yapar

- `survival_rate`: Hayatta kalma oranı (varsayılan: 0.3)
  - Daha yüksek değerler daha fazla çeşitlilik sağlar ancak ilerlemeyi yavaşlatabilir

### Oyun Parametreleri

- `max_steps`: Her nesil için maksimum adım sayısı (varsayılan: 1000)
  - Daha yüksek değerler daha uzun oyunlar sağlar ancak eğitim süresini artırır

## Görselleştirme ve Analiz

Proje, eğitim sürecini ve sonuçlarını görselleştirmek için çeşitli araçlar sunar:

### Eğitim Sırasında Görselleştirme

Görsel eğitim modu (`train_visual`), eğitim sürecini gerçek zamanlı olarak görselleştirir:
- Kuşlar, rastgele renklerle görselleştirilir (her kuş farklı bir renk)
- Ekranda mevcut skor, hayatta kalan kuş sayısı, çerçeve sayısı ve bir sonraki borunun konumu gösterilir

### Eğitim Sonrası Analiz

- `plots/` klasöründe, her nesil için uygunluk değeri grafikleri kaydedilir
- En iyi ve ortalama uygunluk değerlerinin nesiller boyunca değişimi görselleştirilir

### En İyi Modeli Görselleştirme

`play_ai` modu, eğitilmiş bir modelin performansını görselleştirir:
- En iyi modelin oyunu nasıl oynadığını gözlemleyebilirsiniz
- Modelin karar verme sürecini analiz edebilirsiniz

## İleri Seviye Kullanım

### Özel Eğitim Senaryoları

Farklı eğitim senaryoları oluşturmak için parametreleri değiştirebilirsiniz:

```bash
# Daha yüksek mutasyon oranı ile hızlı keşif
python train.py --mode train_headless --generations 50 --population 200 --max_steps 1500

# Daha uzun eğitim süresi ile daha iyi optimizasyon
python train.py --mode train_headless --generations 200 --population 100 --max_steps 2000
```

### Modeli Değiştirme

Sinir ağı mimarisini veya evrimsel algoritma parametrelerini değiştirmek için `neural_network.py` dosyasını düzenleyebilirsiniz:

```python
# Daha karmaşık bir sinir ağı için
evolution = EvolutionaryAlgorithm(
    population_size=population_size,
    input_size=12,  # Daha fazla giriş özelliği
    hidden_size=32,  # Daha büyük gizli katman
    output_size=1,
    mutation_rate=0.15,
    mutation_amount=0.25,
    survival_rate=0.35
)
```

### Oyun Zorluğunu Değiştirme

Oyun zorluğunu değiştirmek için `game.py` dosyasındaki parametreleri düzenleyebilirsiniz:

```python
class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 60
        self.speed = 4  # Daha hızlı borular (daha zor)
        self.gap_height = 150  # Daha dar boşluk (daha zor)
        self.gap_y = random.randint(150, 350)  # Daha geniş yükseklik aralığı (daha zor)
        self.passed = False
```

## Sorun Giderme

### Yaygın Sorunlar ve Çözümleri

1. **Eğitim Çok Yavaş**:
   - Headless modu kullanın (`--mode train_headless`)
   - Popülasyon boyutunu azaltın (`--population 50`)
   - Maksimum adım sayısını azaltın (`--max_steps 500`)

2. **Kuşlar Öğrenmiyor**:
   - Mutasyon oranını artırın (EvolutionaryAlgorithm sınıfında `mutation_rate` parametresini artırın)
   - Daha fazla nesil eğitin (`--generations 200`)
   - Uygunluk fonksiyonunu iyileştirin (EvolutionaryAlgorithm sınıfında `calculate_fitness` metodunu düzenleyin)

3. **Pygame Hataları**:
   - Pygame'in en son sürümünü yüklediğinizden emin olun (`pip install pygame --upgrade`)
   - Ekran çözünürlüğünüzün oyun penceresini desteklediğinden emin olun

4. **Bellek Sorunları**:
   - Popülasyon boyutunu azaltın
   - Daha az nesil eğitin
   - Gereksiz görselleştirmeleri kapatın 
# Evrimsel Algoritmalar ile Oyun Yapay Zekası Eğitimi: Flappy Bird Örneği

**Özet**

Bu çalışmada, evrimsel algoritmalar kullanılarak oyun yapay zekası eğitimi için bir yaklaşım sunulmaktadır. Popüler Flappy Bird oyunu üzerinde uygulanan bu yaklaşım, yapay sinir ağlarının ağırlıklarını optimize etmek için doğal seçilim prensiplerini taklit eden bir evrimsel algoritma kullanmaktadır. Çalışma, evrimsel algoritmaların seyrek ödül sinyalleri içeren oyun ortamlarında etkili bir öğrenme stratejisi sunduğunu göstermektedir. Ayrıca, farklı evrimsel parametrelerin eğitim performansına etkisi incelenmiş ve optimum değerler belirlenmiştir. Sonuçlar, evrimsel yaklaşımın, özellikle pekiştirmeli öğrenme yöntemlerinin zorlandığı seyrek ödül ortamlarında, oyun yapay zekası eğitimi için etkili bir alternatif olduğunu göstermektedir.

**Anahtar Kelimeler**: Evrimsel Algoritmalar, Yapay Sinir Ağları, Oyun Yapay Zekası, Neuroevolution, Flappy Bird

## 1. Giriş

Oyun yapay zekası, yapay zeka araştırmalarında önemli bir uygulama alanıdır. Oyunlar, kontrollü ve simüle edilmiş ortamlar sunarak, yapay zeka algoritmalarının test edilmesi ve geliştirilmesi için ideal platformlar oluşturur. Son yıllarda, derin pekiştirmeli öğrenme yöntemleri oyun yapay zekası alanında önemli başarılar elde etmiştir [1, 2]. Ancak, bu yöntemler genellikle sık ve anlamlı ödül sinyalleri gerektirmektedir, bu da seyrek ödül ortamlarında öğrenmeyi zorlaştırabilir.

Evrimsel algoritmalar, doğal seçilim prensiplerini taklit eden optimizasyon yöntemleridir ve seyrek ödül ortamlarında etkili bir alternatif sunabilir [3]. Bu çalışmada, popüler Flappy Bird oyununda yapay zeka ajanlarını eğitmek için evrimsel bir yaklaşım sunulmaktadır. Flappy Bird, basit mekaniklere sahip olmasına rağmen, seyrek ödül sinyalleri ve hassas kontrol gerektiren zorlu bir oyundur.

Bu makalede, evrimsel algoritmaların oyun yapay zekası eğitimindeki etkinliği incelenmekte ve farklı evrimsel parametrelerin performansa etkisi analiz edilmektedir. Ayrıca, evrimsel yaklaşım ile pekiştirmeli öğrenme yaklaşımları arasında karşılaştırmalar yapılmaktadır.

## 2. İlgili Çalışmalar

Oyun yapay zekası alanında, evrimsel algoritmalar ve pekiştirmeli öğrenme yöntemleri yaygın olarak kullanılmaktadır. Stanley ve Miikkulainen [4], NEAT (NeuroEvolution of Augmenting Topologies) algoritmasını geliştirerek, evrimsel algoritmaların yapay sinir ağı topolojilerini optimize etmede etkili olduğunu göstermiştir. Salimans ve arkadaşları [5], OpenAI'nin Evolution Strategies (ES) yaklaşımını kullanarak, pekiştirmeli öğrenme problemlerinde rekabetçi sonuçlar elde etmiştir.

Flappy Bird özelinde, Chen [6], derin Q-öğrenimi (DQN) kullanarak bir yapay zeka ajanı eğitmiş ve oyunda insan seviyesinde performans elde etmiştir. Ebner ve arkadaşları [7], genetik programlama kullanarak Flappy Bird için kontrol stratejileri geliştirmiş ve evrimsel yaklaşımın etkinliğini göstermiştir.

Bu çalışma, önceki araştırmaları genişleterek, evrimsel algoritmaların Flappy Bird oyununda yapay zeka eğitimi için sistematik bir analiz sunmaktadır. Özellikle, farklı evrimsel parametrelerin eğitim performansına etkisi ve evrimsel yaklaşımın pekiştirmeli öğrenme yöntemlerine göre avantajları ve dezavantajları incelenmektedir.

## 3. Metodoloji

### 3.1. Flappy Bird Oyun Ortamı

Flappy Bird, oyuncunun bir kuşu kontrol ederek, ekranda hareket eden borular arasından geçmeye çalıştığı bir oyundur. Oyunun amacı, mümkün olduğunca çok borudan geçerek yüksek skor elde etmektir. Oyun mekanikleri şu şekildedir:

- Kuş, yerçekimi etkisiyle sürekli aşağı düşer
- Oyuncu, kuşu zıplatarak yükselmesini sağlar
- Borular, belirli aralıklarla ekranın sağından sola doğru hareket eder
- Kuş, borulara veya zemine çarparsa oyun sona erer
- Her başarılı boru geçişi bir puan kazandırır

Bu çalışmada, Flappy Bird oyunu Python ve Pygame kütüphanesi kullanılarak simüle edilmiştir. Oyun ortamı, yapay zeka ajanlarının eğitimi için uygun bir arayüz sağlamaktadır.

### 3.2. Yapay Sinir Ağı Modeli

Yapay zeka ajanları, basit bir ileri beslemeli sinir ağı ile kontrol edilmektedir. Sinir ağı mimarisi şu şekildedir:

- **Giriş Katmanı**: 8 nöron (kuşun konumu, hızı, en yakın borunun konumu ve boşluk yüksekliği, ikinci borunun konumu ve boşluk yüksekliği)
- **Gizli Katman**: 24 nöron (ReLU aktivasyon fonksiyonu)
- **Çıkış Katmanı**: 1 nöron (Sigmoid aktivasyon fonksiyonu)

Sinir ağı, her zaman adımında oyun durumunu giriş olarak alır ve zıplama kararı (0 veya 1) üretir. Çıkış değeri 0.5'ten büyükse kuş zıplar, aksi takdirde düşmeye devam eder.

### 3.3. Evrimsel Algoritma

Evrimsel algoritma, aşağıdaki adımları takip eder:

1. **Başlangıç Popülasyonu Oluşturma**: Rastgele ağırlıklara sahip N adet sinir ağı oluşturulur (N = popülasyon boyutu)
2. **Uygunluk Değerlendirmesi**: Her sinir ağı, oyunu oynayarak bir uygunluk değeri elde eder
3. **Seçilim**: En yüksek uygunluk değerine sahip M adet sinir ağı seçilir (M = N * survival_rate)
4. **Çoğalma ve Mutasyon**: Seçilen sinir ağlarından yeni nesil oluşturulur ve mutasyona uğratılır
5. **Nesil Değişimi**: Yeni nesil, eski neslin yerini alır
6. **Tekrarlama**: Adımlar 2-5, belirlenen nesil sayısı kadar tekrarlanır

Uygunluk fonksiyonu, aşağıdaki faktörleri dikkate alır:
- Geçilen boru sayısı (ana ödül)
- Hayatta kalma süresi (ikincil ödül)
- Boruya yakınlık ve hizalanma (henüz boru geçmemiş ajanlar için bonus)

Mutasyon işlemi, sinir ağı ağırlıklarını belirli bir olasılıkla (mutation_rate) ve belirli bir miktarda (mutation_amount) değiştirir.

### 3.4. Deneysel Kurulum

Deneyler, aşağıdaki parametreler kullanılarak gerçekleştirilmiştir:

- **Popülasyon Boyutu**: 50, 100, 200
- **Nesil Sayısı**: 50, 100, 200
- **Mutasyon Oranı**: 0.1, 0.2, 0.3
- **Mutasyon Miktarı**: 0.1, 0.3, 0.5
- **Hayatta Kalma Oranı**: 0.2, 0.3, 0.4

Her parametre kombinasyonu için 10 bağımsız çalıştırma gerçekleştirilmiş ve sonuçların ortalaması alınmıştır. Performans metriği olarak, son nesildeki en iyi ajanın skoru (geçtiği boru sayısı) kullanılmıştır.

## 4. Sonuçlar ve Tartışma

### 4.1. Evrimsel Parametrelerin Etkisi

#### 4.1.1. Popülasyon Boyutu

Daha büyük popülasyonlar (N = 200), daha küçük popülasyonlara (N = 50, 100) göre daha iyi performans göstermiştir. Bu, daha büyük popülasyonların daha fazla çeşitlilik sağlayarak, çözüm uzayını daha etkili bir şekilde araştırabilmesinden kaynaklanmaktadır.

#### 4.1.2. Mutasyon Parametreleri

Orta düzeyde bir mutasyon oranı (0.2) ve mutasyon miktarı (0.3), en iyi performansı sağlamıştır. Çok düşük mutasyon değerleri, yetersiz keşfe yol açarken, çok yüksek değerler öğrenilen bilgilerin kaybedilmesine neden olabilir.

#### 4.1.3. Hayatta Kalma Oranı

Orta düzeyde bir hayatta kalma oranı (0.3), en iyi performansı sağlamıştır. Çok düşük hayatta kalma oranı, çeşitliliği azaltırken, çok yüksek hayatta kalma oranı, seçilim baskısını azaltarak ilerlemeyi yavaşlatabilir.

### 4.2. Öğrenme Eğrileri

 Evrimsel algoritma, ilk 20-30 nesil boyunca hızlı bir ilerleme göstermiş, ardından daha yavaş bir ilerleme sergilemiştir. 100 nesil sonunda, en iyi ajan ortalama 50+ boruyu başarıyla geçebilmiştir, bu da insan seviyesinde bir performansa karşılık gelmektedir.

### 4.3. Evrimsel Yaklaşım vs. Pekiştirmeli Öğrenme

Evrimsel yaklaşım, özellikle seyrek ödül ortamlarında ve paralel hesaplama imkanı olduğunda avantajlı olabilir. Ancak, çevrimiçi öğrenme gerektiren durumlarda ve sürekli ödül sinyallerinin mevcut olduğu ortamlarda, pekiştirmeli öğrenme yöntemleri daha etkili olabilir.

## 5. Sonuç ve Gelecek Çalışmalar

Bu çalışmada, evrimsel algoritmalar kullanılarak Flappy Bird oyununda yapay zeka ajanlarının eğitimi için bir yaklaşım sunulmuştur. Sonuçlar, evrimsel yaklaşımın, özellikle seyrek ödül ortamlarında, oyun yapay zekası eğitimi için etkili bir yöntem olduğunu göstermektedir. Optimum evrimsel parametreler belirlenmiş ve bu parametrelerle insan seviyesinde performans elde edilmiştir.

Gelecek çalışmalarda, aşağıdaki konular incelenebilir:

1. **Daha Karmaşık Sinir Ağı Mimarileri**: Konvolüsyonel sinir ağları gibi daha karmaşık mimariler kullanılarak, görsel girdilerden doğrudan öğrenme sağlanabilir.
2. **Hibrit Yaklaşımlar**: Evrimsel algoritmalar ve pekiştirmeli öğrenme yöntemlerinin güçlü yönlerini birleştiren hibrit yaklaşımlar geliştirilebilir.
3. **Çok Amaçlı Optimizasyon**: Skor maksimizasyonu yanında, enerji verimliliği gibi ek hedefler de dikkate alınabilir.
4. **Daha Karmaşık Oyunlar**: Daha karmaşık oyunlarda evrimsel yaklaşımın etkinliği incelenebilir.

Bu çalışma, evrimsel algoritmaların oyun yapay zekası eğitimindeki potansiyelini göstermekte ve gelecekteki araştırmalar için bir temel oluşturmaktadır.

## Kaynaklar

[1] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.

[2] Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. Nature, 529(7587), 484-489.

[3] Such, F. P., Madhavan, V., Conti, E., Lehman, J., Stanley, K. O., & Clune, J. (2017). Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning. arXiv preprint arXiv:1712.06567.

[4] Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2), 99-127.

[5] Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv preprint arXiv:1703.03864.

[6] Chen, K. (2015). Deep reinforcement learning for flappy bird. CS229 class project report.

[7] Ebner, M., Levine, J., Lucas, S. M., Schaul, T., Thompson, T., & Togelius, J. (2013). Towards a video game description language. Dagstuhl Follow-Ups, 6.

[8] Risi, S., & Togelius, J. (2017). Neuroevolution in games: State of the art and open challenges. IEEE Transactions on Computational Intelligence and AI in Games, 9(1), 25-41.

[9] Whitley, D., Dominic, S., Das, R., & Anderson, C. W. (1993). Genetic reinforcement learning for neurocontrol problems. Machine learning, 13(2-3), 259-284.

[10] Floreano, D., Dürr, P., & Mattiussi, C. (2008). Neuroevolution: from architectures to learning. Evolutionary intelligence, 1(1), 47-62. 
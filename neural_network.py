import numpy as np
import random
import copy
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Ağırlıkları rastgele değerlerle başlat (küçük değerlerle çarparak başlangıç gradyanlarını kontrol et)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        
        # Bias değerlerini sıfır olarak başlat
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        # Sigmoid aktivasyon fonksiyonu (0-1 arasında değer döndürür)
        return 1 / (1 + np.exp(-x))
    
    def predict(self, inputs):
        # Girdileri numpy dizisine dönüştür
        inputs = np.array(inputs).reshape(1, -1)
        
        # İleri besleme işlemi
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        final_outputs = self.sigmoid(final_inputs)
        
        return final_outputs[0][0]  # Tek çıktı değerini döndür
    
    def copy(self):
        # Sinir ağının derin kopyasını oluştur
        return copy.deepcopy(self)
    
    def mutate(self, mutation_rate=0.1, mutation_amount=0.5):
        """Sinir ağının ağırlıklarını ve bias değerlerini rastgele mutasyona uğrat"""
        
        # Bir ağırlık matrisini mutasyona uğratmak için yardımcı fonksiyon
        def mutate_matrix(matrix):
            # Hangi ağırlıkların mutasyona uğrayacağını belirleyen maske oluştur
            mask = np.random.random(matrix.shape) < mutation_rate
            
            # Rastgele pertürbasyonlar oluştur
            perturbations = np.random.randn(*matrix.shape) * mutation_amount
            
            # Mutasyonları sadece maskenin True olduğu yerlere uygula
            matrix[mask] += perturbations[mask]
            
            return matrix
        
        # Her ağırlık ve bias setini mutasyona uğrat
        self.weights_input_hidden = mutate_matrix(self.weights_input_hidden)
        self.weights_hidden_output = mutate_matrix(self.weights_hidden_output)
        self.bias_hidden = mutate_matrix(self.bias_hidden)
        self.bias_output = mutate_matrix(self.bias_output)
    
    def save(self, filename):
        """Sinir ağını bir dosyaya kaydet"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        """Bir dosyadan sinir ağı yükle"""
        with open(filename, 'rb') as f:
            return pickle.load(f)

class EvolutionaryAlgorithm:
    def __init__(self, population_size=100, input_size=4, hidden_size=8, output_size=1, 
                 mutation_rate=0.1, mutation_amount=0.5, survival_rate=0.2, crossover_rate=0.0):
        # Popülasyon parametreleri
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Evrimsel parametreler
        self.mutation_rate = mutation_rate  # Mutasyon olasılığı
        self.mutation_amount = mutation_amount  # Mutasyon miktarı
        self.survival_rate = survival_rate  # Hayatta kalma oranı
        self.crossover_rate = crossover_rate  # Çaprazlama oranı (şu anda kullanılmıyor)
        
        # En iyi model takibi
        self.best_fitness = 0
        self.best_model = None
        self.generation = 0
        self.fitness_history = []
        self.avg_fitness_history = []
        
        # Popülasyonu başlat
        self.population = self._initialize_population()
    
    def _initialize_population(self):
        """Sinir ağlarından oluşan bir popülasyon başlat"""
        return [NeuralNetwork(self.input_size, self.hidden_size, self.output_size) 
                for _ in range(self.population_size)]
    
    def calculate_fitness(self, birds):
        """Her kuş için uygunluk değerini hesapla"""
        total_fitness = 0
        max_fitness = 0
        
        # Boru değerlendirmesi için oyun durumu (eğer pipes erişilebilirse)
        pipes = None
        if hasattr(birds[0], 'game') and hasattr(birds[0].game, 'pipes'):
            pipes = birds[0].game.pipes
        
        for i, bird in enumerate(birds):
            # Uygunluk değeri öncelikle skora dayanır (daha yüksek çarpan)
            fitness = bird.score * 20  # Daha yüksek çarpan
            
            # Henüz bir boru geçmemiş ama yaşayan kuşlara ek bonus ver
            if bird.score == 0 and bird.alive and pipes:
                # En yakın boruyu bul
                next_pipe = None
                for pipe in pipes:
                    if pipe.x + pipe.width > bird.x:
                        next_pipe = pipe
                        break
                
                if next_pipe:
                    # Boruya yakınlık için bonus (boru yaklaştıkça artan)
                    proximity = max(0, 1 - ((next_pipe.x - bird.x) / 400))
                    fitness += proximity * 3
                    
                    # Boşluğa hizalanma için bonus
                    alignment = max(0, 1 - (abs(next_pipe.gap_y - bird.y) / 150))
                    fitness += alignment * 5
            
            # Hayatta kalma bonusu
            if bird.alive:
                fitness += 2
            
            # Uygunluk değerini kaydet
            bird.fitness = fitness
            
            total_fitness += fitness
            max_fitness = max(max_fitness, fitness)
            
            # En iyi modeli sakla
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_model = bird.brain.copy()
        
        # Uygunluk istatistiklerini güncelle
        self.fitness_history.append(max_fitness)
        self.avg_fitness_history.append(total_fitness / len(birds))
        
        return total_fitness, max_fitness
    
    def selection(self, birds):
        """Uygunluk değerine göre kuşları seçme (uygunluk orantılı seçim)"""
        # Kuşları uygunluk değerine göre azalan sırada sırala
        sorted_birds = sorted(birds, key=lambda x: x.fitness, reverse=True)
        
        # En iyi performans gösterenleri seç
        survivors_count = max(2, int(self.population_size * self.survival_rate))
        survivors = sorted_birds[:survivors_count]
        
        return survivors
    
    def crossover(self, parent1, parent2):
        """İki ebeveynden çaprazlama ile çocuk oluştur (şu anki uygulamada kullanılmıyor)"""
        # Bu basit bir uygulama - daha karmaşık çaprazlama yöntemleriyle genişletilebilir
        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        
        # Her ebeveynden rastgele ağırlıklar seç
        mask = np.random.random(parent1.weights_input_hidden.shape) < 0.5
        child.weights_input_hidden = np.where(mask, parent1.weights_input_hidden, parent2.weights_input_hidden)
        
        mask = np.random.random(parent1.weights_hidden_output.shape) < 0.5
        child.weights_hidden_output = np.where(mask, parent1.weights_hidden_output, parent2.weights_hidden_output)
        
        mask = np.random.random(parent1.bias_hidden.shape) < 0.5
        child.bias_hidden = np.where(mask, parent1.bias_hidden, parent2.bias_hidden)
        
        mask = np.random.random(parent1.bias_output.shape) < 0.5
        child.bias_output = np.where(mask, parent1.bias_output, parent2.bias_output)
        
        return child
    
    def create_next_generation(self, birds):
        """Bir sonraki nesil kuşları oluştur"""
        # Uygunluk değerlerini hesapla
        total_fitness, max_fitness = self.calculate_fitness(birds)
        
        # Ebeveynleri seç
        parents = self.selection(birds)
        
        # Yeni popülasyon oluştur
        new_population = []
        
        # Her zaman en iyi modeli koru (elitizm)
        if self.best_model:
            new_population.append(self.best_model.copy())
        
        # Popülasyonun geri kalanını doldur
        while len(new_population) < self.population_size:
            # Rastgele ebeveyn seç
            parent = random.choice(parents)
            
            # Ebeveynin beyninin bir kopyasını oluştur
            child_brain = parent.brain.copy()
            
            # Mutasyon uygula
            child_brain.mutate(self.mutation_rate, self.mutation_amount)
            
            new_population.append(child_brain)
        
        self.population = new_population
        self.generation += 1
        
        return new_population
    
    def plot_fitness_history(self, save_path=None):
        """Uygunluk değeri geçmişini görselleştir"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.fitness_history, label='En İyi Uygunluk')
        plt.plot(self.avg_fitness_history, label='Ortalama Uygunluk')
        plt.xlabel('Nesil')
        plt.ylabel('Uygunluk Değeri')
        plt.title('Nesiller Boyunca Uygunluk Değeri')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_best_model(self, save_dir='models'):
        """En iyi modeli kaydet"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if self.best_model:
            model_path = os.path.join(save_dir, f'best_model_gen_{self.generation}.pkl')
            self.best_model.save(model_path)
            return model_path
        
        return None 
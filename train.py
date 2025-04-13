import os
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from game import Bird, FlappyBird, GameMode
from neural_network import NeuralNetwork, EvolutionaryAlgorithm

def create_bird_with_brain(x, y, brain):
    """Sinir ağı ile kontrol edilen bir kuş oluşturur"""
    bird = Bird(x, y, brain=brain)
    return bird

def display_best_model(model, max_steps=None):
    """En iyi modeli görsel olarak gösterir"""
    # Pygame'i başlat
    import pygame
    
    # AI modunda oyunu başlat
    game = FlappyBird(mode=GameMode.AI)
    
    # Modeli kullanan bir kuş oluştur
    bird = create_bird_with_brain(100, 300, model)
    game.reset(birds=[bird])
    
    # Oyun döngüsü
    running = True
    step = 0
    
    while running:
        # Pygame olaylarını işle
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Oyunu bir adım ilerlet
        _, _, done = game.step()
        game.render()
        
        # Tüm kuşlar öldüyse döngüyü sonlandır
        if done:
            break
            
        step += 1
        
        # İsteğe bağlı maksimum adım kontrolü
        if max_steps is not None and step >= max_steps:
            break
    
    # Oyunu kapat
    game.close()

def train_headless(generations=100, population_size=100, max_steps=1000, 
                  save_interval=10, render_best=False, render_interval=10):
    """Headless modda AI eğitimi yapar (görselleştirme olmadan)"""
    # Evrimsel algoritmayı başlat
    evolution = EvolutionaryAlgorithm(
        population_size=population_size,
        input_size=8,
        hidden_size=24,
        output_size=1,
        mutation_rate=0.2,
        mutation_amount=0.3,
        survival_rate=0.3
    )
    
    # Çıktı dizinlerini oluştur
    if not os.path.exists('logs'):
        os.makedirs('logs')
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Headless modda oyunu başlat
    game = FlappyBird(mode=GameMode.HEADLESS)
    
    # Eğitim döngüsü
    start_time = time.time()
    best_score = 0
    
    # Nesiller için ilerleme çubuğu
    for generation in tqdm(range(generations), desc="Eğitim Nesilleri"):
        # Sinir ağı beyinleri ile kuşlar oluştur
        birds = []
        for i in range(population_size):
            brain = evolution.population[i]
            birds.append(create_bird_with_brain(100, 300, brain))
        
        # Oyunu yeni kuşlarla sıfırla
        game.reset(birds=birds)
        
        # Tüm kuşlar ölene veya maksimum adım sayısına ulaşılana kadar bu nesli çalıştır
        for step in range(max_steps):
            # Oyunu bir adım ilerlet
            _, _, done = game.step()
            
            # Tüm kuşlar öldüyse bu nesli sonlandır
            if done:
                break
        
        # En iyi skoru güncelle
        if game.score > best_score:
            best_score = game.score
            print(f"Yeni en iyi skor: {best_score} (Nesil {generation})")
        
        # Bir sonraki nesli oluştur
        evolution.create_next_generation(birds)
        
        # Periyodik olarak en iyi modeli kaydet
        if (generation + 1) % save_interval == 0:
            model_path = evolution.save_best_model()
            print(f"En iyi model kaydedildi: {model_path}")
            
            # Uygunluk grafiğini kaydet
            plot_path = os.path.join('plots', f'fitness_gen_{generation}.png')
            evolution.plot_fitness_history(save_path=plot_path)
        
        # İsteğe bağlı olarak periyodik olarak en iyi modeli görselleştir
        if render_best and (generation + 1) % render_interval == 0:
            display_best_model(evolution.best_model)
    
    # Eğitim tamamlandı
    total_time = time.time() - start_time
    print(f"Eğitim tamamlandı! Toplam süre: {total_time:.2f} saniye")
    print(f"En iyi skor: {best_score}")
    
    # Son uygunluk grafiğini kaydet
    plot_path = os.path.join('plots', f'fitness_final.png')
    evolution.plot_fitness_history(save_path=plot_path)
    
    # En son modeli kaydet
    final_model_path = evolution.save_best_model()
    print(f"Son model kaydedildi: {final_model_path}")
    
    return evolution.best_model

def run_visualized_training(generations=20, population_size=50, max_steps=1000):
    """Görselleştirme ile eğitim yapar"""
    # Görselleştirme için pygame'i içe aktar
    import pygame
    
    # Daha küçük bir popülasyon ile evrimsel algoritmayı başlat
    evolution = EvolutionaryAlgorithm(
        population_size=population_size,
        input_size=8,
        hidden_size=24,
        output_size=1,
        mutation_rate=0.3,
        mutation_amount=0.4,
        survival_rate=0.4
    )
    
    # AI modunda oyunu başlat (görünür)
    game = FlappyBird(mode=GameMode.AI)
    
    # Eğitim döngüsü
    for generation in range(generations):
        # Sinir ağı beyinleri ile kuşlar oluştur
        birds = []
        for i in range(population_size):
            brain = evolution.population[i]
            birds.append(create_bird_with_brain(100, 300, brain))
        
        # Oyunu yeni kuşlarla sıfırla
        game.reset(birds=birds)
        
        # Tüm kuşlar ölene veya maksimum adım sayısına ulaşılana kadar bu nesli çalıştır
        running = True
        step = 0
        
        while running and step < max_steps:
            # Pygame olaylarını işle
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return
            
            # Oyunu bir adım ilerlet
            _, _, done = game.step()
            game.render()
            
            # Tüm kuşlar öldüyse bu nesli sonlandır
            if done:
                break
                
            step += 1
        
        # Nesil bilgisini göster
        print(f"Nesil {generation} - En İyi Skor: {game.score}")
        
        # Bir sonraki nesli oluştur
        evolution.create_next_generation(birds)
    
    # Oyunu kapat
    game.close()
    
    return evolution.best_model

def play_human():
    """İnsan oyuncusu olarak oyunu oynat"""
    game = FlappyBird(mode=GameMode.HUMAN)
    game.run_human()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Flappy Bird AI Eğitici')
    parser.add_argument('--mode', type=str, default='train_headless', 
                        choices=['train_headless', 'train_visual', 'play_human', 'play_ai'],
                        help='Programın çalıştırılacağı mod')
    parser.add_argument('--generations', type=int, default=100, 
                        help='Eğitilecek nesil sayısı')
    parser.add_argument('--population', type=int, default=100, 
                        help='Popülasyon boyutu')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Her nesil için maksimum adım sayısı')
    parser.add_argument('--model', type=str, default=None,
                        help='Kullanılacak önceden eğitilmiş model yolu')
    
    args = parser.parse_args()
    
    if args.mode == 'train_headless':
        train_headless(generations=args.generations, population_size=args.population, max_steps=args.max_steps)
    elif args.mode == 'train_visual':
        run_visualized_training(generations=args.generations, population_size=args.population, max_steps=args.max_steps)
    elif args.mode == 'play_human':
        play_human()
    elif args.mode == 'play_ai':
        if args.model:
            model = NeuralNetwork.load(args.model)
            display_best_model(model)
        else:
            print("Lütfen --model ile bir model yolu belirtin")
    else:
        print("Geçersiz mod") 
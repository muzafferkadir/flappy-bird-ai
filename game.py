import pygame
import random
import os
import numpy as np
from enum import Enum

class GameMode(Enum):
    HUMAN = 1   # İnsan oyuncular için mod
    AI = 2      # AI eğitimi için görsel mod
    HEADLESS = 3  # Görsel olmayan hızlı eğitim modu

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
        
    def jump(self):
        # Kuşu zıplat (negatif hız uygula)
        self.velocity = self.jump_strength
    
    def apply_brain(self, pipes):
        # Eğer beyin yoksa veya boru yoksa işlem yapma
        if self.brain is None or not pipes:
            return
        
        # Bir sonraki boruyu bul
        next_pipe = None
        second_pipe = None
        
        # Sonraki iki boruyu bul
        for pipe in pipes:
            if pipe.x + pipe.width > self.x:
                if next_pipe is None:
                    next_pipe = pipe
                elif second_pipe is None:
                    second_pipe = pipe
                    break
        
        if next_pipe:
            # Daha gelişmiş girdileri hesapla
            # Bir sonraki boruya olan mesafeyi normalize et (yatay)
            dist_to_pipe_normalized = (next_pipe.x - self.x) / 400
            
            # Boşluğa göre dikey konumu hesapla (normalize edilmiş)
            gap_center_y = next_pipe.gap_y
            relative_height = (self.y - gap_center_y) / 200  # -1 ile 1 arasında ölçek
            
            # Üst borunun alt kenarı
            upper_pipe_bottom = next_pipe.gap_y - next_pipe.gap_height / 2
            
            # Alt borunun üst kenarı
            lower_pipe_top = next_pipe.gap_y + next_pipe.gap_height / 2
            
            # Üst boruya olan mesafe (normalize edilmiş)
            dist_to_upper = (self.y - upper_pipe_bottom) / 200  # Altındaysa negatif
            
            # Alt boruya olan mesafe (normalize edilmiş)
            dist_to_lower = (lower_pipe_top - self.y) / 200  # Üstündeyse negatif
            
            # Sinir ağı için girdiler
            inputs = [
                self.y / 400,  # Kuşun normalize edilmiş y konumu
                self.velocity / 10,  # Kuşun normalize edilmiş hızı
                dist_to_pipe_normalized,  # Boruya olan normalize edilmiş yatay mesafe
                relative_height,  # Boşluk merkezine göre göreceli yükseklik
                dist_to_upper,    # Üst boruya olan mesafe (altındaysa negatif)
                dist_to_lower,    # Alt boruya olan mesafe (üstündeyse negatif)
            ]
            
            # İkinci boru varsa ek girdiler
            if second_pipe:
                second_dist = (second_pipe.x - self.x) / 400
                second_relative_height = (self.y - second_pipe.gap_y) / 200
                
                inputs.append(second_dist)
                inputs.append(second_relative_height)
            else:
                # İkinci boru yoksa varsayılan değerler ekle
                inputs.append(1.0)  # Uzak bir mesafe
                inputs.append(0.0)  # Nötr yükseklik
            
            # Sinir ağından karar al (tüm girdileri kullanarak)
            if self.brain.predict(inputs) > 0.5:  # Tüm girdileri kullan
                self.jump()
    
    def update(self):
        # Yerçekimi etkisi ve konum güncelleme
        self.velocity += self.gravity
        self.y += self.velocity
        
        # Kuşun ekranın üstünden çıkmasını engelle
        if self.y < 0:
            self.y = 0
            self.velocity = 0
        
        return self.y < 400  # Ekran sınırları içinde olup olmadığını döndür
    
    def collides_with(self, pipe):
        # Üst boru ile çarpışma kontrolü
        if (self.x + self.width > pipe.x and self.x < pipe.x + pipe.width and 
            self.y < pipe.gap_y - pipe.gap_height // 2):
            return True
        
        # Alt boru ile çarpışma kontrolü
        if (self.x + self.width > pipe.x and self.x < pipe.x + pipe.width and 
            self.y + self.height > pipe.gap_y + pipe.gap_height // 2):
            return True
        
        return False

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 60
        self.speed = 3  # Daha yavaş - kuşların tepki vermesi için daha fazla zaman
        self.gap_height = 170  # Daha geniş boşluk - başlangıçta geçmeyi kolaylaştırır
        self.gap_y = random.randint(180, 320)  # Daha dar yükseklik aralığı - uç pozisyonları azaltır
        self.passed = False
    
    def update(self):
        # Boruyu sola doğru hareket ettir
        self.x -= self.speed
        return self.x > -self.width  # Boru hala ekranda mı?
    
    def is_passed_by(self, bird):
        # Kuş boruyu geçti mi kontrol et
        if not self.passed and bird.x > self.x + self.width:
            self.passed = True
            return True
        return False

class FlappyBird:
    def __init__(self, mode=GameMode.HUMAN, width=800, height=600, fps=60):
        self.width = width
        self.height = height
        self.ground_y = height - 100
        self.fps = fps
        self.mode = mode
        
        # Initialize Pygame if not in headless mode
        if mode != GameMode.HEADLESS:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Flappy Bird AI")
            self.clock = pygame.time.Clock()
            
            # Load images
            self.bird_img = pygame.Surface((34, 24))
            self.bird_img.fill((255, 255, 0))  # Yellow bird
            self.pipe_img = pygame.Surface((60, 500))
            self.pipe_img.fill((0, 255, 0))  # Green pipe
            self.bg_color = (135, 206, 235)  # Sky blue
            self.font = pygame.font.SysFont(None, 36)
        
        self.reset()
    
    def reset(self, birds=None):
        # Game state
        self.game_over = False
        self.score = 0
        self.frame_count = 0
        
        # Initialize birds
        if birds:
            self.birds = birds
        else:
            # Initialize one bird at the center of the screen
            self.birds = [Bird(self.width // 4, self.height // 2)]
        
        # Initialize pipes
        self.pipes = []
        self.add_pipe()
        
        # State for AI
        self.living_birds = len(self.birds)
        
        return self.get_state()
    
    def add_pipe(self):
        self.pipes.append(Pipe(self.width))
    
    def get_state(self):
        """Get the current game state for AI"""
        if not self.birds or not self.pipes:
            return None
        
        bird = self.birds[0]  # Use the first bird for state
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > bird.x:
                next_pipe = pipe
                break
        
        if next_pipe:
            return {
                'bird_y': bird.y,
                'bird_velocity': bird.velocity,
                'next_pipe_x': next_pipe.x,
                'next_pipe_gap_y': next_pipe.gap_y
            }
        return None
    
    def step(self, action=None):
        """Step the game forward one frame, return (state, reward, done)"""
        self.frame_count += 1
        reward = 0.1  # Small reward for surviving each frame
        
        # Add new pipes
        if self.frame_count % 60 == 0:
            self.add_pipe()
        
        # Process action for human mode
        if self.mode == GameMode.HUMAN and action:
            self.birds[0].jump()
        
        # Update pipes
        temp_pipes = []
        for pipe in self.pipes:
            if pipe.update():  # Returns False if pipe is off-screen
                temp_pipes.append(pipe)
        self.pipes = temp_pipes
        
        # Update birds and check collisions
        for bird in self.birds:
            if not bird.alive:
                continue
                
            # Apply AI brain if in AI mode
            if self.mode in (GameMode.AI, GameMode.HEADLESS):
                bird.apply_brain(self.pipes)
            
            # Update bird position
            if not bird.update():  # Bird hit the ground
                bird.alive = False
                self.living_birds -= 1
            
            # Check pipe collisions
            for pipe in self.pipes:
                if bird.collides_with(pipe):
                    bird.alive = False
                    self.living_birds -= 1
                    reward = -1  # Negative reward for collision
                    break
                    
                # Score for passing pipes
                if pipe.is_passed_by(bird):
                    bird.score += 1
                    self.score = max(self.score, bird.score)
                    reward = 1  # Positive reward for passing a pipe
        
        # Check if game is over (all birds dead)
        done = self.living_birds <= 0
        self.game_over = done
        
        return self.get_state(), reward, done
    
    def render(self):
        """Render the game state"""
        if self.mode == GameMode.HEADLESS:
            return
        
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Draw pipes
        for pipe in self.pipes:
            # Upper pipe
            upper_pipe = self.pipe_img.copy()
            upper_pipe_height = pipe.gap_y - pipe.gap_height // 2
            upper_pipe = pygame.transform.scale(upper_pipe, (pipe.width, upper_pipe_height))
            self.screen.blit(upper_pipe, (pipe.x, 0))
            
            # Lower pipe
            lower_pipe = self.pipe_img.copy()
            lower_pipe_y = pipe.gap_y + pipe.gap_height // 2
            lower_pipe_height = self.ground_y - lower_pipe_y
            lower_pipe = pygame.transform.scale(lower_pipe, (pipe.width, lower_pipe_height))
            self.screen.blit(lower_pipe, (pipe.x, lower_pipe_y))
            
        # Draw birds
        for i, bird in enumerate(self.birds):
            if bird.alive:
                # Draw bird as a yellow rectangle
                pygame.draw.rect(self.screen, bird.color, 
                               (bird.x, bird.y, bird.width, bird.height))
                
                # Draw bird number in the center of the bird rectangle
                number_text = self.font.render(str(i+1), True, (0, 0, 0))
                text_rect = number_text.get_rect(center=(bird.x + bird.width//2, bird.y + bird.height//2))
                self.screen.blit(number_text, text_rect)
        
        # Draw ground
        pygame.draw.rect(self.screen, (139, 69, 19), 
                        (0, self.ground_y, self.width, self.height - self.ground_y))
        
        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))

        # Draw alive birds count
        birds_text = self.font.render(f"Birds: {self.living_birds}", True, (255, 255, 255))
        self.screen.blit(birds_text, (10, 50))
        
        # Eğitim modu için ek bilgiler
        if self.mode == GameMode.AI:
            # Çerçeve sayısı
            frame_text = self.font.render(f"Frame: {self.frame_count}", True, (255, 255, 255))
            self.screen.blit(frame_text, (10, 90))
            
            # En yakın borunun konumu
            next_pipe = None
            for pipe in self.pipes:
                if pipe.x + pipe.width > 100:  # Kuşun x konumu
                    next_pipe = pipe
                    break
            
            if next_pipe:
                pipe_text = self.font.render(f"Next Pipe: {int(next_pipe.x)}, Gap: {int(next_pipe.gap_y)}", True, (255, 255, 255))
                self.screen.blit(pipe_text, (10, 130))
        
        if self.game_over:
            game_over_text = self.font.render('Game Over! Press R to restart', True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def run_human(self):
        """Run the game for human play"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.step(action=1)  # Jump
                    if event.key == pygame.K_r and self.game_over:
                        self.reset()
            
            if not self.game_over:
                self.step()
                self.render()
        
        pygame.quit()
    
    def close(self):
        if self.mode != GameMode.HEADLESS:
            pygame.quit()

# Run the game in human mode if this file is executed directly
if __name__ == "__main__":
    game = FlappyBird(mode=GameMode.HUMAN)
    game.run_human() 
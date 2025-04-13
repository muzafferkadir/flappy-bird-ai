# Flappy Bird AI - Learning with Evolutionary Algorithm

This project implements the Flappy Bird game using Python and Pygame, demonstrating artificial intelligence training through evolutionary algorithms. Using evolutionary learning methods, birds learn to play the game through generations of development.

## Features

- Simple and optimized Flappy Bird game engine
- Playable mode for human players
- Evolutionary algorithm implementation for training
- Headless (non-visual) or visual training modes
- Training performance tracking and graphs
- Saving and loading trained models

## Installation

1. Install Python 3.6+
2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

## Usage

The project can be run in the following modes:

### Play as Human

```bash
python train.py --mode play_human
```

### AI Training in Headless Mode

```bash
python train.py --mode train_headless --generations 100 --population 100
```

### AI Training in Visual Mode (with fewer birds)

```bash
python train.py --mode train_visual --generations 20 --population 50
```

### Play with Trained AI

```bash
python train.py --mode play_ai --model models/best_model_gen_99.pkl
```

## Evolutionary Algorithm Explanation

This project implements an evolutionary approach using the following main components:

1. **Population**: 100 birds (individuals) are created simultaneously in each generation
2. **Neural Network**: Each bird is controlled by a simple neural network with 4 inputs (bird position, velocity, pipe positions) and 8 hidden neurons
3. **Fitness Function**: Measured based on the bird's score and survival time
4. **Selection**: Individuals with the highest fitness values are selected as parents for the next generation
5. **Mutation**: Neural network weights of birds in the new generation are randomly mutated

## Project Structure

- `game.py`: Flappy Bird game engine
- `neural_network.py`: Neural network model and evolutionary algorithm implementation
- `train.py`: Main file containing training and game modes
- `requirements.txt`: Required Python libraries
- `models/`: Directory for saving trained models
- `plots/`: Directory for saving training statistics graphs

## Performance Improvements

- Headless mode provides fast training without visualization
- Optimized collision detection and game physics
- Progress bar and detailed training logs
- Preservation of the best model between generations 
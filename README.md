# Pacman Reinforcement Learning

Final project for CMPE 260. Implementing and comparing various Reinforcement Learning algorithms in the Pacman game environment.

## Project Overview

This project implements and compares different Deep Q-Network (DQN) variants for training an AI agent to play Pacman. The implementation includes:

- Basic DQN with strategic network architecture
- Sophisticated DQN with enhanced action space and advanced features
- Evaluation and comparison framework for analyzing agent performance

## Agents

### Basic DQN Agent

- Implements standard DQN architecture with strategic network layers
- Uses basic action space (UP, RIGHT, DOWN, LEFT)
- Features batch normalization and reward shaping
- Located in `DQN_agent.py`

### Sophisticated DQN Agent

- Enhanced action space with diagonal and special movements
- Advanced reward shaping and strategic decision making
- Complex network architecture with additional features
- Located in `sophisticated_DQN_agent.py`

## Evaluation Framework

The project includes a comprehensive evaluation framework (`evaluation.py`) that enables:

### Features

- Comparative analysis between DQN variants
- Performance metrics tracking:
  - Win rate
  - Completion rate
  - Average score
  - Steps per episode
  - Total rewards
- Automated visualization generation
- Episode replay saving as GIFs

### Usage

To evaluate and compare agents:

```python
python evaluation.py
```

This will:

1. Run evaluation episodes for both agents
2. Generate comparison plots
3. Save results to `evaluation_results.png`
4. Create episode replay GIFs in `episode_gifs/`

## Project Structure

```
.
├── DQN_agent.py              # Basic DQN implementation
├── sophisticated_DQN_agent.py # Enhanced DQN implementation
├── pacman_env.py             # Basic Pacman environment
├── pacman_wrapper.py         # Enhanced environment wrapper
├── evaluation.py             # Evaluation and comparison framework
└── episode_gifs/             # Directory for episode replays
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- OpenCV (for environment rendering)

## Results

The evaluation framework generates comprehensive comparisons between the basic DQN and sophisticated DQN implementations, measuring:

- Learning efficiency
- Game performance
- Strategic behavior
- Completion rates

Results are automatically visualized and saved as `evaluation_results.png`.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

import torch as th
import gymnasium as gym
from gymnasium import spaces

class TokenGenerationEnv(gym.Env):

    def __init__(self, discriminator, vocab_size, seq_length, start_token, device):
        super(TokenGenerationEnv, self).__init__()
        
        self.discriminator = discriminator
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.start_token = start_token
        self.device = device
        
        self.action_space = spaces.Discrete(vocab_size)             # Action space is the vocabulary size
        self.observation_space = spaces.Discrete(vocab_size)        # Observation space is the same as action space
        
        self.current_sequence = None
        self.current_position = None
    
    def reset(self, seed = None, options = None):
        
        # Initialize with start token
        self.current_sequence = [self.start_token]
        self.current_position = 1  # Position after start token
        
        return int(self.start_token), {}
    
    def step(self, action):

        # Add the selected token to the sequence
        self.current_sequence.append(int(action))
        self.current_position += 1
        
        # Check if the sequence is complete
        done = (self.current_position >= self.seq_length)
        
        # Calculate reward
        reward = 0.0

        if done:
            reward = self._get_reward(self.current_sequence)
            
        # Return observation, reward, done flag, truncated and info
        return action, reward, done, False, {}
    
    def _get_reward(self, sequence):
        
        sequence_tensor = th.tensor([sequence], dtype=th.long, device=self.device)
        reward = float(self.discriminator.get_reward(sequence_tensor).cpu())

        return reward


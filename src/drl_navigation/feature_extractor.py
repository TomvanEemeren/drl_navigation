import gymnasium as gym
import torch as th
from torch import nn
import rospy

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "costmap":
                n_convW1 = ((subspace.shape[1] - 6) // 3 + 1)
                n_convH1 = ((subspace.shape[2] - 6) // 3 + 1)
                n_poolW1 = (n_convW1 - 3) // 3 + 1
                n_poolH1 = (n_convH1 - 3) // 3 + 1
                n_flatten = n_poolW1 * n_poolH1 * 6
                extractors[key] = nn.Sequential(
                    nn.Conv2d(subspace.shape[0], 6, kernel_size=7, stride=3, padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=3),
                    nn.Flatten(),
                )

                total_concat_size += n_flatten

            else:
                extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size
        rospy.logwarn(f"Features dim: {self._features_dim}")

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
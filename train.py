import argparse

import torch

from src.engine import Engine

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default='resnet18', help="Config name")
	args = parser.parse_args()

	engine = Engine(args.config)
	engine.train()
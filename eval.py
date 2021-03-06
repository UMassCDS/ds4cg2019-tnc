import argparse
import torch

from src.engine import Engine

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='resnet18',
                        help="Config name")
    parser.add_argument('--tag', default='',
                        help="tag to discern evaluation instances")
    args = parser.parse_args()

    engine = Engine(mode='eval', config_name=args.config, tag=args.tag)
    engine.evaluate()

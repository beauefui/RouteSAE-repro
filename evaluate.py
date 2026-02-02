# evaluate.py
"""评估入口"""

from src.utils import parse_args
from src.evaluator import Evaluator


if __name__ == '__main__':
    cfg = parse_args()
    
    evaluator = Evaluator(cfg)
    result = evaluator.run()
    
    print(f"Evaluation complete. {cfg.metric}: {result:.4f}")
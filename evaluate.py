# evaluate.py
"""评估入口"""

from src.utils import parse_args, set_seed
from src.evaluator import Evaluator


if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    
    evaluator = Evaluator(cfg)
    result = evaluator.run()
    
    print(f"Evaluation complete. {cfg.metric}: {result:.4f}")
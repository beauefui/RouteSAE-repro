# train.py
"""训练入口"""

from src.utils import parse_args, set_seed
from src.trainer import Trainer


if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    
    trainer = Trainer(cfg)
    final_loss = trainer.run()
    
    print(f"Training complete. Final loss: {final_loss:.4f}")
# interpret.py
"""Interpretation script entry point"""

from src.utils import parse_args, set_seed
from src.interpreter import Interpreter

if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    
    interpreter = Interpreter(cfg)
    
    # 运行解释
    # data_path 默认为 None (使用 cfg.data_path)
    # output_path 默认为 None (自动生成)
    interpreter.run(
        sample_latents=100 # 默认采样 100 个特征进行解释
    )

# apply.py
"""Apply scripts entry point"""

from src.utils import parse_args, set_seed
from src.applier import Applier

if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    
    applier = Applier(cfg)
    
    # 提取上下文
    # 参数可以通过命令行传递吗？
    # parse_args 中没有 max_per_token 等参数
    # 我们可以复用 parse_args 中的参数或者添加新的
    # 为了简单，我们这里使用硬编码或添加参数到 utils.parse_args
    # 但 utils.parse_args 在 src/utils.py 中，修改它可能影响 train.py
    # 作为一个 helper script，我们可以直接在这里调用，或者扩展 parse_args
    
    # 不过为了方便，我们假设用户通过 standard args 传入了基本配置
    # 额外的参数可以在这里定义 defaults
    
    applier.get_context(
        threshold=10.0,      # 默认阈值
        max_length=64,       # 上下文长度
        max_per_token=5,     # 每个 token 多样性
        lines=3,             # 最小样本数
        output_path=None     # 自动生成或通过 args 传入(如果 utils 支持)
    )

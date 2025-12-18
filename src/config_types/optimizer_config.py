from .adamw_config import AdamWConfig
from .adagrad_config import AdagradConfig
from .ftrl_config import FTRLConfig

# Union type for any optimizer config
OptimizerConfig = AdamWConfig | AdagradConfig | FTRLConfig

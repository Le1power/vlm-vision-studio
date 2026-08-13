"""pytest 公共配置：保证从项目根目录可导入 src 包。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

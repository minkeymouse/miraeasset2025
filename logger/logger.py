import os
import re
import time
import tempfile
from pathlib import Path
import nest_asyncio
from dotenv import load_dotenv
from typing import Dict, Optional, Union, List
import logging
from datetime import datetime
from tqdm import tqdm

# 환경변수 및 기본 설정
load_dotenv()
nest_asyncio.apply()

# 기본 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# LlamaParse의 HTTP 로그 숨기기
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("llama_parse").setLevel(logging.WARNING)

# 메인 로거 설정
logger = logging.getLogger(__name__)

class StepLogger:
    """단계별 로깅을 위한 유틸리티 클래스"""
    _step = 1
    
    @classmethod
    def log_step(cls, message: str) -> None:
        """단계별 로그 출력"""
        logger.info(f"[Step {cls._step}] {message}")
        cls._step += 1
    
    @classmethod
    def reset_step(cls) -> None:
        """스텝 카운터 초기화"""
        cls._step = 1
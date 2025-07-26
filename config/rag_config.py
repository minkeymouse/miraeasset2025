from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass
class RAGConfig:
    # 이미지로 구성된 애널리스트 분석보고서 pdf 파일의 OCR 파싱을 위한 무료 Gemini API key 목록(일일 허용량까지 순차적으로 활용)
    google_api_keys: List[str] = field(default_factory=lambda: [
        'AIzaSyAsiUeeZJmKBa-by6r5wWlJNDo_8HC4Fgs',
        'AIzaSyCyFXmSB0iEaTgpxUR57ifBBdj1WMSCaaM',
        'AIzaSyBPyf2TTqHDAVJe9w0BeBwIkaATSJjj4_U',
        'AIzaSyA1Lvw5uW3OWvZYMOj3iDtEyZHgoGgVwsM',
        'AIzaSyBfOucWyehvgJ5L2vapjtBU_lATYCqPB1I',
        'AIzaSyCY05E7Gk1l9quSm5gNZhAToXjsffqeDyQ',
        'AIzaSyDfQLq5R-otVlC4B0wxw6IEAfswzninG0M',
        'AIzaSyB1L50EJVn0-1v5HAbbr-di6ux6N75QWR8',
        'AIzaSyBCAFuCle-ih6Bzlkt1qVAtGE7sIEBpeUw',
        'AIzaSyApGXUrF0y91gLZy_mLUyUA81IQeaNnJE8',
        'AIzaSyCtyDT87rQBR_oPZ2HpDezPAcedzF4gT_8',
        'AIzaSyCjkomKJp3pHBSY1IeUPFxIGhV_VWGKBtk'
    ])

class RAGPipelineError(Exception):#
    """RAG 파이프라인 관련 기본 예외"""
    pass

class GroundednessCheckError(RAGPipelineError):
    """Groundedness 체크 중 발생하는 예외"""
    pass

class DocumentNotFoundError(RAGPipelineError):
    """문서를 찾을 수 없을 때 발생하는 예외"""
    pass
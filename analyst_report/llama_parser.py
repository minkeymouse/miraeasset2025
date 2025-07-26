import os
import re
import time
import json
import logging
import tempfile
import traceback
import nest_asyncio
import pdfplumber
import warnings
import logging
import pymupdf
from pykrx import stock
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional, Union, List, Any
from miraeasset2025.utils.utils import standardize_date
from llama_parse import LlamaParse
from pypdf import PdfReader, PdfWriter
from datetime import datetime
from tqdm import tqdm
from miraeasset2025.logger.logger import logger, StepLogger
from langchain_core.prompts import PromptTemplate
from langchain_naver import ChatClovaX
from miraeasset2025.config.rag_config import (
    RAGConfig,
    RAGPipelineError,
    GroundednessCheckError,
    DocumentNotFoundError
)


logging.getLogger('fitz').setLevel(logging.ERROR)  # WARNING 이하 레벨 무시

# 환경변수 및 기본 설정
load_dotenv()
nest_asyncio.apply()

class ParsingError(Exception):
    """파싱 관련 커스텀 예외"""
    pass

class LlamaParser:
    _tickers = None 

    @classmethod
    def _initialize_tickers(cls):
        """tickers를 한 번만 초기화하는 클래스 메서드"""
        if cls._tickers is None:
            logger.info("KOSPI 종목 데이터 호출 시작")
            cls._tickers = stock.get_market_ticker_list(market="ALL")
            logger.info("KOSPI 종목 데이터 호출 완료")
        return cls._tickers

    def __init__(self, pdf_path: str):
        """
        PDF 리포트 분석을 위한 클래스 초기화
        
        Args:
            pdf_path (str): PDF 파일 경로
            
        Raises:
            FileNotFoundError: PDF 파일이 존재하지 않을 경우
            ParsingError: 파싱 과정에서 오류가 발생한 경우
        """
        StepLogger.log_step("초기화 시작")
        self.config = RAGConfig()
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # 클래스 속성 초기화
        self.ref_date: Optional[str] = None
        self.ticker: Optional[str] = None
        self.name: Optional[str] = None
        self.firm: Optional[str] = None
        self.paragraph: Optional[str] = None
        self.title: Optional[str] = None
        self.table: Dict[str, pd.DataFrame] = {}  # 테이블 저장용 딕셔너리
        self.raw_paragraph : Optional[str] = None
        self.raw_metadata = None
        self.temp_dict = None
        self.tickers = self._initialize_tickers()
        self.additional_metadata = {}

    def _wait_for_result(self, parsed_docs: List, desc: str = "파싱 진행 중") -> Optional[Dict]:
        """
        파싱 결과를 기다리면서 진행바 표시
        
        Args:
            parsed_docs: 파싱된 문서 리스트
            desc: 진행바 설명
            
        Returns:
            Optional[Dict]: 파싱 결과
        """
        if not parsed_docs:
            logger.warning("No documents were parsed")
            return None
            
        with tqdm(total=100, desc=desc) as pbar:
            for i in range(33):  # 3초 동안 진행바 표시
                pbar.update(3)
                time.sleep(0.1)
            pbar.update(1)  # 마지막 1% 업데이트
        
        return parsed_docs[0]

    def _validate_content(self, text: str, min_length: int = 500) -> bool:
        """텍스트 유효성 검증"""
        StepLogger.log_step("텍스트 유효성 검증 시작")
        
        if not text or len(text.strip()) < min_length:
            logger.warning("의미 있는 컨텐츠가 부족하거나 이미지로 구성된 pdf입니다. LlamaParser를 호출합니다.")
            return False
        
        meaningful_content = re.sub(r'[\s\d\W]+', '', text)
        if len(meaningful_content) < min_length // 10:
            logger.warning("의미 있는 컨텐츠가 부족하거나 이미지로 구성된 pdf입니다. LlamaParser를 호출합니다.")
            return False
        
        StepLogger.log_step("텍스트 유효성 검증 완료")
        return True

    def _parse_first_page_only(self) -> Optional[Dict]:
        """첫 페이지 메타데이터 파싱"""
        StepLogger.log_step("첫 페이지 파싱 시작")
        
        try:
            pdf = PdfReader(str(self.pdf_path))
            if len(pdf.pages) == 0:
                raise ParsingError("PDF file is empty")
            
            writer = PdfWriter()
            writer.add_page(pdf.pages[0])
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                writer.write(temp_pdf)
                temp_pdf_path = temp_pdf.name
            
            try:
                parser = LlamaParse(
                    use_vendor_multimodal_model=True,
                    vendor_multimodal_model_name="gemini-2.0-flash",
                    vendor_multimodal_api_key=self.config.google_api_keys,
                    result_type="markdown",
                    language="ko",
                    parsing_instruction="""
                    당신은 PDF 메타데이터 정제 전문가입니다. 주어진 텍스트를 아래 규칙에 따라 메타데이터를 정제해주세요.

                    규칙:               
                    1. 주어진 텍스트의 메타데이터를 반드시 아래와 같은 형식으로 한글로만 추출해주세요.
                    2. 영어 텍스트도 반드시 한글로 번역해서 추출해주세요.
                    3. 항목이 존재하지 않는다면 반드시 None으로 출력해주세요.
                    4. **두 가지 이상 종목명이나 종목코드가 존재한다면 반드시 None으로 출력해주세요.** 
                    5. 반드시 JSON 형식으로만 응답해주세요.
                    6. 다른 설명이나 부가적인 텍스트 없이 순수 JSON만 응답해주세요.

                    {{"제목": "문서 제목",
                    "증권사명": "증권사 이름",
                    "종목명": "종목 이름",
                    "종목코드": "종목 코드",
                    "작성일자": "작성 날짜"}}
                    """

                )
                
                # 파일 파싱 및 진행상황 표시
                parsed_docs = parser.load_data(file_path=temp_pdf_path)
                response = self._wait_for_result(parsed_docs, "메타데이터 파싱 중")
                
                # 응답 로깅
                logger.debug(f"LLM 응답: {response.text if response else 'No response'}")
                
                if not response or not response.text:
                    raise ParsingError("LLM이 메타데이터를 추출하지 못했습니다")

                # 응답 로깅 및 기본 검증
                response_text = response.text.strip()
                logger.debug(f"LLM 응답: {response_text}")

                if not response_text or not ('{' in response_text and '}' in response_text):
                    raise ParsingError("유효하지 않은 응답 형식입니다")
                
                # JSON 파싱
                try:
                    # 응답에서 JSON 부분만 추출
                    response_text = response.text
                    # JSON 형식의 텍스트만 추출하기 위한 처리
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response_text[json_start:json_end]
                        data = json.loads(json_str)
                        data = {k: self._clean_value(v) for k, v in data.items()}
                    else:
                        raise ParsingError("응답에서 JSON 형식을 찾을 수 없습니다")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 실패. 응답 내용: {response_text}")
                    raise ParsingError(f"JSON 파싱 실패: {str(e)}")
                
                # 필수 필드 검증
                required_fields = {'제목', '증권사명', '종목명', '종목코드', '작성일자'}
                missing_fields = required_fields - set(data.keys())
                if missing_fields:
                    raise ParsingError(f"필수 메타데이터 누락: {', '.join(missing_fields)}")
                
                logger.debug(f"메타데이터 파싱 완료: {data}")
                return data
                
            finally:
                try:
                    os.unlink(temp_pdf_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error in parsing first page: {str(e)}")
            raise ParsingError(f"Failed to parse first page: {str(e)}")

    def _map_parsed_data_llama(self) -> None:
        """메타데이터 매핑"""
        StepLogger.log_step("파싱 데이터 매핑 시작")

        try:
            parsed_data = self._parse_first_page_only()
            if not parsed_data:
                raise ParsingError("No data was parsed from the PDF")
            
            # 데이터 매핑
            self.title = parsed_data.get('제목')
            self.firm = parsed_data.get('증권사명')
            self.name = parsed_data.get('종목명')
            self.ticker = parsed_data.get('종목코드')
            self.ref_date = standardize_date(parsed_data.get('작성일자'))

            if (not self.name or self.name=='None' or self.name=='') and (not self.ticker or self.ticker=="None" or self.ticker==""):
                raise ParsingError("파싱된 데이터에 누락이 발생했거나 존재하지 않습니다.")

            # 이름과 티커 정규화 함수
            def normalize_name(name):
                if not name:
                    return ""
                # 특수문자, 공백 제거 및 소문자 변환
                name = re.sub(r'[^\w\s가-힣]', '', name)
                name = name.replace(' ', '').lower()
                return name

            # 현재 있는 티커로 검증
            if self.ticker and self.ticker != "None":
                try:
                    db_name = stock.get_market_ticker_name(self.ticker)
                    if normalize_name(db_name) != normalize_name(self.name):
                        logger.warning(f"티커({self.ticker})에 해당하는 기업명({db_name})이 현재 기업명({self.name})과 다릅니다.")
                        self.name = db_name  # DB의 이름으로 업데이트
                except Exception as e:
                    logger.warning(f"티커 {self.ticker}로 기업명 조회 실패: {str(e)}")
                    self.ticker = "None"  # 잘못된 티커인 경우 초기화

            # 이름으로 티커 검색 (티커가 없거나 "None"인 경우)
            if not self.name:
                logger.warning("기업명이 없습니다.")
            elif not self.ticker or self.ticker == "None":
                logger.info(f"기업명으로 티커 검색 시작: name={self.name}")
                normalized_input_name = normalize_name(self.name)
                
                for ticker in self.tickers:
                    try:
                        market_name = stock.get_market_ticker_name(ticker)
                        normalized_market_name = normalize_name(market_name)
                        
                        logger.debug(f"비교: DB={normalized_market_name} vs Input={normalized_input_name}")
                        
                        if normalized_market_name == normalized_input_name:
                            self.ticker = ticker
                            self.name = market_name  # DB의 공식 이름으로 업데이트
                            logger.info(f"일치하는 티커 발견: {ticker} ({market_name})")
                            break
                    except Exception as e:
                        logger.debug(f"티커 {ticker} 조회 실패: {str(e)}")
                        continue
                
                if not self.ticker or self.ticker == "None":
                    logger.warning(f"'{self.name}' 에 대한 티커를 찾지 못했습니다.")
                
        except Exception as e:
            logger.error(f"Error in mapping parsed data: {str(e)}")
            raise ParsingError(f"Failed to map parsed data: {str(e)}")

    def _extract_main_content(self) -> str:
        """본문 내용 추출"""
        StepLogger.log_step("본문 데이터 파싱 시작")
        
        try:
            pymupdf_text = []
            temp_dict = {}
            pymupdf.TOOLS.mupdf_display_errors(False)
            with pymupdf.open(str(self.pdf_path)) as pdf:
                # 전체 텍스트를 한번에 모으기
                for i in range(pdf.page_count):
                    page = pdf.load_page(i)
                    text = page.get_text('text').strip()  # 연속된 strip() 불필요
                    pymupdf_text.extend(text)
                
                temp_text = ''.join(pymupdf_text)

                # ticker 찾기
                for ticker in self.tickers:
                    # 괄호 있는 패턴 먼저 찾기
                    pattern_in_brackets = r'\(\s*' + re.escape(ticker) + r'\s*\)'
                    match = re.search(pattern_in_brackets, temp_text)

                    # 괄호 없는 패턴은 괄호 있는 패턴이 없을 때만 찾기
                    if match is None:  # == None 대신 is None 사용
                        pattern_without_brackets = r'\b' + re.escape(ticker) + r'\b'
                        match = re.search(pattern_without_brackets, temp_text)

                        if match:
                            ticker = match.group()
                            temp_dict['ticker'] = ticker
                            temp_dict['name'] = stock.get_market_ticker_name(ticker)
                            break
                    else :
                        temp_dict['ticker'] = None
                        temp_dict['name'] = None                          
                
                # 메타데이터 처리
                self.raw_metadata = pdf.metadata
                title = self.raw_metadata.get('title', '')
                temp_dict['title'] = title
                
                # 증권사 찾기
                temp_title = re.findall(r'(.*?증권)', title)
                if temp_title:
                    temp_dict['firm'] = temp_title[0]
                
                # 날짜 처리
                temp_time = self.raw_metadata.get('creationDate', '')
                match = re.search(r"D:(\d{4})(\d{2})(\d{2})", temp_time)
                if match:
                    year, month, day = match.groups()
                    publish_date = f"{year}-{month}-{day}"
                    temp_dict["ref_date"] = publish_date
                
            self.temp_dict = temp_dict
            
            StepLogger.log_step("본문 파싱 완료")
            return temp_text
            
        except Exception as e:
            logger.error(f"본문 파싱 중 오류 발생: {str(e)}")
            raise ParsingError(f"본문 파싱 실패: {str(e)}")
    
    def analyze(self, min_length: int = 500) -> None:
        """PDF 파일 분석 실행"""
        StepLogger.log_step("PDF 전체 분석 시작")
        
        try:
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {self.pdf_path}")
            
            if self.pdf_path.stat().st_size == 0:
                raise ValueError("PDF 파일이 비어있습니다.")
            
            # 본문 추출 및 검증
            extracted_text = self._extract_main_content()

            if not extracted_text:
                raise ValueError("텍스트 추출에 실패했습니다.")
            
            # pymupdf로 읽어오기에 실패한 경우, 이미지 pdf 파일로 인식하여 OCR parser(Llama parser)로 전환합니다.
            if not self._validate_content(extracted_text, min_length):
                StepLogger.log_step("LlamaParser 파싱 시작")

                try:
                    StepLogger.log_step("메타데이터 파싱 시작")
                    self._map_parsed_data_llama()
                    StepLogger.log_step("초기화 완료")

                except Exception as e:
                    logger.error(f"Initialization failed: {str(e)}")
                    raise ParsingError(f"Failed to initialize parser: {str(e)}")            
        
                try:
                    parser = LlamaParse(
                        use_vendor_multimodal_model=True,
                        vendor_multimodal_model_name="gemini-2.0-flash",
                        vendor_multimodal_api_key=self.config.google_api_keys,
                        result_type="markdown",
                        language="ko"
                    )
                    
                    # 파일 파싱 및 진행상황 표시
                    parsed_docs = parser.load_data(file_path=str(self.pdf_path))
                    parsing_result = self._wait_for_result(parsed_docs, "본문 파싱 중")
                    
                    if not parsing_result:
                        return None
                        
                    StepLogger.log_step("파싱된 텍스트 처리 중")
                    
                    # 모든 페이지의 텍스트를 리스트로 추출
                    texts = [doc.text for doc in parsed_docs if doc.text]
                    combined_text = '\n\n'.join(texts)
                    
                    StepLogger.log_step("LlamaParser 파싱 완료")

                except Exception as e:
                    logger.error(f"LlamaParser 파싱 중 오류 발생: {str(e)}")
                    raise ParsingError(f"LlamaParser 파싱 실패: {str(e)}")  

                self.raw_paragraph = combined_text
                self.paragraph = combined_text
                extracted_text = combined_text

            else:
                self.raw_paragraph = extracted_text
                self._map_parsed_data()
                self.paragraph = extracted_text
            
            # 본문 줄글 텍스트 분리
            self._parse_text()
            
            if (self.paragraph is None or 
                self.paragraph == 'NO_CONTENT_HERE' or
                len(self.paragraph) < 20 or 
                not self.paragraph.strip()):  # 빈 문자열이나 공백만 있는 경우도 체크
                self.paragraph = '보안처리된 분석 보고서입니다.'
                print('해당 보고서는 보안처리가 되어있어 본문 추출이 불가능합니다.')

            # 결과 출력
            StepLogger.log_step("분석 결과 요약")
            print(f"\nPDF 분석 완료:")
            print(f"- 파일명: {self.pdf_path.name}")
            print(f"- 추출된 텍스트 길이: {len(extracted_text):,}자")
            print(f"\n분석된 내용:")
            print(f"- 종목명: {self.name}")
            print(f"- 종목코드: {self.ticker}")
            print(f"- 작성날짜: {self.ref_date}")
            print(f"- 증권사: {self.firm}")
            StepLogger.log_step("전체 분석 완료")
            
        except Exception as e:
            logger.error(f"PDF 분석 중 오류 발생: {str(e)}")
            raise

    def get_parsed_data(self) -> Dict[str, str]:
        """파싱된 데이터 반환"""
        return {
            'title': self.title,
            'firm': self.firm,
            'name': self.name,
            'ticker': self.ticker,
            'ref_date': self.ref_date,
            'paragraph': self.paragraph,
            'table': self.table
        }
    
    def _parse_text(self) -> None:
        """본문 줄글 데이터 파싱"""
        StepLogger.log_step("본문 줄글 데이터 파싱 시작") 

        try:
            # PDF 파일 로드 및 유효성 검사
            pdf = PdfReader(str(self.pdf_path))
            if len(pdf.pages) == 0:
                raise ParsingError("PDF file is empty")

            try:
                extraction_prompt = '''
                당신은 PDF 텍스트 정제 및 번역 전문가입니다. 주어진 텍스트를 다음 규칙에 따라 정제하고 번역해주세요.

                규칙:
                1. 주어진 텍스트에서 의미 있는 본문 내용만 추출하고, 불필요한 모든 요소를 제거합니다.
                2. 제거할 요소:
                    - 형식적 문구: Compliance Note, Disclaimer, 면책조항, 유의사항, 투자자 고지사항 등
                    - "본 보고서", "이 보고서", "당사" 등으로 시작하는 형식적인 문구들
                    - 법적 책임, 위험 고지, 투자 위험 등에 관한 일체의 면책 문구
                    - 머리글과 바닥글: 페이지 번호, 문서 제목, 날짜 등
                    - 연구원 정보: 이름, 이메일, 직위 등
                    - 문맥 없이 반복되고 연속되는 숫자 패턴 데이터 숫자(%, 단위 포함) 줄바꿈으로 구분된 데이터, 열거된 통계값 등
                    - 문맥 없이 반복되고 연속되는 재무제표, 손익계산서 등 표 데이터 정보: 단위 표시 등(연속된 표 형태의 데이터, 나열된 숫자)
                    이 때, 문맥 내부에 반복 없이 위치한 수치 데이터는 삭제 금지 
                    - 표나 그림에 대한 언급: "<표>", "<그림>", "자료: [출처]" 등
                    - 증권사 관련 정보: 라이선스, 규제 관련 문구, 회사 소개 등
                    - 보고서 배포, 복제, 저작권 관련 주의사항
                    반드시 위 규칙이 잘 지켜졌는지 다시 한번 확인 및 검증하세요.
                3. 본문의 의미와 문맥을 보존하며, 원본 내용의 요약, 재구성, 변형 없이 작성합니다.
                4. 추출된 텍스트만 출력하며, 설명이나 주석을 포함하지 않습니다.
                5. 영어 텍스트는 자연스럽고 정확한 한국어로 번역합니다.
                6. 불필요한 공백과 줄바꿈을 제거하여, 가독성이 높은 상태로 최종 결과를 제공합니다.
                7. 중요: 숫자가 반복되거나 연속적으로 나오는 경우, 혹은 숫자를 중심으로 구성된 문장들은 모두 삭제합니다. 예외적으로 의미 전달에 필수적인 경우만 남깁니다.
                반드시 위 규칙이 잘 지켜졌는지 다시 한번 확인 및 검증하세요.

                텍스트:
                {text}
                '''
                # LangChain 프롬프트 템플릿 생성
                prompt_template = PromptTemplate(
                    input_variables=["text"],
                    template=extraction_prompt
                )

                # LLM 설정 
                llm = ChatClovaX(
                    model="HCX-003",
                    api_key=os.environ["CLOVA-STUDIO-API-KEY"]
                    )

                # 체인 생성 및 실행
                chain = prompt_template | llm
                response = chain.invoke({"text": self.paragraph})
                self.response = response
                # 정제된 텍스트를 paragraph에 저장
                self.paragraph = response.content

            except Exception as e:
                print("에러 발생:", str(e))
                print("에러 타입:", type(e))

        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {str(e)}")
            raise ParsingError(f"PDF 파싱 실패: {str(e)}")

    def _parse_meta_data(self) -> Optional[Dict]:
        """메타데이터 파싱"""
        StepLogger.log_step("원문데이터에서 메타데이터 파싱 시작")
        
        try:
            pdf = PdfReader(str(self.pdf_path))
            if len(pdf.pages) == 0:
                raise ParsingError("PDF file is empty")
            
            try:
                # 프롬프트 템플릿 정의
                extraction_prompt = """
                당신은 PDF 메타데이터 정제 전문가입니다. 주어진 텍스트를 아래 규칙에 따라 메타데이터를 정제해주세요.

                규칙:               
                1. 주어진 텍스트의 메타데이터를 반드시 아래와 같은 형식으로 한글로만 추출해주세요.
                2. 영어 텍스트도 반드시 한글로 번역해서 추출해주세요.
                3. 항목이 존재하지 않는다면 반드시 None으로 출력해주세요.
                4. **두 가지 이상 종목명이나 종목코드가 존재한다면 반드시 None으로 출력해주세요.** 
                5. 반드시 JSON 형식으로만 응답해주세요.
                6. 다른 설명이나 부가적인 텍스트 없이 순수 JSON만 응답해주세요.

                {{"제목": "문서 제목",
                "증권사명": "증권사 이름",
                "종목명": "종목 이름",
                "종목코드": "종목 코드",
                "작성일자": "작성 날짜"}}

                텍스트:
                {text}
                """
                
                # 프롬프트 템플릿 생성
                prompt_template = PromptTemplate(
                    input_variables=["text"],  # 여기서 "text"만 필요
                    template=extraction_prompt
                )

                # LLM 설정 
                llm = ChatClovaX(
                    model="HCX-003", 
                    api_key=os.environ["CLOVA-STUDIO-API-KEY"]
                    )

                chain = prompt_template | llm
                
                # 응답 파싱 전 로깅
                logger.debug(f"템플릿에 전달되는 텍스트: {self.raw_paragraph}")
                
                response = chain.invoke({"text": self.raw_paragraph})
                
                # 응답 로깅
                logger.debug(f"LLM 응답: {response.content if response else 'No response'}")
                
                if not response or not response.content:
                    raise ParsingError("LLM이 메타데이터를 추출하지 못했습니다")

                # 응답 로깅 및 기본 검증
                response_text = response.content.strip()
                logger.debug(f"LLM 응답: {response_text}")

                if not response_text or not ('{' in response_text and '}' in response_text):
                    raise ParsingError("유효하지 않은 응답 형식입니다")
                
                # JSON 파싱
                try:
                    # 응답에서 JSON 부분만 추출
                    response_text = response.content
                    # JSON 형식의 텍스트만 추출하기 위한 처리
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = response_text[json_start:json_end]
                        data = json.loads(json_str)
                        data = {k: self._clean_value(v) for k, v in data.items()}
                    else:
                        raise ParsingError("응답에서 JSON 형식을 찾을 수 없습니다")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 파싱 실패. 응답 내용: {response_text}")
                    raise ParsingError(f"JSON 파싱 실패: {str(e)}")
                
                # 필수 필드 검증
                required_fields = {'제목', '증권사명', '종목명', '종목코드', '작성일자'}
                missing_fields = required_fields - set(data.keys())
                if missing_fields:
                    raise ParsingError(f"필수 메타데이터 누락: {', '.join(missing_fields)}")
                
                logger.debug(f"메타데이터 파싱 완료: {data}")
                return data
                    
            except Exception as e:
                error_msg = f"""
                    메타데이터 추출 실패
                    에러 타입: {type(e).__name__}
                    에러 메시지: {str(e)}
                    트레이스백:
                    {traceback.format_exc()}
                    """
                logger.error(error_msg)
                raise ParsingError(f"메타데이터 추출 실패: {error_msg}")
                    
        except Exception as e:
            error_msg = f"""
                PDF 처리 중 오류 발생
                에러 타입: {type(e).__name__}
                에러 메시지: {str(e)}
                트레이스백:
                {traceback.format_exc()}
                """
            logger.error(error_msg)
            raise ParsingError(f"PDF 파싱 실패: {str(e)}")

    def _map_parsed_data(self) -> None:
        """메타데이터 매핑"""
        StepLogger.log_step("파싱 데이터 매핑 시작")
        
        try:
            # 필수 키가 없거나 빈 값인 경우 체크
            required_keys = ['title', 'firm', 'name', 'ticker', 'ref_date']
            if any(key not in self.temp_dict or not self.temp_dict[key] for key in required_keys):
                logger.info("원문에서 메타데이터를 추출합니다.")
                # pdf 파일 자체에서 제공되는 메타데이터가 존재하지 않을 경우, LLM 모델을 활용해 직접 메타데이터 추출
                parsed_data = self._parse_meta_data()
                if not parsed_data:
                    raise ParsingError("파싱된 데이터에 누락이 발생했거나 존재하지 않습니다.")
                # 데이터 매핑
                self.title = parsed_data['제목']
                self.firm = parsed_data['증권사명']
                self.name = parsed_data['종목명']
                self.ticker = parsed_data['종목코드']
                self.ref_date = standardize_date(self.temp_dict['ref_date'])

                if (not self.name or self.name=='None' or self.name=='') and (not self.ticker or self.ticker=="None" or self.ticker==""):
                    raise ParsingError("파싱된 데이터에 누락이 발생했거나 존재하지 않습니다.")
                
                if self.ref_date is None:
                    self.ref_date = standardize_date(parsed_data.get('작성일자'))
            else:
                self.title = self.temp_dict['title']
                self.firm = self.temp_dict['firm']
                self.name = self.temp_dict['name']
                self.ticker = self.temp_dict['ticker']
                self.ref_date = self.temp_dict['ref_date']

            # 모든 경우에 대해 티커 검증
            logger.info(f"티커 검증 시작 - 현재 상태: name={self.name}, ticker={self.ticker}")
            
            # 이름과 티커 정규화 함수
            def normalize_name(name):
                if not name:
                    return ""
                # 특수문자, 공백 제거 및 소문자 변환
                name = re.sub(r'[^\w\s가-힣]', '', name)
                name = name.replace(' ', '').lower()
                return name

            # 현재 있는 티커로 검증
            if self.ticker and self.ticker != "None":
                try:
                    db_name = stock.get_market_ticker_name(self.ticker)
                    if normalize_name(db_name) != normalize_name(self.name):
                        logger.warning(f"티커({self.ticker})에 해당하는 기업명({db_name})이 현재 기업명({self.name})과 다릅니다.")
                        self.name = db_name  # DB의 이름으로 업데이트
                except Exception as e:
                    logger.warning(f"티커 {self.ticker}로 기업명 조회 실패: {str(e)}")
                    self.ticker = "None"  # 잘못된 티커인 경우 초기화

            # 이름으로 티커 검색 (티커가 없거나 "None"인 경우)
            if not self.name:
                logger.warning("기업명이 없습니다.")
            elif not self.ticker or self.ticker == "None":
                logger.info(f"기업명으로 티커 검색 시작: name={self.name}")
                normalized_input_name = normalize_name(self.name)
                
                for ticker in self.tickers:
                    try:
                        market_name = stock.get_market_ticker_name(ticker)
                        normalized_market_name = normalize_name(market_name)
                        
                        logger.debug(f"비교: DB={normalized_market_name} vs Input={normalized_input_name}")
                        
                        if normalized_market_name == normalized_input_name:
                            self.ticker = ticker
                            self.name = market_name  # DB의 공식 이름으로 업데이트
                            logger.info(f"일치하는 티커 발견: {ticker} ({market_name})")
                            break
                    except Exception as e:
                        logger.debug(f"티커 {ticker} 조회 실패: {str(e)}")
                        continue
                
                if not self.ticker or self.ticker == "None":
                    logger.warning(f"'{self.name}' 에 대한 티커를 찾지 못했습니다.")
                            
        except Exception as e:
            logger.error(f"데이터 매핑 중 예상치 못한 오류 발생: {str(e)}")
            raise ParsingError(f"데이터 매핑 실패: {str(e)}")
        
    def add_metadata(self, key: str, value: Any) -> None:
        """추가 메타데이터 설정"""
        self.additional_metadata[key] = value

    def _clean_value(self, value):
        """값 정제"""
        if value is None or value.lower() == 'none':
            return "None"
        return value.strip()
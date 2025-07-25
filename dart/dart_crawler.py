import OpenDartReader
import pandas as pd
import requests
import io
import re
import os
from dotenv import load_dotenv
from miraeasset2025.utils.utils import standardize_date
from miraeasset2025.logger.logger import logger, StepLogger
from pykrx import stock
from typing import Dict, Optional, List, Any, Tuple
from datetime import date, datetime, timedelta

load_dotenv()

class DartCrawler:
    def __init__(self):
        api_key = os.getenv('DART_API_KEY')
        self.dart = OpenDartReader(api_key)
        self.report_codes = {
            '1분기사업보고서': '11013',
            '반기사업보고서': '11012',
            '3분기사업보고서': '11014',
            '사업보고서': '11011'
        }
        self.keywords = ['연결재무제표',
                         '연결재무제표주석']
        self.debug_mode = False
        self.url_dict = {}
        self.xml_content = None

    def get_html_directly(self, url):
        """URL에서 직접 HTML 데이터를 가져옵니다."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.3904.108 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to download HTML: {response.status_code}")
            
        return response.content

    def remove_none_dataframes(self, b):
        # None이 아닌 데이터프레임만 필터링하여 새로운 리스트 생성
        filtered_list = [df for df in b if not df.isnull().all().all()]
        return filtered_list 

    def merge_table_with_metadata(self, df_list, ref_date):
        merged_tables = []
        current_descriptions = set()  # 리스트 대신 set 사용하여 중복 제거
        
        def clean_description(text):
            # 연속된 공백을 하나로 치환
            text = re.sub(r'\s+', ' ', text.strip())
            return text
                
        def check_metadata_patterns(text):
            metadata_patterns = [
                r'^\(주\d*\)\s.*',  
                r'^제\s*\d+\s*(?:\([당전]\))?\s*기\s*(?:반기|분기)?말?',
                r'^\((?:당|전)(?:반기|분기|기)말?\)',
                r'^\(단위\s*:[^)]*\)',
                r'^\(\*\).*',                    # 단순 (*) 패턴
                r'^\(\*\d+\).*',                 # (*1), (*2) 등 번호가 있는 패턴
                r'^(?:당|전)(?:반기|분기|기)말?'
            ]
            return any(re.match(pattern, text.strip()) for pattern in metadata_patterns)
        
        for df in df_list:
            try:
                if df is None or df.empty:
                    continue
                    
                def contains_many_numbers(text):
                    if '(단위' in text:
                        return False 
                    # 날짜 형식 제거 (YYYY.MM.DD, YYYY-MM-DD, YYYY년 MM월 DD일)
                    text = re.sub(r'\d{4}[.년-]\s*\d{1,2}[.월-]\s*\d{1,2}[일]?', '', text)
                    
                    # 기수 표시 제거 (제 53 기, 제 52 기 등)
                    text = re.sub(r'제\s*\d+\s*기', '', text)
                    
                    # 분기 표시 제거 (1분기 등)
                    text = re.sub(r'\d+분기', '', text)
                    
                    # 이제 실제 데이터 값만 찾기
                    numbers = re.findall(r'(?:\([-\d,]+\)|\b[-\d,]+\.?\d*\b)', text)
                    
                    return len(numbers) >= 3

                if len(df) == 1:
                    description_values = df.iloc[0].dropna().values
                    description_text = clean_description(' '.join(str(val) for val in description_values))
                    
                    # 메타데이터 패턴 체크를 먼저 수행
                    if any(check_metadata_patterns(str(val)) for val in description_values):
                        current_descriptions.add(description_text)
                    # 숫자가 많은지는 그 다음에 체크
                    elif not contains_many_numbers(description_text):
                        current_descriptions.add(description_text)
                    continue
                    
                elif len(df) <= 5:  # 5행 이하인 경우
                    found_pattern = False
                    
                    # 먼저 패턴이 있는지 확인
                    for _, row in df.iterrows():
                        row_text = clean_description(' '.join(str(val) for val in row.dropna().values))
                        if not contains_many_numbers(row_text) and check_metadata_patterns(row_text):
                            found_pattern = True
                            break
                    
                    # 패턴이 발견되면 모든 행을 포함
                    if found_pattern:
                        all_rows_text = []
                        for _, row in df.iterrows():
                            row_text = clean_description(' '.join(str(val) for val in row.dropna().values))
                            # 숫자가 많은 행은 제외하고 나머지는 모두 포함
                            all_rows_text.append(row_text)
                        
                        if all_rows_text:
                            for text in all_rows_text:
                                current_descriptions.add(text)
                        continue
                
                else:
                    if current_descriptions:  
                        # set을 정렬된 리스트로 변환하여 일관된 순서 보장
                        descriptions_list = sorted(current_descriptions)
                        df.attrs['description'] = ' | '.join(descriptions_list)
                        df.attrs['ref_date'] = ref_date
                        current_descriptions.clear()  # set 초기화
                    merged_tables.append(df)
                    
            except Exception as e:
                print(f"데이터프레임 처리 중 오류 발생: {str(e)}")
                continue
        
        return merged_tables
        
    def crawl_company_reports(self, company_code, year):
        """특정 회사의 특정 연도 보고서들을 처리합니다."""
        results = {}
        company_name = stock.get_market_ticker_name(company_code)
        # 현재 날짜 체크
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month

        for report_name, report_code in self.report_codes.items():
            if year == current_year:
                if report_code == '11011':  # 사업보고서
                    logger.info(f"{year}년 사업보고서는 아직 제출 시기가 아닙니다.")
                    continue
                elif report_code == '11014' and current_month < 11:  # 3분기
                    logger.info(f"{year}년 3분기보고서는 아직 제출 시기가 아닙니다.")
                    continue
                elif report_code == '11012' and current_month < 8:  # 반기
                    logger.info(f"{year}년 반기보고서는 아직 제출 시기가 아닙니다.")
                    continue
                elif report_code == '11013' and current_month < 5:  # 1분기
                    logger.info(f"{year}년 1분기보고서는 아직 제출 시기가 아닙니다.")
                    continue
            try:
                # 1. 공시 번호 가져오기
                finstate = self.dart.finstate(company_code, year, reprt_code=report_code)
                if finstate is None or finstate.empty or 'rcept_no' not in finstate.columns:
                    logger.info(f"No data found for {company_code} in {year}년_{report_name}")
                    continue
                    
                rcept_no = finstate['rcept_no'].iloc[0]
                date_part = str(rcept_no)[:8]
                ref_date = standardize_date(date_part)
                logger.info(f"{company_code}_{year}년_{report_name} 검색 완료")

                # 2. 문서 URL 가져오기
                for keyword in self.keywords:
                    try:
                        df_docs = self.dart.sub_docs(rcept_no, match=keyword)
                        if df_docs.empty:
                            logger.info(f"키워드 '{keyword}'에 대한 공시 문서가 존재하지 않습니다.")
                            continue
                        
                        matching_docs = df_docs.head(1)
                        if matching_docs.empty:
                            logger.info(f"키워드 '{keyword}'와 정확히 일치하는 문서가 없습니다.")
                            continue
                        keyword = keyword.replace(' ', '_')
                        url = matching_docs['url'].iloc[0]
                        self.url_dict[keyword] = url
                    except Exception as e:
                        logger.error(f"URL 검색 중 오류 발생 (keyword: {keyword}): {str(e)}")
                        continue
                
                # URL을 찾지 못한 경우 다음 보고서로
                if not self.url_dict:
                    logger.info(f"{report_name}에서 처리할 URL을 찾지 못했습니다.")
                    continue
                
                # 3. 각 URL에 대해 처리
                for keyword, url in self.url_dict.items():
                    try:
                        # XML 직접 가져오기
                        html_content = self.get_html_directly(url)
                        
                        # 테이블 추출 
                        tables = pd.read_html(html_content)
                        preprocessed_tables = self.remove_none_dataframes(tables)
                        
                        # 키워드에 따라 다른 처리 로직 적용
                        if '주석' in keyword:
                            # 주석용 처리 로직
                            merged_meta_tables = self.merge_table_with_metadata_2(preprocessed_tables, ref_date)  # 임시로 그대로 통과
                            # TODO: 나중에 merge_note_with_metadata 함수 구현
                            # merged_meta_tables = self.merge_note_with_metadata(preprocessed_tables, ref_date)
                            
                        else:
                            # 기존 재무제표 처리 로직
                            merged_meta_tables = self.merge_table_with_metadata(preprocessed_tables, ref_date)
                        
                        # 결과 저장
                        results[f"{company_name}_{company_code}_{year}_{report_code}_{report_name}_{keyword}"] = merged_meta_tables
                        logger.info(f"{company_name}_{company_code}_{year}_{report_code}_{report_name}_{keyword} 저장 완료")
                    except Exception as e:
                        logger.error(f"테이블 처리 중 오류 발생 (URL: {url}): {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"보고서 처리 중 오류 발생 ({report_name}, {year}): {str(e)}")
                continue
                
        return results

    def crawl_company_annotations(self, company_code, year):
        """특정 회사의 특정 연도의 연결재무제표 주석들을 분류합니다."""
        results = {}
        company_name = stock.get_market_ticker_name(company_code)
        # 현재 날짜 체크
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month

        for report_name, report_code in self.report_codes.items():
            if year == current_year:
                if report_code == '11011':  # 사업보고서
                    logger.info(f"{year}년 사업보고서는 아직 제출 시기가 아닙니다.")
                    continue
                elif report_code == '11014' and current_month < 11:  # 3분기
                    logger.info(f"{year}년 3분기보고서는 아직 제출 시기가 아닙니다.")
                    continue
                elif report_code == '11012' and current_month < 8:  # 반기
                    logger.info(f"{year}년 반기보고서는 아직 제출 시기가 아닙니다.")
                    continue
                elif report_code == '11013' and current_month < 5:  # 1분기
                    logger.info(f"{year}년 1분기보고서는 아직 제출 시기가 아닙니다.")
                    continue
            try:
                # 1. 공시 번호 가져오기
                finstate = self.dart.finstate(company_code, year, reprt_code=report_code)
                if finstate is None or finstate.empty or 'rcept_no' not in finstate.columns:
                    logger.info(f"No data found for {company_code} in {year}년_{report_name}")
                    continue
                    
                rcept_no = finstate['rcept_no'].iloc[0]
                date_part = str(rcept_no)[:8]
                ref_date = standardize_date(date_part)
                logger.info(f"{company_code}_{year}년_{report_name} 검색 완료")

                # 2. 문서 URL 가져오기
                for keyword in self.keywords:
                    try:
                        df_docs = self.dart.sub_docs(rcept_no, match=keyword)
                        if df_docs.empty:
                            logger.info(f"키워드 '{keyword}'에 대한 공시 문서가 존재하지 않습니다.")
                            continue
                        
                        matching_docs = df_docs.head(1)
                        if matching_docs.empty:
                            logger.info(f"키워드 '{keyword}'와 정확히 일치하는 문서가 없습니다.")
                            continue
                        keyword = keyword.replace(' ', '_')
                        url = matching_docs['url'].iloc[0]
                        self.url_dict[keyword] = url
                    except Exception as e:
                        logger.error(f"URL 검색 중 오류 발생 (keyword: {keyword}): {str(e)}")
                        continue
                
                # URL을 찾지 못한 경우 다음 보고서로
                if not self.url_dict:
                    logger.info(f"{report_name}에서 처리할 URL을 찾지 못했습니다.")
                    continue
                
                # 3. 각 URL에 대해 처리
                for keyword, url in self.url_dict.items():
                    try:
                        # XML 직접 가져오기
                        html_content = self.get_html_directly(url)
                        
                        # 키워드에 따라 다른 처리 로직 적용
                        if '주석' in keyword:
                            # 주석용 처리 로직
                            # merged_meta_tables = self.merge_table_with_metadata_2(preprocessed_tables, ref_date)  # 임시로 그대로 통과
                            # TODO: 나중에 merge_note_with_metadata 함수 구현
                            # merged_meta_tables = self.merge_note_with_metadata(preprocessed_tables, ref_date)
                            decoded_text = html_content.decode("utf-8")
                            section_dict = self.split_html_sections_by_numbered_p(decoded_text)
                            results = section_dict
                        logger.info(f"{company_name}_{company_code}_{year}_{report_code}_{report_name}_{keyword} 주석 분류 저장 완료")
                    except Exception as e:
                        logger.error(f"테이블 처리 중 오류 발생 (URL: {url}): {str(e)}")
                        continue
                
            except Exception as e:
                logger.error(f"보고서 처리 중 오류 발생 ({report_name}, {year}): {str(e)}")
                continue
                
        return results
    
    def merge_table_with_metadata_2(self, df_list, ref_date):
        merged_tables = []
        current_metadata = []
        current_table = None

        def clean_description(text):
            # 연속된 공백을 하나로 치환
            text = re.sub(r'\s+', ' ', text.strip())
            return text

        def check_metadata_patterns(text):
            metadata_patterns = [
                r'^\(주\d*\)\s.*',  
                r'^제\s*\d+\s*(?:\([당전]\))?\s*기\s*(?:반기|분기)?말?',
                r'^\((?:당|전)(?:반기|분기|기)말?\)',  # 수정된 패턴
                r'^\(단위\s*:[^)]*\)'  
            ]
            return any(re.match(pattern, text.strip()) for pattern in metadata_patterns)

        def contains_many_numbers(text):
            # 날짜 형식 제거 (YYYY.MM.DD, YYYY-MM-DD, YYYY년 MM월 DD일)
            if '(단위' in text:
                return False
            
            text = re.sub(r'\d{4}[.년-]\s*\d{1,2}[.월-]\s*\d{1,2}[일]?', '', text)
            
            # 기수 표시 제거 (제 53 기, 제 52 기 등)
            text = re.sub(r'제\s*\d+\s*기', '', text)
            
            # 분기 표시 제거 (1분기 등)
            text = re.sub(r'\d+분기', '', text)
            
            # 이제 실제 데이터 값만 찾기
            numbers = re.findall(r'(?:\([-\d,]+\)|\b[-\d,]+\.?\d*\b)', text)
            
            return len(numbers) >= 3

        def is_metadata_row(text):
            return check_metadata_patterns(text) and not contains_many_numbers(text)

        def is_metadata_table(df):
            if df is None or df.empty:
                return False

            # 1행인 경우 특별 처리
            if len(df) == 1:
                row = df.iloc[0]
                row_text = clean_description(' '.join(str(val) for val in row.dropna().values))
                # 메타데이터 패턴이 있으면 바로 True 반환
                if check_metadata_patterns(row_text):
                    return True
            
            # 기존 로직
            metadata_row_count = 0
            for _, row in df.iterrows():
                row_text = clean_description(' '.join(str(val) for val in row.dropna().values))
                if is_metadata_row(row_text):
                    metadata_row_count += 1

            return metadata_row_count / len(df) > 0.5

        def process_metadata(df):
            metadata_texts = []
            for _, row in df.iterrows():
                row_text = clean_description(' '.join(str(val) for val in row.dropna().values))
                if is_metadata_row(row_text):
                    metadata_texts.append(row_text)
            return metadata_texts

        try:
            for df in df_list:
                if df is None or df.empty:
                    continue

                # 메타데이터 테이블 판별
                if is_metadata_table(df):
                    # 메타데이터 테이블인 경우
                    metadata_texts = process_metadata(df)
                    current_metadata.extend(metadata_texts)
                    
                    # 이전 테이블이 있으면 메타데이터 추가
                    if current_table is not None:
                        if current_metadata:
                            current_table.attrs['description'] = ' | '.join(sorted(set(current_metadata)))
                        merged_tables.append(current_table)
                        current_table = None
                        current_metadata = []
                else:
                    # 본문 데이터 테이블인 경우
                    if current_table is not None:
                        # 이전 테이블이 있다면 먼저 처리
                        if current_metadata:
                            current_table.attrs['description'] = ' | '.join(sorted(set(current_metadata)))
                        merged_tables.append(current_table)
                    
                    # 새로운 본문 테이블 설정
                    current_table = df
                    current_table.attrs['ref_date'] = ref_date  # 기본 메타데이터로 ref_date 설정
                    current_metadata = []

            # 마지막 테이블 처리
            if current_table is not None:
                if current_metadata:
                    current_table.attrs['description'] = ' | '.join(sorted(set(current_metadata)))
                merged_tables.append(current_table)

        except Exception as e:
            print(f"데이터프레임 처리 중 오류 발생: {str(e)}")

        return merged_tables

    def split_html_sections_by_numbered_p(self, html: str) -> dict:
        # <P>, <P><BR/>, <P><SPAN> 등 다양한 조합을 포괄적으로 대응
        pattern = re.compile(
            r"<P[^>]*>(?:<BR/?>|\s|&nbsp;|<SPAN[^>]*>|</SPAN>)*\s*(\d{1,2})[.]\s*([^<\r\n]*)",
            re.IGNORECASE
        )

        matches = list(pattern.finditer(html))
        result = {}

        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(html)
            section_html = html[start:end]

            number = matches[i].group(1).strip()
            title = matches[i].group(2).strip()
            key = f"{number}. {title}" if title else f"{number}."

            result[key] = section_html.strip()

        return result
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Dict, Optional, Union, List, Any
from miraeasset2025.logger.logger import logger, StepLogger
from IPython.display import display, HTML

def standardize_date(input_date: str) -> datetime:
    if not input_date:
        raise ValueError("Empty date string")
        
    input_date = str(input_date).replace(" ", "")
    logger.debug(f"입력된 날짜 문자열: '{input_date}'")
    
    date_formats = [
            # 한글 포맷 (공백 다양화)
            '%Y년 %m월 %d일',      # 2024년 11월 01일
            '%Y년 %m월 %-d일',     # 2024년 11월 1일
            '%Y년 %-m월 %d일',     # 2024년 1월 01일
            '%Y년 %-m월 %-d일',    # 2024년 1월 1일
            '%Y년%m월%d일',        # 2024년11월01일
            '%Y년%m월%-d일',       # 2024년11월1일
            '%Y년%-m월%d일',       # 2024년1월01일
            '%Y년%-m월%-d일',      # 2024년1월1일
            
            # 연월 포맷 추가
            '%Y년%m월',           # 2024년8월
            '%Y년 %m월',         # 2024년 8월
            '%Y년%-m월',         # 2024년8월
            '%Y년 %-m월',        # 2024년 8월
            
            # 점 구분 포맷
            '%Y.%m.%d.',          # 2024.11.01.
            '%Y.%-m.%-d.',        # 2024.1.1.
            '%Y.%m.%d',           # 2024.11.01
            '%Y.%-m.%-d',         # 2024.1.1
            
            # 영문 포맷
            '%b %d, %Y',          # Oct 31, 2024
            '%b %-d, %Y',         # Oct 1, 2024
            '%B %d, %Y',          # October 31, 2024
            '%B %-d, %Y',         # October 1, 2024
            '%b %d %Y',           # Oct 31 2024
            '%b %-d %Y',          # Oct 1 2024
            
            # 기타 포맷
            '%Y-%m-%d',           # 2024-11-01
            '%Y-%m-%-d',          # 2024-11-1
            '%Y/%m/%d',           # 2024/11/01
            '%Y/%m/%-d',          # 2024/11/1
            '%Y, %m, %d',         # 2024, 11, 01
            '%Y, %-m, %-d',       # 2024, 1, 1
            '%Y. %m. %d.',        # 2024. 11. 01.
            '%Y. %-m. %-d.',      # 2024. 1. 1.
            '%Y%m%d',
            
            # 온점과 반점 교차 포맷
            '%Y.%m,%d',           # 2024.11,01
            '%Y.%-m,%-d',         # 2024.1,1
            '%Y,%m.%d',           # 2024,11.01
            '%Y,%-m.%-d',         # 2024,1.1
            '%Y. %m, %d',         # 2024. 11, 01
            '%Y. %-m, %-d',       # 2024. 1, 1
            '%Y, %m. %d',         # 2024, 11. 01
            '%Y, %-m. %-d',       # 2024, 1. 1
            '%Y.%m.%d,',          # 2024.11.01,
            '%Y.%-m.%-d,',        # 2024.1.1,
            '%Y,%m,%d.',          # 2024,11,01.
            '%Y,%-m,%-d.',        # 2024,1,1.
        ]
    
    for date_format in date_formats:
        try:
            return datetime.strptime(input_date, date_format).date()
        except ValueError:
            continue
    
    # 파싱 실패 시 현재 날짜 반환
    logger.warning('본문에서 작성날짜를 찾을 수 없어 현재 날짜를 사용합니다.')
    return datetime.now()

def standardize_date_str(input_date: str) -> str:
    """날짜 문자열을 datetime 객체로 변환"""
    if not input_date:
        logger.warning('본문에서 작성날짜를 찾을 수 없습니다.')
        return '0000-00-00'
        
    input_date = str(input_date).replace(" ", "")
    
    # 로깅 추가
    logger.debug(f"입력된 날짜 문자열: '{input_date}'")
    
    date_formats = [
            # 한글 포맷 (공백 다양화)
            '%Y년 %m월 %d일',      # 2024년 11월 01일
            '%Y년 %m월 %-d일',     # 2024년 11월 1일
            '%Y년 %-m월 %d일',     # 2024년 1월 01일
            '%Y년 %-m월 %-d일',    # 2024년 1월 1일
            '%Y년%m월%d일',        # 2024년11월01일
            '%Y년%m월%-d일',       # 2024년11월1일
            '%Y년%-m월%d일',       # 2024년1월01일
            '%Y년%-m월%-d일',      # 2024년1월1일
            
            # 연월 포맷 추가
            '%Y년%m월',           # 2024년8월
            '%Y년 %m월',         # 2024년 8월
            '%Y년%-m월',         # 2024년8월
            '%Y년 %-m월',        # 2024년 8월
            
            # 점 구분 포맷
            '%Y.%m.%d.',          # 2024.11.01.
            '%Y.%-m.%-d.',        # 2024.1.1.
            '%Y.%m.%d',           # 2024.11.01
            '%Y.%-m.%-d',         # 2024.1.1
            
            # 영문 포맷
            '%b %d, %Y',          # Oct 31, 2024
            '%b %-d, %Y',         # Oct 1, 2024
            '%B %d, %Y',          # October 31, 2024
            '%B %-d, %Y',         # October 1, 2024
            '%b %d %Y',           # Oct 31 2024
            '%b %-d %Y',          # Oct 1 2024
            
            # 기타 포맷
            '%Y-%m-%d',           # 2024-11-01
            '%Y-%m-%-d',          # 2024-11-1
            '%Y/%m/%d',           # 2024/11/01
            '%Y/%m/%-d',          # 2024/11/1
            '%Y, %m, %d',         # 2024, 11, 01
            '%Y, %-m, %-d',       # 2024, 1, 1
            '%Y. %m. %d.',        # 2024. 11. 01.
            '%Y. %-m. %-d.',      # 2024. 1. 1.
            '%Y%m%d',
            
            # 온점과 반점 교차 포맷
            '%Y.%m,%d',           # 2024.11,01
            '%Y.%-m,%-d',         # 2024.1,1
            '%Y,%m.%d',           # 2024,11.01
            '%Y,%-m.%-d',         # 2024,1.1
            '%Y. %m, %d',         # 2024. 11, 01
            '%Y. %-m, %-d',       # 2024. 1, 1
            '%Y, %m. %d',         # 2024, 11. 01
            '%Y, %-m. %-d',       # 2024, 1. 1
            '%Y.%m.%d,',          # 2024.11.01,
            '%Y.%-m.%-d,',        # 2024.1.1,
            '%Y,%m,%d.',          # 2024,11,01.
            '%Y,%-m,%-d.',        # 2024,1,1.
        ]
    
    for date_format in date_formats:
        try:
            return datetime.strptime(input_date, date_format).strftime('%Y-%m-%d')  # 문자열로 변환
        except ValueError:
            continue
    
    # 모든 형식이 실패한 경우
    logger.warning('본문에서 작성날짜를 찾을 수 없습니다.')
    return '0000-00-00'

def convert_structured_query_to_metadata_filter(query_output):
    metadata_filter = {}
    
    if query_output.filter:
        if hasattr(query_output.filter, 'operator'):  # AND/OR 연산인 경우
            for arg in query_output.filter.arguments:
                if arg.attribute == 'ref_date':
                    # 날짜 비교 - ChromaDB 스타일로 변경
                    if arg.comparator == 'gt':
                        metadata_filter["ref_date"] = {"$gt": arg.value['date'].replace('-', '.')}
                    elif arg.comparator == 'lt':
                        metadata_filter["ref_date"] = {"$lt": arg.value['date'].replace('-', '.')}
                    elif arg.comparator == 'gte':
                        metadata_filter["ref_date"] = {"$gte": arg.value['date'].replace('-', '.')}
                    elif arg.comparator == 'lte':
                        metadata_filter["ref_date"] = {"$lte": arg.value['date'].replace('-', '.')}
                    elif arg.comparator == 'eq':
                        metadata_filter["ref_date"] = {"$eq": arg.value['date'].replace('-', '.')}
                    elif arg.comparator == 'ne':
                        metadata_filter["ref_date"] = {"$ne": arg.value['date'].replace('-', '.')}
                else:
                    # 일반 필드 비교
                    if arg.comparator == 'eq':
                        metadata_filter[arg.attribute] = {"$eq": arg.value}
                    elif arg.comparator == 'ne':
                        metadata_filter[arg.attribute] = {"$ne": arg.value}
        else:  # 단일 조건인 경우
            if query_output.filter.attribute == 'ref_date':
                if query_output.filter.comparator == 'gt':
                    metadata_filter["ref_date"] = {"$gt": query_output.filter.value['date'].replace('-', '.')}
                elif query_output.filter.comparator == 'lt':
                    metadata_filter["ref_date"] = {"$lt": query_output.filter.value['date'].replace('-', '.')}
                elif query_output.filter.comparator == 'gte':
                    metadata_filter["ref_date"] = {"$gte": query_output.filter.value['date'].replace('-', '.')}
                elif query_output.filter.comparator == 'lte':
                    metadata_filter["ref_date"] = {"$lte": query_output.filter.value['date'].replace('-', '.')}
                elif query_output.filter.comparator == 'eq':
                    metadata_filter["ref_date"] = {"$eq": query_output.filter.value['date'].replace('-', '.')}
                elif query_output.filter.comparator == 'ne':
                    metadata_filter["ref_date"] = {"$ne": query_output.filter.value['date'].replace('-', '.')}
            else:
                if query_output.filter.comparator == 'eq':
                    metadata_filter[query_output.filter.attribute] = {"$eq": query_output.filter.value}
                elif query_output.filter.comparator == 'ne':
                    metadata_filter[query_output.filter.attribute] = {"$ne": query_output.filter.value}

    return metadata_filter
    
def pretty_print_conversation(result: Dict[Any, Any]) -> None:
    """
    대화 내용을 보기 좋게 출력하는 메서드
    
    Args:
        response: RAG 체인의 응답 딕셔너리
    """
    print("\n" + "="*50)
    print(f"📅 대화 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 질문 출력
    print("\n👤 질문:")
    print(f"   {result['question']}")
    
    # 답변 출력
    print("\n🤖 답변:")
    print(f"   {result['answer']}")  # self.answer 사용
    
    # 참고 문서 출력
    if 'source_documents' in result.keys():
        print("\n📚 참고 문서:")
        for i, doc in enumerate(result['source_documents'], 1):
            content = doc['content'].split('---문서 내용---\n')[-1][:200]

            print(f"\n   문서 {i}:")
            print(f"   - 기업명: {doc['metadata']['company_name']}")
            print(f"   - 종목코드: {doc['metadata']['ticker']}")
            print(f"   - 작성일: {doc['metadata']['ref_date']}")
            print(f"   - 증권사: {doc['metadata']['publisher']}")
            print(f"   - 내용: {content}...")
    # 토큰 사용량 출력
    if 'token_usage' in result.keys():
        print("\n📊 토큰 사용량:")
        print(f"   - 입력 토큰: {result['token_usage']['input_tokens']}")
        print(f"   - 출력 토큰: {result['token_usage']['output_tokens']}")
    
    print("\n" + "="*50 + "\n")

def display_financial_tables(dataframes):
    """재무제표 데이터프레임을 다크 모드로 출력"""
    for df in dataframes:
        # 다크 모드 스타일 적용
        styled_df = df.style.set_properties(**{
            'background-color': '#2d2d2d',  # 어두운 배경색
            'color': '#ffffff',  # 흰색 텍스트
            'border-color': '#404040',  # 어두운 회색 테두리
            'border-style': 'solid',
            'border-width': '1px',
            'text-align': 'right',
            'padding': '8px'  # 여백 추가로 가독성 향상
        })
        
        # 속성 정보를 다크 모드로 출력
        if hasattr(df, 'attrs'):
            attr_html = '<div style="margin-bottom: 10px; padding: 10px; background-color: #1a1a1a; color: #ffffff; border-radius: 5px;">'
            for key, value in df.attrs.items():
                attr_html += f'<strong style="color: #00b4d8;">{key}:</strong> <span style="color: #ffffff;">{value}</span><br>'
            attr_html += '</div>'
            display(HTML(attr_html))
            
        display(styled_df)
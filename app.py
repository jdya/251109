import json
import requests
import streamlit as st
from PyPDF2 import PdfReader
from supabase import create_client, Client
from openai import OpenAI # OpenAI 라이브러리 임포트
from langchain_text_splitters import RecursiveCharacterTextSplitter # 텍스트 분할을 위해 추가
import time # API 호출 간 지연을 위해 추가

try:
    # 전용 DeepSeek Python 라이브러리 (클라이언트 클래스는 api.DeepSeekAPI)
    from deepseek.api import DeepSeekAPI  # type: ignore
except Exception:
    DeepSeekAPI = None  # 라이브러리 미설치 시 None 처리

st.set_page_config(page_title="교사용 AI 에이전트 v3", page_icon="🤖", layout="centered")
st.title("교사용 AI 에이전트 v3")

# Supabase 클라이언트 초기화
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        supabase = None
        st.warning(f"Supabase 초기화 실패: {e}", icon="⚠️")
else:
    st.warning("Supabase 설정(SUPABASE_URL, SUPABASE_KEY)이 없어 저장 기능이 비활성화됩니다.", icon="⚠️")

# DeepSeek 임베딩 모델 설정(요구사항에 맞춰 고정값 사용)
EMBEDDING_MODEL = "deepseek-embed"

# DeepSeek 클라이언트(임베딩용) 초기화
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY")
deepseek_client = None
class _DSResp:
    def __init__(self, data):
        self.data = data

class DeepseekCompatClient:
    """OpenAI 스타일 embeddings.create를 제공하는 간단 래퍼.

    /v1/embeddings 또는 /embeddings를 호출하고,
    404/405 시 임시 1536차원 0 벡터로 폴백합니다.
    """
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is missing")
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.embeddings = self.Embeddings(self)

    class Embeddings:
        def __init__(self, parent: "DeepseekCompatClient"):
            self.parent = parent

        def create(self, model: str, input: str):
            payload = {"model": model, "input": input}
            # 1차: /v1/embeddings
            url1 = f"{self.parent.base_url}/v1/embeddings"
            r = requests.post(url1, headers=self.parent.headers, json=payload, timeout=60)
            if r.status_code in (404, 405):
                # 2차: /embeddings
                url2 = f"{self.parent.base_url}/embeddings"
                r2 = requests.post(url2, headers=self.parent.headers, json=payload, timeout=60)
                if r2.status_code < 300:
                    j2 = r2.json()
                    data2 = j2.get("data", [])
                    return _DSResp(data2)
                # 최종 폴백: 임시 벡터 반환
                return _DSResp([{"embedding": [0.0] * 1536}])
            r.raise_for_status()
            j = r.json()
            data = j.get("data", [])
            return _DSResp(data)

if DEEPSEEK_API_KEY:
    try:
        deepseek_client = DeepseekCompatClient(DEEPSEEK_API_KEY)
    except Exception as e:
        deepseek_client = None
        st.warning(f"DeepSeek 클라이언트 초기화 실패: {e}", icon="⚠️")


# 세션 상태 초기화: 대화 기록(messages)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "당신은 친절한 AI 코딩 선생님입니다. 초보자에게 한국어로 명확히 설명하고, 단계별로 안내합니다. 필요 시 예제 코드와 실습 팁을 제공합니다."}, # 시스템 메시지 추가
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
    ]

# DeepSeek 채팅 클라이언트 초기화 (OpenAI 호환)
deepseek_chat_client = None
if DEEPSEEK_API_KEY:
    try:
        deepseek_chat_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")
    except Exception as e:
        st.warning(f"DeepSeek 채팅 클라이언트 초기화 실패: {e}", icon="⚠️")


def _deepseek_stream(messages: list[dict], client: OpenAI, model: str = "deepseek-chat"):
    """DeepSeek Chat Completions 스트리밍 응답을 제너레이터로 반환 (OpenAI 클라이언트 사용)."""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.2,
            timeout=60.0 # 60초 타임아웃 추가
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
    except requests.exceptions.Timeout:
        st.error("DeepSeek 채팅 API 호출 시간이 초과되었습니다.")
        yield from _fallback_stream("DeepSeek 채팅 API 호출 시간 초과")
    except Exception as e:
        st.error(f"DeepSeek 채팅 API 호출 실패: {e}")
        yield from _fallback_stream("DeepSeek 채팅 API 호출 실패")


def _fallback_stream(prompt: str):
    """API 키가 없거나 오류가 난 경우를 위한 간단한 스트리밍 데모."""
    demo = f"(데모) 입력하신 내용에 대한 응답: {prompt}"
    for ch in demo:
        yield ch


# PDF 텍스트 추출
def get_pdf_text(pdf_file) -> str:
    try:
        reader = PdfReader(pdf_file)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            texts.append(t)
        return "\n".join(texts).strip()
    except Exception as e:
        raise RuntimeError(f"PDF 텍스트 추출 실패: {e}")


# DeepSeek 임베딩 생성
def get_embedding(text: str, client) -> list[float]:
    # 요구사항: client.embeddings.create(model="deepseek-embed", ...)
    if client is None:
        raise RuntimeError(
            "DeepSeek 라이브러리가 없거나 클라이언트가 초기화되지 않았습니다. "
            "터미널에서 'pip install deepseek' 실행 후, .streamlit/secrets.toml에 DEEPSEEK_API_KEY를 설정하세요."
        )
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    except requests.exceptions.Timeout:
        raise TimeoutError("임베딩 생성 요청 시간이 초과되었습니다.")
    except Exception as e:
        st.sidebar.warning(f"임베딩 호출 실패: {e}. 임시 벡터로 저장합니다.")
        return [0.0] * 1536

    vec = getattr(resp, "data", [None])[0]
    if isinstance(vec, dict):
        vec = vec.get("embedding") or vec.get("vector")

    if not isinstance(vec, list):
        st.sidebar.warning("임베딩 응답 형식 오류: 임시 벡터(0)로 저장합니다.")
        vec = [0.0] * 1536

    # 벡터 길이 정규화(테이블 스키마: 1536차원)
    target_dim = 1536
    if len(vec) > target_dim:
        vec = vec[:target_dim]
    return vec


# Supabase에 임베딩 저장
def save_embedding_to_supabase(file_name: str, text_chunk: str, embedding: list[float]):
    if supabase:
        try:
            # 'documents' 테이블에 저장
            supabase.table("documents").insert({
                "file_name": file_name,
                "content": text_chunk,
                "embedding": embedding
            }).execute()
            return True
        except Exception as e:
            st.error(f"Supabase 저장 실패: {e}")
            return False
    return False


# 텍스트를 청크로 분할
def split_text_into_chunks(text: str) -> list[str]:
    # RecursiveCharacterTextSplitter를 사용해 텍스트를 문단 조각으로 쪼갭니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_saved_files_from_supabase():
    if supabase:
        try:
            response = supabase.table("documents").select("file_name").execute()
            # 중복 제거를 위해 set을 사용한 후 list로 변환
            file_names = sorted(list(set([item['file_name'] for item in response.data])))
            return file_names
        except Exception as e:
            st.exception(e) # 오류를 Streamlit UI에 표시
            st.error(f"Supabase에서 저장된 파일 목록을 가져오는 데 실패했습니다: {e}")
            return []
    return []

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("비밀번호를 입력해주세요.")
    password = st.text_input("", type="password", key="password_input")
    if st.button("로그인", key="login_button"):
        if password == st.secrets.get("APP_PASSWORD"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 틀렸습니다.")
    st.stop()

# --- Streamlit UI --- 

# 세션 상태에 챗봇 모드 및 메시지 초기화
if "chatbot_mode" not in st.session_state:
    st.session_state.chatbot_mode = "일반 챗봇"
if "general_chatbot_messages" not in st.session_state:
    st.session_state.general_chatbot_messages = []
if "rag_chatbot_messages" not in st.session_state:
    st.session_state.rag_chatbot_messages = []

with st.sidebar:
    st.header("챗봇 모드 선택")
    if st.button("일반 챗봇", key="btn_general_chatbot", use_container_width=True):
        st.session_state.chatbot_mode = "일반 챗봇"
    if st.button("RAG 챗봇", key="btn_rag_chatbot", use_container_width=True):
        st.session_state.chatbot_mode = "RAG 챗봇"

    st.markdown(f"현재 모드: **{st.session_state.chatbot_mode}**")

    if st.session_state.chatbot_mode == "RAG 챗봇":
        st.header("파일 업로드")
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"], key="rag_file_uploader")

        if uploaded_file and st.button("임베딩 및 저장", key="rag_upload_button"):
            if not supabase:
                st.error("Supabase 연결이 필요합니다.")
            elif not deepseek_client:
                st.error("DeepSeek API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다.")
            else:
                with st.spinner("PDF 처리 중..."):
                    try:
                        pdf_text = get_pdf_text(uploaded_file)
                        chunks = split_text_into_chunks(pdf_text)
                        
                        saved_count = 0
                        for i, chunk in enumerate(chunks):
                            if not chunk:
                                continue
                            embedding = get_embedding(chunk, deepseek_client)
                            if embedding:
                                if save_embedding_to_supabase(uploaded_file.name, chunk, embedding):
                                    saved_count += 1
                            time.sleep(0.1) # API 호출 간 지연 추가
                        
                        st.success(f"{saved_count}개의 텍스트 청크가 Supabase에 저장되었습니다.")
                    except Exception as e:
                        st.error(f"파일 처리 중 오류 발생: {e}")

        st.header("학습된 파일 목록")
        saved_files = get_saved_files_from_supabase()
        if saved_files:
            for f_name in saved_files:
                st.write(f"- {f_name}")
        else:
            st.write("저장된 파일이 없습니다.")

# RAG 모드 스위치 제거
# rag_mode = st.toggle("🤖 맞춤형 RAG 모드 켜기", help="개인 자료를 기반으로 답변합니다")

# 메인 채팅 인터페이스
# 현재 모드에 맞는 메시지 리스트 선택
if st.session_state.chatbot_mode == "일반 챗봇":
    current_messages = st.session_state.general_chatbot_messages
else: # RAG 챗봇
    current_messages = st.session_state.rag_chatbot_messages

# 이전 메시지 표시
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("질문을 입력하세요...")

if user_query:
    current_messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    full_response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # RAG 모드 선택에 따라 조건부 실행
        if st.session_state.chatbot_mode == "RAG 챗봇" and supabase and deepseek_client: # RAG 챗봇 모드 활성화 및 필요한 클라이언트 존재
            # 1. 사용자 쿼리 임베딩
            try:
                with st.spinner("쿼리 임베딩 생성 중..."):
                    query_embedding = get_embedding(user_query, deepseek_client)
            except TimeoutError:
                st.error("임베딩 생성 요청 시간이 초과되었습니다. 다시 시도해주세요.")
                query_embedding = None
            except Exception as e:
                st.error(f"쿼리 임베딩 생성 실패: {e}")
                query_embedding = None

            if query_embedding:
                # 2. Supabase에서 유사한 문서 검색 (pg_vector 사용)
                try:
                    with st.spinner("관련 문서 검색 중..."):
                        response = supabase.rpc(
                            'match_documents',
                            {
                                'query_embedding': query_embedding,
                                'match_threshold': 0.7, # 유사도 임계값
                                'match_count': 5 # 가져올 문서 수
                            }
                        ).execute()
                    retrieved_docs = response.data

                    if retrieved_docs:
                        context = "\n\n".join([doc["content"] for doc in retrieved_docs])
                        rag_messages = [
                            {"role": "system", "content": "다음 문서를 참고하여 질문에 답변하세요:\n\n" + context},
                        ] + current_messages
                        current_messages.append({"role": "system", "content": f"RAG 검색을 통해 {len(retrieved_docs)}개의 관련 문서를 찾았습니다."})
                    else:
                        rag_messages = current_messages
                        current_messages.append({"role": "system", "content": "RAG 검색을 통해 관련 문서를 찾지 못했습니다."})

                except Exception as e:
                    st.error(f"Supabase 검색 실패: {e}")
                    current_messages.append({"role": "system", "content": "Supabase 문서 검색에 실패했습니다."})
                    rag_messages = current_messages
            else:
                st.warning("쿼리 임베딩 생성 실패로 RAG 검색을 수행할 수 없습니다.")
                rag_messages = current_messages
        else: # 일반 챗봇 모드 또는 RAG 챗봇 모드이지만 조건 불충족
            rag_messages = current_messages

        if deepseek_chat_client: # DeepSeek 채팅 클라이언트가 초기화된 경우
            for chunk in _deepseek_stream(rag_messages, deepseek_chat_client):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        else:
            # API 키가 없거나 클라이언트 초기화 실패 시 폴백
            for chunk in _fallback_stream(user_query):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    current_messages.append({"role": "assistant", "content": full_response})
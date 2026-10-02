import json
import os

import openai
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일의 환경변수 로드 (배포 환경에서는 파일이 없어도 무시됨)
load_dotenv()

DEFAULT_MODEL = "gpt-6-astra"
MAX_CHARS = 5000
# 5,000자 번역에 약 75초 소요 → 여유를 둔 요청당 대기 한도(초)
REQUEST_TIMEOUT = 120
# 이 글자 수를 넘으면 스피너에 오래 걸린다는 안내 표시
LONG_TEXT_CHARS = 1000

# 응답 JSON 키 → (화면 표시 이름, 프롬프트용 언어 이름)
LANGUAGES = {
    "english": ("🇺🇸 English", "English"),
    "japanese": ("🇯🇵 日本語", "Japanese"),
    "vietnamese": ("🇻🇳 Tiếng Việt", "Vietnamese"),
}


class TranslationParseError(Exception):
    """모델 응답이 올바른 JSON 형식이 아닐 때 발생"""


def get_config(key: str, default: str | None = None) -> str | None:
    """설정값 조회: 환경변수(.env) → st.secrets → 기본값 순서"""
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets[key]
    except Exception:
        # secrets.toml이 없거나 키가 없는 경우
        return default


def build_system_prompt(lang_keys: tuple[str, ...]) -> str:
    """선택한 언어만 번역하도록 시스템 프롬프트 생성"""
    names = ", ".join(LANGUAGES[k][1] for k in lang_keys)
    keys = ", ".join(f'"{k}"' for k in lang_keys)
    return f"""You are a professional translator.
Detect the source language automatically and translate the user's text into {names}.
Rules:
- Preserve the original meaning, nuance, and tone (formal/informal).
- Keep numbers, URLs, and emojis unchanged.
- Write proper nouns (names of people, places, companies) in the standard form used in each target language
  (e.g. 서울 → "Seoul" in English, "ソウル" in Japanese). Never leave the source-language script in the output.
- Use natural polite form for Japanese and correct tone marks for Vietnamese.
- Output ONLY a JSON object with the keys {keys}. Each value is the translated text only. No explanations."""


def parse_response(content: str | None, lang_keys: tuple[str, ...]) -> dict:
    """응답 JSON을 파싱하고 필요한 키가 모두 있는지 검사"""
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError as e:
        raise TranslationParseError("JSON 파싱 실패") from e
    if not isinstance(data, dict) or any(not isinstance(data.get(k), str) for k in lang_keys):
        raise TranslationParseError("필수 키 누락")
    return {k: data[k] for k in lang_keys}


# 캐시는 모든 사용자가 공유하므로 만료 시간과 최대 개수를 두어 메모리 증가를 막음
@st.cache_data(show_spinner=False, ttl=3600, max_entries=500)
def translate(text: str, lang_keys: tuple[str, ...], model: str) -> dict:
    """선택한 언어로 번역 (API 1회 호출, 파싱 실패 시 1회 재시도)

    인자 전체(텍스트, 언어, 모델명)가 캐시 키가 되며, 예외는 캐시되지 않음.
    """
    # SDK 자동 재시도는 1회로 줄여 최악의 대기 시간을 제한
    client = OpenAI(api_key=get_config("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT, max_retries=1)
    messages = [
        {"role": "system", "content": build_system_prompt(lang_keys)},
        {"role": "user", "content": text},
    ]
    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            # gpt-6-astra는 temperature 지정을 지원하지 않아(기본값 1만 허용) 생략
            response_format={"type": "json_object"},
            messages=messages,
        )
        try:
            return parse_response(response.choices[0].message.content, lang_keys)
        except TranslationParseError:
            if attempt == 1:
                raise


def error_message(error: Exception, model: str) -> str:
    """예외를 사용자 안내 문구로 변환 (PRD 8장)"""
    if isinstance(error, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다."
    if isinstance(error, openai.RateLimitError):
        return "요청이 많습니다. 잠시 후 다시 시도해주세요."
    if isinstance(error, openai.NotFoundError):
        return f"모델({model})을 사용할 수 없습니다. 모델명을 확인해주세요."
    # APITimeoutError는 APIConnectionError의 하위 클래스이므로 먼저 검사
    if isinstance(error, openai.APITimeoutError):
        return "응답 시간이 초과되었습니다. 글을 나누어 다시 시도해주세요."
    if isinstance(error, openai.APIConnectionError):
        return "네트워크 연결에 실패했습니다. 인터넷 연결을 확인해주세요."
    return "번역 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."


def inject_css() -> None:
    """기본 테마로 부족한 부분만 최소한으로 보정 (부제, 글자 수, 결과 카드)"""
    st.markdown(
        """
        <style>
        /* Windows에는 국기 이모지 글꼴이 없어 "US"처럼 문자로 보이므로 Noto Color Emoji로 보완 */
        /* 일본어는 Source Sans에 글리프가 없어 간격이 벌어지므로 Noto Sans JP로 보완 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&family=Noto+Sans+JP:wght@400;600&display=swap');
        .card-title, [data-testid="stSidebar"] label p {
            font-family: "Source Sans", "Source Sans Pro", "Noto Sans JP", "Noto Color Emoji", sans-serif;
        }
        /* 모바일 폭에서 제목이 한 글자씩 줄바꿈되지 않도록 크기 축소 */
        @media (max-width: 640px) {
            h1 { font-size: 1.9rem !important; }
        }
        .app-subtitle { color: #64748B; margin-top: -0.75rem; margin-bottom: 1.5rem; }
        .char-count { text-align: right; font-size: 0.85rem; color: #64748B; margin-top: -0.5rem; }
        .char-count.over { color: #DC2626; font-weight: 600; }
        .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
        /* 사이드바 펼침 여부에 따라 본문 폭이 달라지므로 화면 폭이 아닌 결과 영역 폭 기준으로,
           3열 카드가 너무 좁아지는(약 230px 미만) 720px 미만에서는 세로로 쌓음 */
        .st-key-results { container-type: inline-size; }
        @container (max-width: 720px) {
            .st-key-results [data-testid="stHorizontalBlock"] { flex-direction: column; }
            .st-key-results [data-testid="stColumn"] { width: 100% !important; flex: 1 1 auto !important; }
        }
        /* 번역문은 코드가 아니므로 고정폭 대신 본문 글꼴로 표시 */
        [data-testid="stCode"] code {
            font-family: "Source Sans", "Source Sans Pro", "Noto Sans JP", "Noto Color Emoji", sans-serif;
            font-size: 1.05rem; line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clear_all() -> None:
    """지우기 버튼 콜백: 입력과 결과 초기화 (위젯 생성 전에 실행되어야 하므로 콜백 사용)"""
    st.session_state["input_text"] = ""
    st.session_state.pop("results", None)


def render_sidebar(model: str) -> tuple[str, ...]:
    """사이드바: 언어 선택 체크박스와 사용 모델 표시, 선택된 언어 키 반환"""
    with st.sidebar:
        st.header("번역 언어")
        selected = tuple(
            key for key, (label, _) in LANGUAGES.items() if st.checkbox(label, value=True, key=f"lang_{key}")
        )
        st.divider()
        st.subheader("사용법")
        st.markdown(
            "1. 번역할 글을 입력하세요.\n"
            "2. **번역하기** 버튼을 누르세요.\n"
            "3. 결과 오른쪽 위 아이콘으로 복사하세요."
        )
        st.divider()
        st.caption(f"사용 모델: `{model}`")
    return selected


def render_results(results: dict) -> None:
    """번역 결과를 언어별 카드로 가로 배치 (모바일 폭에서는 세로로 쌓임)"""
    # key를 주면 st-key-results 클래스가 붙어 CSS 컨테이너 쿼리 대상이 됨
    with st.container(key="results"):
        columns = st.columns(len(results))
    for column, (key, text) in zip(columns, results.items()):
        with column, st.container(border=True):
            st.markdown(f'<div class="card-title">{LANGUAGES[key][0]}</div>', unsafe_allow_html=True)
            # st.code 우측 상단에 복사 버튼이 자동으로 제공됨
            st.code(text, language=None, wrap_lines=True)


def main() -> None:
    st.set_page_config(page_title="AI 다국어 번역기", page_icon="🌐", layout="wide")
    inject_css()

    st.title("🌐 AI 다국어 번역기")
    st.markdown(
        '<p class="app-subtitle">영어 · 일본어 · 베트남어로 한 번에 번역합니다</p>', unsafe_allow_html=True
    )

    model = get_config("OPENAI_MODEL", DEFAULT_MODEL)
    if not get_config("OPENAI_API_KEY"):
        st.error("API 키가 설정되지 않았습니다. .env 또는 Secrets를 확인해주세요.")
        st.stop()

    lang_keys = render_sidebar(model)

    text = st.text_area(
        "번역할 내용",
        placeholder="번역할 내용을 입력하세요...",
        height=200,
        key="input_text",
    )
    over_class = " over" if len(text) > MAX_CHARS else ""
    st.markdown(
        f'<div class="char-count{over_class}">{len(text):,} / {MAX_CHARS:,}자</div>', unsafe_allow_html=True
    )

    translate_col, clear_col = st.columns([4, 1])
    clicked = translate_col.button("🔄 번역하기", type="primary", width="stretch")
    clear_col.button("지우기", on_click=clear_all, width="stretch")

    if clicked:
        if not text.strip():
            st.warning("번역할 내용을 입력해주세요.")
        elif len(text) > MAX_CHARS:
            st.warning(f"최대 {MAX_CHARS:,}자까지 입력할 수 있습니다.")
        elif not lang_keys:
            st.warning("번역할 언어를 사이드바에서 하나 이상 선택해주세요.")
        else:
            spinner_text = "번역 중입니다..."
            if len(text) > LONG_TEXT_CHARS:
                spinner_text += " 긴 글은 1분 이상 걸릴 수 있습니다."
            with st.spinner(spinner_text):
                try:
                    st.session_state["results"] = translate(text, lang_keys, model)
                except Exception as e:
                    # 이전 결과가 이번 입력의 결과로 오해되지 않도록 제거
                    st.session_state.pop("results", None)
                    st.error(error_message(e, model))

    st.divider()
    # 리런(사이드바 조작 등) 후에도 마지막 결과 유지
    if "results" in st.session_state:
        render_results(st.session_state["results"])
    else:
        st.info("번역할 내용을 입력하고 **번역하기** 버튼을 누르면 여기에 결과가 표시됩니다.")


if __name__ == "__main__":
    main()

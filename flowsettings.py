import os
from importlib.metadata import version
from inspect import currentframe, getframeinfo
from pathlib import Path

from decouple import config
from ktem.utils.lang import SUPPORTED_LANGUAGE_MAP
from theflow.settings.default import *  # noqa

cur_frame = currentframe()
if cur_frame is None:
    raise ValueError("Cannot get the current frame.")
this_file = getframeinfo(cur_frame).filename
this_dir = Path(this_file).parent

# change this if your app use a different name
KH_PACKAGE_NAME = "traffic_agent"

# App display name (read by ktem.app.BaseApp)
KH_APP_NAME = "TrafficAgent"

KH_APP_VERSION = config("KH_APP_VERSION", None)
if not KH_APP_VERSION:
    try:
        # Caution: This might produce the wrong version
        # https://stackoverflow.com/a/59533071
        KH_APP_VERSION = version(KH_PACKAGE_NAME)
    except Exception:
        KH_APP_VERSION = "local"

KH_GRADIO_SHARE = config("KH_GRADIO_SHARE", default=False, cast=bool)
KH_ENABLE_FIRST_SETUP = config("KH_ENABLE_FIRST_SETUP", default=True, cast=bool)
KH_DEMO_MODE = config("KH_DEMO_MODE", default=False, cast=bool)
KH_OLLAMA_URL = config("KH_OLLAMA_URL", default="http://localhost:11434/v1/")

# App can be ran from anywhere and it's not trivial to decide where to store app data.
# So let's use the same directory as the flowsetting.py file.
KH_APP_DATA_DIR = this_dir / "ktem_app_data"
KH_APP_DATA_EXISTS = KH_APP_DATA_DIR.exists()
KH_APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# User data directory
KH_USER_DATA_DIR = KH_APP_DATA_DIR / "user_data"
KH_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# markdown output directory
KH_MARKDOWN_OUTPUT_DIR = KH_APP_DATA_DIR / "markdown_cache_dir"
KH_MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# chunks output directory
KH_CHUNKS_OUTPUT_DIR = KH_APP_DATA_DIR / "chunks_cache_dir"
KH_CHUNKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# zip output directory
KH_ZIP_OUTPUT_DIR = KH_APP_DATA_DIR / "zip_cache_dir"
KH_ZIP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# zip input directory
KH_ZIP_INPUT_DIR = KH_APP_DATA_DIR / "zip_cache_dir_in"
KH_ZIP_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# HF models can be big, let's store them in the app data directory so that it's easier
# for users to manage their storage.
# ref: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
os.environ["HF_HOME"] = str(KH_APP_DATA_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(KH_APP_DATA_DIR / "huggingface")
# HuggingFace mirror (China) so FastEmbed embeddings model can be downloaded
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Disable HuggingFace Xet large-file transfer (cas-bridge.xethub.hf.co is unreachable
# in China and times out). Falls back to standard mirror download.
os.environ["HF_HUB_DISABLE_XET"] = "1"
# FastEmbed cache + offline mode: the embedding model is pre-cached locally
# (downloaded via ModelScope because HF/Xet is blocked in this network), so force
# fastembed to read from the local cache and skip any network call.
os.environ["FASTEMBED_CACHE_PATH"] = str(KH_APP_DATA_DIR / "huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
# fast_langdetect (used by ktem for PDF preview language detection) downloads
# lid.176.bin and loads it with fasttext's C++ backend, which on Windows cannot
# open file paths containing non-ASCII characters (the user profile has a Chinese
# name). Point its cache to a pure-ASCII path so the model loads instead of
# spamming "Failed to load model using temporary file".
os.environ["FTLANG_CACHE"] = "D:/fasttext-langdetect"

# doc directory
KH_DOC_DIR = this_dir / "docs"

KH_MODE = "dev"
KH_SSO_ENABLED = config("KH_SSO_ENABLED", default=False, cast=bool)

KH_FEATURE_CHAT_SUGGESTION = config(
    "KH_FEATURE_CHAT_SUGGESTION", default=False, cast=bool
)
KH_FEATURE_USER_MANAGEMENT = config(
    "KH_FEATURE_USER_MANAGEMENT", default=True, cast=bool
)
KH_USER_CAN_SEE_PUBLIC = None
KH_FEATURE_USER_MANAGEMENT_ADMIN = str(
    config("KH_FEATURE_USER_MANAGEMENT_ADMIN", default="admin")
)
KH_FEATURE_USER_MANAGEMENT_PASSWORD = str(
    config("KH_FEATURE_USER_MANAGEMENT_PASSWORD", default="admin")
)
KH_ENABLE_ALEMBIC = False
KH_DATABASE = f"sqlite:///{KH_USER_DATA_DIR / 'sql.db'}"
KH_FILESTORAGE_PATH = str(KH_USER_DATA_DIR / "files")
KH_WEB_SEARCH_BACKEND = (
    "kotaemon.indices.retrievers.tavily_web_search.WebSearch"
    # "kotaemon.indices.retrievers.jina_web_search.WebSearch"
)

KH_DOCSTORE = {
    # "__type__": "kotaemon.storages.ElasticsearchDocumentStore",
    # "__type__": "kotaemon.storages.SimpleFileDocumentStore",
    "__type__": "kotaemon.storages.LanceDBDocumentStore",
    "path": str(KH_USER_DATA_DIR / "docstore"),
}
KH_VECTORSTORE = {
    # 说明：docstore 已用 LanceDB（持久化、Windows 友好）；vectorstore 保持
    # Chroma。尝试切换 LanceDB 向量库时发现 lancedb 0.25.x 与
    # llama-index-vector-stores-lancedb 0.1.7 的 schema 集成层不兼容
    # （pyarrow StructType.fields 报错），且降级 lancedb 会破坏已建的
    # LanceDB docstore。选型对比见 docs/02_retrieval_report.md。
    "__type__": "kotaemon.storages.ChromaVectorStore",
    # "__type__": "kotaemon.storages.LanceDBVectorStore",
    # "__type__": "kotaemon.storages.MilvusVectorStore",
    # "__type__": "kotaemon.storages.QdrantVectorStore",
    "path": str(KH_USER_DATA_DIR / "vectorstore"),
}
KH_LLMS = {}
KH_EMBEDDINGS = {}
KH_RERANKINGS = {}

# populate options from config
if config("AZURE_OPENAI_API_KEY", default="") and config(
    "AZURE_OPENAI_ENDPOINT", default=""
):
    if config("AZURE_OPENAI_CHAT_DEPLOYMENT", default=""):
        KH_LLMS["azure"] = {
            "spec": {
                "__type__": "kotaemon.llms.AzureChatOpenAI",
                "temperature": 0,
                "azure_endpoint": config("AZURE_OPENAI_ENDPOINT", default=""),
                "api_key": config("AZURE_OPENAI_API_KEY", default=""),
                "api_version": config("OPENAI_API_VERSION", default="")
                or "2024-02-15-preview",
                "azure_deployment": config("AZURE_OPENAI_CHAT_DEPLOYMENT", default=""),
                "timeout": 20,
            },
            "default": False,
        }
    if config("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", default=""):
        KH_EMBEDDINGS["azure"] = {
            "spec": {
                "__type__": "kotaemon.embeddings.AzureOpenAIEmbeddings",
                "azure_endpoint": config("AZURE_OPENAI_ENDPOINT", default=""),
                "api_key": config("AZURE_OPENAI_API_KEY", default=""),
                "api_version": config("OPENAI_API_VERSION", default="")
                or "2024-02-15-preview",
                "azure_deployment": config(
                    "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", default=""
                ),
                "timeout": 10,
            },
            "default": False,
        }

OPENAI_DEFAULT = "<YOUR_OPENAI_KEY>"
OPENAI_API_KEY = config("OPENAI_API_KEY", default=OPENAI_DEFAULT)
GOOGLE_API_KEY = config("GOOGLE_API_KEY", default="your-key")
IS_OPENAI_DEFAULT = len(OPENAI_API_KEY) > 0 and OPENAI_API_KEY != OPENAI_DEFAULT

if OPENAI_API_KEY:
    KH_LLMS["openai"] = {
        "spec": {
            "__type__": "kotaemon.llms.ChatOpenAI",
            "temperature": 0,
            "base_url": config("OPENAI_API_BASE", default="")
            or "https://api.openai.com/v1",
            "api_key": OPENAI_API_KEY,
            "model": config("OPENAI_CHAT_MODEL", default="deepseek-v4-flash"),
            "timeout": 20,
        },
        "default": IS_OPENAI_DEFAULT,
    }

# NOTE: DeepSeek (via OPENAI_*) has NO embeddings endpoint, so we intentionally do
# NOT register an OpenAI embeddings provider. Vectorization uses the local FastEmbed
# model registered below (KH_EMBEDDINGS["fast_embed"], default=True). This prevents
# the retriever from accidentally calling DeepSeek's missing /embeddings and 404-ing.

# Local embedding (no API key needed). DeepSeek has no embeddings endpoint,
# so vectorization uses a local FastEmbed multilingual model. This becomes the
# default retriever embedding; chat/LLM still goes to DeepSeek via OPENAI_*.
KH_EMBEDDINGS["fast_embed"] = {
    "spec": {
        "__type__": "kotaemon.embeddings.FastEmbedEmbeddings",
        "model_name": "BAAI/bge-small-zh-v1.5",
    },
    "default": True,
}

if config("LOCAL_MODEL", default=""):
    KH_LLMS["ollama"] = {
        "spec": {
            "__type__": "kotaemon.llms.ChatOpenAI",
            "base_url": KH_OLLAMA_URL,
            "model": config("LOCAL_MODEL", default="qwen2.5:7b"),
            "api_key": "ollama",
        },
        "default": False,
    }
    KH_LLMS["ollama-long-context"] = {
        "spec": {
            "__type__": "kotaemon.llms.LCOllamaChat",
            "base_url": KH_OLLAMA_URL.replace("v1/", ""),
            "model": config("LOCAL_MODEL", default="qwen2.5:7b"),
            "num_ctx": 8192,
        },
        "default": False,
    }

    KH_EMBEDDINGS["ollama"] = {
        "spec": {
            "__type__": "kotaemon.embeddings.OpenAIEmbeddings",
            "base_url": KH_OLLAMA_URL,
            "model": config("LOCAL_MODEL_EMBEDDINGS", default="nomic-embed-text"),
            "api_key": "ollama",
        },
        "default": False,
    }
    KH_EMBEDDINGS["fast_embed"] = {
        "spec": {
            "__type__": "kotaemon.embeddings.FastEmbedEmbeddings",
            "model_name": "BAAI/bge-base-en-v1.5",
        },
        "default": False,
    }

# additional LLM configurations
KH_LLMS["claude"] = {
    "spec": {
        "__type__": "kotaemon.llms.chats.LCAnthropicChat",
        "model_name": "claude-3-5-sonnet-20240620",
        "api_key": "your-key",
    },
    "default": False,
}
KH_LLMS["google"] = {
    "spec": {
        "__type__": "kotaemon.llms.chats.LCGeminiChat",
        "model_name": "gemini-1.5-flash",
        "api_key": GOOGLE_API_KEY,
    },
    "default": not IS_OPENAI_DEFAULT,
}
KH_LLMS["groq"] = {
    "spec": {
        "__type__": "kotaemon.llms.ChatOpenAI",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
        "api_key": "your-key",
    },
    "default": False,
}
KH_LLMS["cohere"] = {
    "spec": {
        "__type__": "kotaemon.llms.chats.LCCohereChat",
        "model_name": "command-r-plus-08-2024",
        "api_key": config("COHERE_API_KEY", default="your-key"),
    },
    "default": False,
}
KH_LLMS["mistral"] = {
    "spec": {
        "__type__": "kotaemon.llms.ChatOpenAI",
        "base_url": "https://api.mistral.ai/v1",
        "model": "ministral-8b-latest",
        "api_key": config("MISTRAL_API_KEY", default="your-key"),
    },
    "default": False,
}

# default reranking models
KH_RERANKINGS["cohere"] = {
    "spec": {
        "__type__": "kotaemon.rerankings.CohereReranking",
        "model_name": "rerank-v4.0-fast",
        "cohere_api_key": config("COHERE_API_KEY", default=""),
    },
    "default": False,
}

# 本地 bge-reranker（无需 API key）。默认关闭，启用后作为检索链路的默认重排器。
# 需本地安装 sentence-transformers + torch，并联网下载 BAAI/bge-reranker-base
# 权重（走 HF_ENDPOINT 镜像）。加载失败时 BgeReranking 内部会降级为保持原顺序，
# 不会破坏检索链路。
KH_RERANKINGS["bge"] = {
    "spec": {
        "__type__": "kotaemon.rerankings.BgeReranking",
        "model_name": config(
            "BGE_RERANKER_MODEL", default="BAAI/bge-reranker-base"
        ),
    },
    "default": False,
}

# 检索层开关（Phase 2 引入）：控制文档检索是否启用「混合检索 + 重排」两阶段。
# USE_HYBRID_RETRIEVAL：是否用 BM25+向量 RRF 混合检索（关闭则退回原向量检索）
# USE_RERANKING：是否在检索后追加 bge-reranker 重排
KH_USE_HYBRID_RETRIEVAL = config("USE_HYBRID_RETRIEVAL", default=True, cast=bool)
KH_USE_RERANKING = config("USE_RERANKING", default=True, cast=bool)

KH_REASONINGS = [
    "ktem.reasoning.simple.FullQAPipeline",
    "ktem.reasoning.simple.FullDecomposeQAPipeline",
]
KH_REASONINGS_USE_MULTIMODAL = config("USE_MULTIMODAL", default=False, cast=bool)

# 领域化提示词开关：开启后，Simple QA / Complex QA 管线使用
# libs/ktem/ktem/prompts/traffic.py 中的运维场景 prompt（要求条款号+原文引用+出处）。
# 关闭则退回 kotaemon 默认 prompt。
KH_REASONING_USE_TRAFFIC_PROMPT = config(
    "USE_TRAFFIC_PROMPT", default=True, cast=bool
)

# ---------------------------------------------------------------------------
# 双层缓存（Phase 4 引入）：Embedding 结果缓存 + LLM 响应缓存。
# USE_CACHE：总开关，关闭则两层缓存全部直通底层（用于 A/B 对比延迟）。
# USE_EMBEDDING_CACHE / USE_LLM_CACHE：分层开关。
# EMBEDDING_CACHE_TTL / LLM_CACHE_TTL：缓存有效期（秒），0 表示永不过期。
# 缓存目录默认 KH_APP_DATA_DIR / cache / {embedding_cache, llm_cache}。
# ---------------------------------------------------------------------------
USE_CACHE = config("USE_CACHE", default=True, cast=bool)
USE_EMBEDDING_CACHE = config("USE_EMBEDDING_CACHE", default=True, cast=bool)
USE_LLM_CACHE = config("USE_LLM_CACHE", default=True, cast=bool)
EMBEDDING_CACHE_TTL = config("EMBEDDING_CACHE_TTL", default=0, cast=int)
LLM_CACHE_TTL = config("LLM_CACHE_TTL", default=3600, cast=int)

# 结构化日志开关与目录（Phase 4 引入）
USE_STRUCTLOG = config("USE_STRUCTLOG", default=False, cast=bool)
KH_LOG_DIR = this_dir / "logs"

KH_VLM_ENDPOINT = "{0}/openai/deployments/{1}/chat/completions?api-version={2}".format(
    config("AZURE_OPENAI_ENDPOINT", default=""),
    config("OPENAI_VISION_DEPLOYMENT_NAME", default="gpt-4o"),
    config("OPENAI_API_VERSION", default=""),
)


SETTINGS_APP: dict[str, dict] = {}


SETTINGS_REASONING = {
    "use": {
        "name": "Reasoning options",
        "value": None,
        "choices": [],
        "component": "radio",
    },
    "lang": {
        "name": "Language",
        "value": "en",
        "choices": [(lang, code) for code, lang in SUPPORTED_LANGUAGE_MAP.items()],
        "component": "dropdown",
    },
    "max_context_length": {
        "name": "Max context length (LLM)",
        "value": 32000,
        "component": "number",
    },
}

USE_GLOBAL_GRAPHRAG = config("USE_GLOBAL_GRAPHRAG", default=True, cast=bool)
USE_NANO_GRAPHRAG = config("USE_NANO_GRAPHRAG", default=False, cast=bool)
USE_LIGHTRAG = config("USE_LIGHTRAG", default=True, cast=bool)
USE_MS_GRAPHRAG = config("USE_MS_GRAPHRAG", default=True, cast=bool)

GRAPHRAG_INDEX_TYPES = []

if USE_MS_GRAPHRAG:
    GRAPHRAG_INDEX_TYPES.append("ktem.index.file.graph.GraphRAGIndex")
if USE_NANO_GRAPHRAG:
    GRAPHRAG_INDEX_TYPES.append("ktem.index.file.graph.NanoGraphRAGIndex")
if USE_LIGHTRAG:
    GRAPHRAG_INDEX_TYPES.append("ktem.index.file.graph.LightRAGIndex")

KH_INDEX_TYPES = [
    "ktem.index.file.FileIndex",
    *GRAPHRAG_INDEX_TYPES,
]

GRAPHRAG_INDICES = [
    {
        "name": graph_type.split(".")[-1].replace("Index", "")
        + " Collection",  # get last name
        "config": {
            "supported_file_types": (
                ".png, .jpeg, .jpg, .tiff, .tif, .pdf, .xls, .xlsx, .doc, .docx, "
                ".pptx, .csv, .html, .mhtml, .txt, .md, .zip"
            ),
            "private": True,
        },
        "index_type": graph_type,
    }
    for graph_type in GRAPHRAG_INDEX_TYPES
]

KH_INDICES = [
    {
        "name": "File Collection",
        "config": {
            "supported_file_types": (
                ".png, .jpeg, .jpg, .tiff, .tif, .pdf, .xls, .xlsx, .doc, .docx, "
                ".pptx, .csv, .html, .mhtml, .txt, .md, .zip"
            ),
            "private": True,
        },
        "index_type": "ktem.index.file.FileIndex",
    },
    *GRAPHRAG_INDICES,
]


# ---------------------------------------------------------------------------
# Force a default LOCAL embedding (no API key needed).
#
# The `if config("LOCAL_MODEL", default="")` block above can redefine
# KH_EMBEDDINGS["fast_embed"] with default=False (and a different model), and
# when an OpenAI/DeepSeek chat key is also set, no embedding ends up marked
# default. That leaves the EmbeddingManager with no `default` entry, which makes
# indexing/retrieval raise `KeyError: 'default'`. Pin fast_embed as the default
# local model here so retrieval always works regardless of LOCAL_MODEL.
# ---------------------------------------------------------------------------
if "fast_embed" in KH_EMBEDDINGS:
    KH_EMBEDDINGS["fast_embed"]["spec"]["model_name"] = "BAAI/bge-small-zh-v1.5"
    KH_EMBEDDINGS["fast_embed"]["default"] = True

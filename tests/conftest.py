"""pytest 公共配置：仓库路径、语料与 Mock 后端 fixture。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 真实模型测试需要 .env 里的密钥（没有也无妨：相关测试自动跳过）
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

from src.corpus import Corpus  # noqa: E402
from src.llm import MockBackend  # noqa: E402


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus(REPO_ROOT / "data" / "docs")


@pytest.fixture(scope="session")
def corpus_alt() -> Corpus:
    return Corpus(REPO_ROOT / "examples" / "corpus_alt")


@pytest.fixture()
def mock_backend(corpus) -> MockBackend:
    return MockBackend(corpus)

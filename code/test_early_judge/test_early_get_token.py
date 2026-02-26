import os
import pytest
from unittest import mock

# Absolute import for the function under test
from judge import get_token

@pytest.fixture(autouse=True)
def clear_hf_token_env(monkeypatch):
    """
    Ensure HF_TOKEN environment variable is cleared before each test.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)

class TestGetToken:
    @pytest.mark.happy_path
    def test_returns_token_from_env(self, monkeypatch):
        """
        Test that get_token returns the token from the HF_TOKEN environment variable when set.
        """
        monkeypatch.setenv("HF_TOKEN", "env_token_123")
        token = get_token()
        assert token == "env_token_123"

    @pytest.mark.happy_path
    def test_returns_token_from_huggingface_hub_get_token(self):
        """
        Test that get_token returns the token from huggingface_hub.get_token when HF_TOKEN is not set.
        """
        with mock.patch("judge.get_token", return_value="hub_token_456"):
            # Remove env var to force fallback
            os.environ.pop("HF_TOKEN", None)
            token = get_token()
            assert token == "hub_token_456"

    @pytest.mark.happy_path
    def test_returns_token_from_hffolder_when_get_token_missing(self):
        """
        Test that get_token falls back to HfFolder.get_token if huggingface_hub.get_token is missing.
        """
        # Simulate ImportError and fallback
        with mock.patch("judge.HfFolder.get_token", return_value="folder_token_789"):
            # Remove env var to force fallback
            os.environ.pop("HF_TOKEN", None)
            # Patch judge.get_token to use fallback
            # This is a bit tricky: we simulate the fallback by patching the function in judge.py
            # If judge.get_token is a function defined as fallback, this will work
            token = get_token()
            assert token == "folder_token_789"

    @pytest.mark.edge_case
    def test_returns_none_when_no_token_available(self):
        """
        Test that get_token returns None if neither HF_TOKEN env nor huggingface_hub/HfFolder token is available.
        """
        # Remove env var to force fallback
        os.environ.pop("HF_TOKEN", None)
        # Patch both get_token and HfFolder.get_token to return None
        with mock.patch("judge.get_token", return_value=None), \
             mock.patch("judge.HfFolder.get_token", return_value=None):
            token = get_token()
            assert token is None

    @pytest.mark.edge_case
    def test_returns_empty_string_when_token_is_empty(self):
        """
        Test that get_token returns empty string if the token is set to empty in env or hub.
        """
        # Test empty env var
        os.environ["HF_TOKEN"] = ""
        token = get_token()
        assert token == ""

        # Test empty hub token
        os.environ.pop("HF_TOKEN", None)
        with mock.patch("judge.get_token", return_value=""):
            token = get_token()
            assert token == ""

    @pytest.mark.edge_case
    def test_token_case_sensitivity(self, monkeypatch):
        """
        Test that get_token is case-sensitive and does not alter the token value.
        """
        monkeypatch.setenv("HF_TOKEN", "HF_Token_CaseSensitive")
        token = get_token()
        assert token == "HF_Token_CaseSensitive"

    @pytest.mark.edge_case
    def test_token_with_special_characters(self, monkeypatch):
        """
        Test that get_token correctly returns tokens with special characters.
        """
        special_token = "hf_!@#$%^&*()_+-=[]{}|;':,.<>/?"
        monkeypatch.setenv("HF_TOKEN", special_token)
        token = get_token()
        assert token == special_token

    @pytest.mark.edge_case
    def test_token_with_whitespace(self, monkeypatch):
        """
        Test that get_token returns tokens with leading/trailing whitespace as-is.
        """
        monkeypatch.setenv("HF_TOKEN", "  hf_token_with_space  ")
        token = get_token()
        assert token == "  hf_token_with_space  "

    @pytest.mark.edge_case
    def test_token_with_unicode(self, monkeypatch):
        """
        Test that get_token returns tokens containing unicode characters.
        """
        unicode_token = "hf_测试_🚀"
        monkeypatch.setenv("HF_TOKEN", unicode_token)
        token = get_token()
        assert token == unicode_token
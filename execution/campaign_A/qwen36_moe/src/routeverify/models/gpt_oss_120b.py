"""120B identity over the existing native GPT-OSS MXFP4 implementation."""

from routeverify.models.gpt_oss import GptOssAdapter


class GptOss120BAdapter(GptOssAdapter):
    alias = "gpt_oss_120b"
    model_id = "openai/gpt-oss-120b"

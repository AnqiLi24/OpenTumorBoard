import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

_API_VERSION = "2024-12-01-preview"
_AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
_MANAGED_IDENTITY_CLIENT_ID = os.environ.get("AZURE_CLIENT_ID")

_token_provider = get_bearer_token_provider(
    DefaultAzureCredential(managed_identity_client_id=_MANAGED_IDENTITY_CLIENT_ID),
    "https://cognitiveservices.azure.com/.default",
)

client = AzureOpenAI(
    api_version=_API_VERSION,
    azure_endpoint=_AZURE_ENDPOINT,
    azure_ad_token_provider=_token_provider,
)

DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

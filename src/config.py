from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central place for all env vars. pydantic-settings validates these
    at import time — if a required key is missing, we fail loudly here
    instead of getting a silent None deep inside a tool call later.
    (Same lesson learned the hard way on the Imoth bot.)
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GROQ_API_KEY: str
    SPOONACULAR_API_KEY: str
    TAVILY_API_KEY: str
    LANGSMITH_API_KEY: str
    POSTGRES_URI: str = ""
    DEPLOYMENT_MODE: str = ""  # "dev" | "studio" | "production"


settings = Settings()
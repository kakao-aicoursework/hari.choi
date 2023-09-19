import pynecone as pc

class KakaoHelperBotConfig(pc.Config):
    pass

config = KakaoHelperBotConfig(
    app_name="kakao_helper_bot",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)
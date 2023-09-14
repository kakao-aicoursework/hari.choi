import pynecone as pc

class HarichatConfig(pc.Config):
    pass

config = HarichatConfig(
    app_name="harichat",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
)
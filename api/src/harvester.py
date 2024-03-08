import os

from dotenv import load_dotenv
from realtime.connection import Socket

load_dotenv()

SUPABASE_ID = os.getenv("SUPABASE_ID")
API_KEY = os.getenv("SUPABASE_API_KEY")


def derive_facts(payload):
    print("Derive Facts: ", payload)


if __name__ == "__main__":
    URL = f"wss://{SUPABASE_ID}.supabase.co/realtime/v1/websocket?apikey={API_KEY}&vsn=1.0.0"
    s = Socket(URL)
    s.connect()

    channel = s.set_channel("realtime:public:messages")
    channel.join().on("INSERT", derive_facts)
    s.listen()

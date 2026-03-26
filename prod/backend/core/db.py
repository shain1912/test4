from supabase import create_client, Client
from .config import settings

# Initialize Supabase client
def get_supabase() -> Client:
    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the .env file")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

supabase = get_supabase() if settings.SUPABASE_URL and settings.SUPABASE_KEY else None

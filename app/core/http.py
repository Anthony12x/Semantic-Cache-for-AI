import aiohttp


class HttpClientManager:
    """Manages a shared aiohttp.ClientSession for the app lifecycle."""

    _session: aiohttp.ClientSession | None = None

    @classmethod
    def get_session(cls) -> aiohttp.ClientSession:
        if cls._session is None or cls._session.closed:
            raise RuntimeError("HTTP Client session is not initialized.")
        return cls._session

    @classmethod
    async def start(cls):
        if cls._session is None or cls._session.closed:
            cls._session = aiohttp.ClientSession()

    @classmethod
    async def stop(cls):
        if cls._session and not cls._session.closed:
            await cls._session.close()
            cls._session = None

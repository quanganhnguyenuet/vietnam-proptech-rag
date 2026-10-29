"""Crawler HTTP session using verified TLS with the operating system CA store."""
import ssl

import requests
import truststore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SystemTLSAdapter(HTTPAdapter):
    def __init__(self, **kwargs):
        self._ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        super().__init__(**kwargs)

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host, pool = super().build_connection_pool_key_attributes(request, verify, cert)
        if verify is True:
            pool["ssl_context"] = self._ssl_context
        return host, pool


def create_session(headers: dict) -> requests.Session:
    session = requests.Session()
    session.headers.update(headers)
    session.mount("https://", SystemTLSAdapter(max_retries=Retry(
        total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )))
    return session

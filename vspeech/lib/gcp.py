from typing import TypeAlias

from google.auth import default as google_auth_default
from google.auth.compute_engine import Credentials as CeCredentials
from google.auth.compute_engine import IDTokenCredentials as CeIdTokenCredentials
from google.auth.exceptions import TransportError
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google.oauth2.service_account import IDTokenCredentials

from vspeech.config import GcpConfig
from vspeech.config import ServiceAccountInfo

GcpIDTokenCredentials: TypeAlias = IDTokenCredentials | CeIdTokenCredentials


def unescape_private_key(service_account_info: ServiceAccountInfo):
    decoded = {k: v.get_secret_value() for k, v in service_account_info.items()}
    if "private_key" in service_account_info:
        return {
            **decoded,
            "private_key": decoded["private_key"].replace("\\n", "\n"),
        }
    return decoded


def get_credentials(config: GcpConfig) -> tuple[Credentials | CeCredentials, str]:
    if config.service_account_file_path:
        file_path = config.service_account_file_path.expanduser()
        cred = Credentials.from_service_account_file(file_path)
        return cred, cred.project_id  # type: ignore
    elif config.service_account_info:
        decoded = unescape_private_key(config.service_account_info)
        cred = Credentials.from_service_account_info(decoded)
        return cred, cred.project_id  # type: ignore
    elif config.use_ce_credentials:
        cred = CeCredentials()
        return cred, ""
    else:
        cred, project_id = google_auth_default()
        return cred, project_id or ""


def get_id_token_credentials(
    config: GcpConfig,
) -> GcpIDTokenCredentials | None:
    if config.service_account_file_path:
        file_path = config.service_account_file_path.expanduser()
        return IDTokenCredentials.from_service_account_file(
            filename=file_path, target_audience=""
        )
    elif config.service_account_info:
        decoded = unescape_private_key(config.service_account_info)
        return IDTokenCredentials.from_service_account_info(
            info=decoded, target_audience=""
        )
    else:
        try:
            return CeIdTokenCredentials(request=Request(), target_audience="")
        except TransportError:
            return None

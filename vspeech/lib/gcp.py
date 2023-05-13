from typing import TypeAlias

from google.auth.compute_engine import Credentials as CeCredentials
from google.auth.compute_engine import IDTokenCredentials as CeIdTokenCredentials
from google.auth.transport import Request
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


def get_credentials(config: GcpConfig) -> Credentials | CeCredentials:
    if config.service_account_file_path:
        file_path = config.service_account_file_path.expanduser()
        return Credentials.from_service_account_file(file_path)
    elif config.service_account_info:
        decoded = unescape_private_key(config.service_account_info)
        return Credentials.from_service_account_info(decoded)
    else:
        return CeCredentials()


def get_id_token_credentials(
    config: GcpConfig,
) -> GcpIDTokenCredentials:
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
        return CeIdTokenCredentials(request=Request(), target_audience="")

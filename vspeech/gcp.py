from google.oauth2.service_account import Credentials
from google.oauth2.service_account import IDTokenCredentials

from vspeech.config import GcpConfig


def unescape_private_key(service_account_info: dict[str, str]):
    if "private_key" in service_account_info:
        return {
            **service_account_info,
            "private_key": service_account_info["private_key"].replace("\\n", "\n"),
        }
    return service_account_info


def get_credentials(config: GcpConfig) -> Credentials:
    if config.service_account_file_path:
        return Credentials.from_service_account_file(config.service_account_file_path)
    else:
        config.service_account_info = unescape_private_key(config.service_account_info)
        return Credentials.from_service_account_info(config.service_account_info)


def get_id_token_credentials(config: GcpConfig) -> IDTokenCredentials:
    if config.service_account_file_path:
        return IDTokenCredentials.from_service_account_file(
            config.service_account_file_path, target_audience=""
        )
    else:
        config.service_account_info = unescape_private_key(config.service_account_info)
        return IDTokenCredentials.from_service_account_info(
            config.service_account_info, target_audience=""
        )

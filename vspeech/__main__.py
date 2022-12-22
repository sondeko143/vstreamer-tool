from pydantic_cli import run_and_exit

from vspeech.config import Config
from vspeech.main import cmd

if __name__ == "__main__":
    run_and_exit(Config, cmd, description=__doc__)

"""Create a local demo .env with unique secrets; never overwrite existing configuration."""
from pathlib import Path
import secrets


def main():
    root = Path(__file__).resolve().parents[1]
    template = (root / ".env.example").read_text(encoding="utf-8")
    template = template.replace("POSTGRES_PASSWORD=\n", f"POSTGRES_PASSWORD={secrets.token_hex(24)}\n")
    template = template.replace("SECRET_KEY=\n", f"SECRET_KEY={secrets.token_urlsafe(48)}\n")
    with (root / ".env").open("x", encoding="utf-8", newline="\n") as output:
        output.write(template)
    print("Created .env for an isolated synthetic demo. Keep this file private.")


if __name__ == "__main__":
    main()

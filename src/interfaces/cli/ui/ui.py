"""Console UI helpers for consistent output."""

BANNER_WIDTH = 60


def header(title: str, width: int = BANNER_WIDTH) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def subheader(title: str, width: int = BANNER_WIDTH) -> None:
    print("\n" + title)
    print("-" * width)


def pause(prompt: str = "\nPress Enter to continue...") -> None:
    input(prompt)


def info(message: str) -> None:
    print(f"\n[INFO] {message}")


def warn(message: str) -> None:
    print(f"\n[!] {message}")


def error(message: str) -> None:
    print(f"\n[ERROR] {message}")

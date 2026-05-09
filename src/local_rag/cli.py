"""Terminal arayüzü — `python -m local_rag` veya `local-rag` ile çalıştırılır."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .rag_pipeline import RAGPipeline

console = Console()


def main() -> int:
    console.print(
        Panel.fit(
            "[bold cyan]Local RAG[/bold cyan]\n"
            "Kişisel hafıza destekli yerel chatbot\n"
            "[dim]çıkış: /exit  |  hafıza: /memory  |  yardım: /help[/dim]",
            border_style="cyan",
        )
    )

    pipe = RAGPipeline()

    while True:
        try:
            user_input = Prompt.ask("[bold green]sen[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]görüşürüz![/dim]")
            return 0

        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            console.print("[dim]görüşürüz![/dim]")
            return 0

        if user_input == "/help":
            _print_help()
            continue

        if user_input == "/memory":
            console.print(f"[dim]hafıza kayıt sayısı: {pipe.memory.count()}[/dim]")
            continue

        # Soru-cevap
        try:
            result = pipe.ask(user_input)
        except Exception as e:  # pragma: no cover
            console.print(f"[red]hata:[/red] {e}")
            continue

        console.print(f"[bold magenta]asistan[/bold magenta]: {result.answer}")
        if result.used_memories:
            console.print(
                f"[dim]({len(result.used_memories)} anı kullanıldı)[/dim]"
            )


def _print_help() -> None:
    console.print(
        "[bold]Komutlar:[/bold]\n"
        "  /exit, /quit  — çıkış\n"
        "  /memory       — kayıt sayısını göster\n"
        "  /help         — bu yardım"
    )


if __name__ == "__main__":
    sys.exit(main())

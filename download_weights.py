#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baixa o checkpoint vg_sgg.pth (Pix2Grp / PGSG-CVPR2024) para a pasta weights/.

Uso:
    python download_weights.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request

URL = (
    "https://huggingface.co/rj979797/PGSG-CVPR2024/resolve/main/vg_sgg.pth"
)
DEST_DIR = Path("weights")
DEST_FILE = DEST_DIR / "vg_sgg.pth"
CHUNK = 16_384  # 16 KB


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Calcula SHA-256 do arquivo (1 MiB por vez)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # HEAD request para descobrir tamanho (opcional)
    size_remote = int(
        urlopen(Request(URL, method="HEAD")).headers["Content-Length"]
    )

    if DEST_FILE.exists() and DEST_FILE.stat().st_size == size_remote:
        print(f"[✓] '{DEST_FILE}' já existe ({size_remote/1e6:.2f} MB).")
        return

    print(f"↓ Baixando {size_remote/1e6:.2f} MB → {DEST_FILE} …")
    with urlopen(URL) as resp, DEST_FILE.open("wb") as out:
        downloaded = 0
        while data := resp.read(CHUNK):
            out.write(data)
            downloaded += len(data)
            # barra de progresso simples
            done = int(50 * downloaded / size_remote)
            bar = f"[{'=' * done}{'.' * (50 - done)}]"
            percent = 100 * downloaded / size_remote
            sys.stdout.write(f"\r{bar} {percent:6.2f}%")
            sys.stdout.flush()

    print("\n[✓] Download concluído.")

    # (Opcional) checksum rápido
    print("SHA-256:", sha256(DEST_FILE)[:16], "…")


if __name__ == "__main__":
    try:
        download()
    except KeyboardInterrupt:
        if DEST_FILE.exists():
            os.remove(DEST_FILE)
        print("\nDownload cancelado.")

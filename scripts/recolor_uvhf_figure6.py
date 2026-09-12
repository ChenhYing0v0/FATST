"""Recolor scope identities without changing PDF geometry or data."""

import argparse
import re
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject


PALETTE = {
    "405F73": "47799B",  # Scope 1: blue.
    "648395": "40877E",  # Scope 48: teal.
    "91ABB5": "A97940",  # Scope 144: ochre.
    "9188AD": "8570A3",  # Scope 360: violet.
    "B47D96": "B36985",  # Scope 720: rose.
}


def rgb(value: str) -> tuple[float, ...]:
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = PdfReader(args.source)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    replacements = 0
    for page in writer.pages:
        def replace_color(match: re.Match[bytes]) -> bytes:
            nonlocal replacements
            operands = match.group(1).split()
            for old, new in PALETTE.items():
                if all(abs(float(a) - b) < 1e-8
                       for a, b in zip(operands, rgb(old))):
                    replacements += 1
                    return (" ".join(f"{v:.10f}" for v in rgb(new)).encode()
                            + b" " + match.group(2))
            return match.group()

        data = page.get_contents().get_data()
        updated = re.sub(rb"([0-9.]+\s+[0-9.]+\s+[0-9.]+)\s+(rg|RG)\b",
                         replace_color, data)
        stream = DecodedStreamObject()
        stream.set_data(updated)
        page[NameObject("/Contents")] = writer._add_object(stream)
    if replacements != 64:
        raise ValueError(f"Unexpected source palette: {replacements} operations")
    with args.output.open("wb") as output:
        writer.write(output)


if __name__ == "__main__":
    main()

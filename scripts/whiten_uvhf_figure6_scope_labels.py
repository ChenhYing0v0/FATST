"""Make the two scope-144 strip labels white without changing geometry."""

import argparse
import re
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    writer = PdfWriter()
    writer.clone_document_from_reader(PdfReader(args.source))
    page = writer.pages[0]
    data = page.get_contents().get_data()
    color = (rb"0\.1490196078 0\.168627451 0\.1882352941 rg "
             rb"0\.1490196078 0\.168627451\n0\.1882352941 RG "
             rb"0\.1490196078 0\.168627451 0\.1882352941 rg")
    label = (rb"(\nq\n1 0 -0 1 [0-9.]+ 235\.8311387104 cm\nBT\n"
             rb"/F4 5\.2 Tf\n0 0 Td\n\[ \(\x001\x004\x004\) \] TJ\nET\nQ)")
    updated, count = re.subn(color + label, rb"1 g 1 G 1 g\1", data)
    if count != 2:
        raise ValueError(f"Expected two scope-144 labels, found {count}")
    stream = DecodedStreamObject()
    stream.set_data(updated)
    page[NameObject("/Contents")] = writer._add_object(stream)
    with args.output.open("wb") as output:
        writer.write(output)


if __name__ == "__main__":
    main()

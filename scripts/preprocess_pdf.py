"""
PDF Pre-processor - Extract tables as structured data
This improves RAG accuracy for table-heavy documents
"""
import pdfplumber
import json
import sys
from pathlib import Path

def extract_tables_from_pdf(pdf_path: str, output_dir: str = None):
    """
    Extract tables from PDF and save as structured JSON

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted data (default: same as PDF)
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir) if output_dir else pdf_path.parent
    output_dir.mkdir(exist_ok=True)

    filename = pdf_path.stem
    all_tables = []
    all_text = []

    print(f"📄 Processing: {pdf_path.name}")

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            print(f"  Page {page_num}/{len(pdf.pages)}...")

            # Extract tables from this page
            tables = page.extract_tables()

            if tables:
                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:  # Has header + data
                        # Convert to structured format
                        headers = table[0] if table[0] else [f"Col{i}" for i in range(len(table[1]))]
                        headers = [h.strip() if h else f"Col{i}" for i, h in enumerate(headers)]

                        rows = []
                        for row in table[1:]:
                            if row and any(cell for cell in row):  # Non-empty row
                                row_dict = {}
                                for i, cell in enumerate(row):
                                    if i < len(headers):
                                        row_dict[headers[i]] = (cell.strip() if cell else "")
                                rows.append(row_dict)

                        if rows:
                            table_data = {
                                "page": page_num,
                                "table_index": table_idx + 1,
                                "headers": headers,
                                "rows": rows,
                                "row_count": len(rows)
                            }
                            all_tables.append(table_data)
                            print(f"    ✓ Table {table_idx + 1}: {len(rows)} rows, {len(headers)} columns")

            # Also extract plain text (for non-table content)
            text = page.extract_text()
            if text:
                all_text.append({
                    "page": page_num,
                    "content": text.strip()
                })

    # Save extracted tables
    tables_file = output_dir / f"{filename}_tables.json"
    with open(tables_file, 'w', encoding='utf-8') as f:
        json.dump(all_tables, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved {len(all_tables)} tables to: {tables_file}")

    # Save plain text
    text_file = output_dir / f"{filename}_text.json"
    with open(text_file, 'w', encoding='utf-8') as f:
        json.dump(all_text, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved text from {len(all_text)} pages to: {text_file}")

    # Generate markdown version of tables (easier for LLM to read)
    md_file = output_dir / f"{filename}_tables.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        for table in all_tables:
            f.write(f"\n## Table from Page {table['page']}\n\n")

            # Write headers
            f.write("| " + " | ".join(table['headers']) + " |\n")
            f.write("| " + " | ".join(["---"] * len(table['headers'])) + " |\n")

            # Write rows
            for row in table['rows']:
                cells = [str(row.get(h, "")) for h in table['headers']]
                f.write("| " + " | ".join(cells) + " |\n")

            f.write("\n")
    print(f"✅ Saved markdown tables to: {md_file}")

    return {
        "tables_count": len(all_tables),
        "pages_count": len(all_text),
        "tables_file": str(tables_file),
        "text_file": str(text_file),
        "markdown_file": str(md_file)
    }


def process_directory(input_dir: str, output_dir: str = None):
    """Process all PDFs in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "extracted"

    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")

    results = []
    for pdf_file in pdf_files:
        try:
            result = extract_tables_from_pdf(pdf_file, output_dir)
            results.append({"file": pdf_file.name, **result})
        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            results.append({"file": pdf_file.name, "error": str(e)})
        print()

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python preprocess_pdf.py <pdf_file>")
        print("  python preprocess_pdf.py <directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        extract_tables_from_pdf(path)
    elif path.is_dir():
        process_directory(path)
    else:
        print(f"❌ Path not found: {path}")
        sys.exit(1)

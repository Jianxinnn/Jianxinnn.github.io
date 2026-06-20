from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MINERU_ROOT = Path(
    "/Users/jxtang/MinerU/lecture_notes.pdf-afa87b40-318d-4443-9656-9fb8362ad1c4"
)
PDF_PATH = Path("/Users/jxtang/Downloads/lecture_notes.pdf")
SOURCE_MD = MINERU_ROOT / "full.md"
CONTENT_LIST = MINERU_ROOT / "c5150c96-7695-4b51-840d-5785245e5b84_content_list.json"
SOURCE_IMAGES = MINERU_ROOT / "images"
SLUG = "flow-matching-diffusion-models-cn"
POST_ROOT = REPO_ROOT / "content" / "blog" / "posts" / SLUG
PUBLIC_IMAGES = REPO_ROOT / "public" / "blog" / SLUG / "images"
SEGMENTS_TS = POST_ROOT / "segments.ts"
META_TS = POST_ROOT / "meta.ts"

ORIGINAL_TITLE = "An Introduction to Flow Matching and Diffusion Models"
ORIGINAL_URL = "https://diffusion.csail.mit.edu/"
EXPECTED_CONTENT_TYPE_COUNTS = {
    "equation": 293,
    "image": 34,
    "chart": 16,
}
REQUIRED_PDF_PHRASES = [
    ORIGINAL_TITLE,
    "Website: https://diffusion.csail.mit.edu/",
    "1 Introduction",
    "2 Flow and Diffusion Models",
    "3 Flow Matching",
    "4 Score Functions and Score Matching",
    "4.3 Score Matching",
    "5 Guidance: How To Condition on a Prompt",
    "5.2 Classifer-Free Guidance",
    "6 Building Large-Scale Image or Video Generators",
    "6.3 Case Study: Stable Diffusion 3 and Meta Movie Gen",
    "7 Discrete Diffusion Models: Building Language Models with Diffusion",
    "8 References",
    "A A Reminder on Probability Theory",
    "E A Guide to the Diffusion Model Literature",
]
EXPECTED_FIGURE_NUMBERS = set(range(1, 23))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def strip_mineru_details(text: str) -> str:
    return re.sub(r"\n?<details>\n<summary>.*?</summary>\n\n.*?\n</details>\n?", "\n", text, flags=re.S)


def split_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if line.startswith("```"):
            start = i
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                i += 1
            if i < len(lines):
                i += 1
            blocks.append("\n".join(lines[start:i]))
            continue

        if line.strip().startswith("$$"):
            start = i
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"):
                i += 1
            if i < len(lines):
                i += 1
            blocks.append("\n".join(lines[start:i]))
            continue

        if line.startswith("## "):
            blocks.append(line)
            i += 1
            continue

        if line.startswith("![]("):
            blocks.append(line)
            i += 1
            continue

        start = i
        i += 1
        while i < len(lines):
            next_line = lines[i]
            if (
                not next_line.strip()
                or next_line.startswith("## ")
                or next_line.startswith("```")
                or next_line.strip().startswith("$$")
                or next_line.startswith("![](")
            ):
                break
            i += 1
        blocks.append("\n".join(lines[start:i]))
    return blocks


def expected_body_blocks(source: str) -> list[str]:
    blocks = split_blocks(source)
    expected: list[str] = []
    in_front_toc = True

    for block in blocks:
        if in_front_toc:
            stripped = block.strip()
            if stripped == "## 1 Introduction":
                in_front_toc = False
            elif stripped.startswith("Peter Holderrieth") or stripped.startswith("Website:"):
                expected.append(block)
                continue
            else:
                continue
        expected.append(block)
    return expected


def load_segments() -> list[dict[str, str]]:
    text = SEGMENTS_TS.read_text(encoding="utf-8")
    match = re.search(r"export const lectureNotesSegments = (\[.*\]) satisfies", text, flags=re.S)
    if not match:
        raise ValueError("Cannot parse lectureNotesSegments JSON payload.")
    return json.loads(match.group(1))


def strip_tags(value: str) -> str:
    math_tokens: dict[str, str] = {}

    def stash_math(match: re.Match[str]) -> str:
        token = f"@@AUDITMATH{len(math_tokens)}@@"
        math_tokens[token] = match.group(0)
        return token

    value = re.sub(r"(?s)\$\$.*?\$\$", stash_math, value)
    value = re.sub(r"\$[^$\n]+\$", stash_math, value)
    value = re.sub(r"<script.*?</script>", " ", value, flags=re.S)
    value = re.sub(r"<style.*?</style>", " ", value, flags=re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    for token, math_value in math_tokens.items():
        value = value.replace(token, math_value)
    return html.unescape(value)


def normalize_text(value: str) -> str:
    value = strip_tags(value)
    return normalize_plain_text(value)


def normalize_plain_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = value.replace("“", '"').replace("”", '"')
    value = value.replace("’", "'").replace("‘", "'")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_for_lookup(value: str) -> str:
    value = normalize_plain_text(value)
    value = re.sub(r"\.{2,}", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_html_for_lookup(value: str) -> str:
    return normalize_for_lookup(strip_tags(value))


def normalize_source_block(block: str) -> str:
    block = re.sub(r"^## +", "", block.strip())
    if not block.startswith("$$"):
        block = re.sub(r"(?m)^\s*(?:\d+\.|-)\s+", "", block)
    block = re.sub(r"!\[[^\]]*\]\(images/([^)]+)\)", r"\1", block)
    block = re.sub(r"^```.*?```$", "CODE_BLOCK", block, flags=re.S)
    return normalize_plain_text(block)


def image_refs(value: str) -> set[str]:
    return {Path(match).name for match in re.findall(r"(?:images/|/images/)([a-f0-9]{64}\.jpg)", value)}


def markdown_image_refs(value: str) -> set[str]:
    return {Path(match).name for match in re.findall(r"!\[[^\]]*\]\(images/([^)]+)\)", value)}


def display_math(value: str) -> list[str]:
    return [re.sub(r"\s+", " ", item.strip()) for item in re.findall(r"\$\$(.*?)\$\$", value, flags=re.S)]


def inline_math(value: str) -> list[str]:
    without_display = re.sub(r"\$\$.*?\$\$", " ", value, flags=re.S)
    return [re.sub(r"\s+", " ", item.strip()) for item in re.findall(r"(?<!\$)\$([^$\n]+)\$(?!\$)", without_display)]


def all_math(value: str) -> list[str]:
    return display_math(value) + inline_math(value)


def figure_numbers(value: str) -> set[int]:
    return {int(match) for match in re.findall(r"\bFigure\s+([0-9]+)\s*:", value)}


def html_list_item_count(value: str) -> int:
    return len(re.findall(r"<li(?:\s|>)", value))


def has_chinese(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))


def is_nontranslated_allowed(en_html: str) -> bool:
    text = normalize_text(en_html)
    return (
        not text
        or en_html.strip().startswith("<p><img")
        or bool(display_math(en_html))
        or text.startswith("▶")
        or text == "CODE_BLOCK"
        or bool(re.match(r"^\[[0-9]+\] ", text))
        or bool(re.fullmatch(r"\$.*\$", text, flags=re.S))
        or bool(re.fullmatch(r"\[[0-9,\s]+\]", text))
    )


def pdf_page_count() -> int:
    result = subprocess.run(
        ["pdfinfo", str(PDF_PATH)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.M)
    if not match:
        raise ValueError("pdfinfo did not report a page count.")
    return int(match.group(1))


def pdf_text() -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(PDF_PATH), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    for path in [PDF_PATH, SOURCE_MD, CONTENT_LIST, SEGMENTS_TS, META_TS]:
        if not path.exists():
            fail(errors, f"Missing required file: {path}")

    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        raise SystemExit(1)

    source = strip_mineru_details(SOURCE_MD.read_text(encoding="utf-8"))
    source_blocks = expected_body_blocks(source)
    segments = load_segments()
    content_items = json.loads(CONTENT_LIST.read_text(encoding="utf-8"))
    meta = META_TS.read_text(encoding="utf-8")
    pdf_lookup = normalize_for_lookup(pdf_text())
    source_lookup = normalize_for_lookup(source)
    english_lookup = normalize_html_for_lookup("\n".join(segment["en"] for segment in segments))
    generated_lookup = f"{normalize_for_lookup(meta)} {english_lookup}"

    page_count = pdf_page_count()
    if page_count != 84:
        fail(errors, f"Expected 84 PDF pages, found {page_count}.")

    content_pages = {item.get("page_idx") for item in content_items if item.get("page_idx") is not None}
    if len(content_pages) != page_count:
        fail(errors, f"MinerU content_list covers {len(content_pages)} pages, expected {page_count}.")

    for item_type, expected_count in EXPECTED_CONTENT_TYPE_COUNTS.items():
        actual_count = sum(1 for item in content_items if item.get("type") == item_type)
        if actual_count != expected_count:
            fail(errors, f"MinerU {item_type} count changed: expected {expected_count}, found {actual_count}.")

    for phrase in REQUIRED_PDF_PHRASES:
        normalized_phrase = normalize_for_lookup(phrase)
        if normalized_phrase not in pdf_lookup:
            fail(errors, f"PDF text is missing required phrase: {phrase}")
        if normalized_phrase not in source_lookup:
            fail(errors, f"MinerU markdown is missing required PDF phrase: {phrase}")
        if normalized_phrase not in generated_lookup:
            fail(errors, f"Generated English blog is missing required PDF phrase: {phrase}")

    if len(source_blocks) != len(segments):
        fail(errors, f"Expected {len(source_blocks)} blog segments from source blocks, found {len(segments)}.")

    source_images = markdown_image_refs(source)
    public_images = {path.name for path in PUBLIC_IMAGES.glob("*.jpg")}
    segment_images = set().union(*(image_refs(segment["en"]) | image_refs(segment["zh"]) for segment in segments))
    content_images = {Path(item["img_path"]).name for item in content_items if item.get("img_path")}
    pdf_figures = figure_numbers(pdf_lookup)
    source_figures = figure_numbers(source)
    english_figures = figure_numbers(english_lookup)

    if pdf_figures != EXPECTED_FIGURE_NUMBERS:
        fail(errors, f"PDF figure numbers changed: {sorted(pdf_figures)}")
    if source_figures != EXPECTED_FIGURE_NUMBERS:
        fail(errors, f"MinerU figure numbers changed: {sorted(source_figures)}")
    if english_figures != EXPECTED_FIGURE_NUMBERS:
        fail(errors, f"Generated English figure numbers changed: {sorted(english_figures)}")

    if source_images != segment_images:
        fail(errors, f"Image refs differ between source and segments: source={len(source_images)} segments={len(segment_images)}.")
    if source_images - public_images:
        fail(errors, f"Missing copied public images: {sorted(source_images - public_images)[:5]}")
    if source_images != content_images:
        warnings.append(
            f"Markdown uses {len(source_images)} image refs; structured content_list has {len(content_images)} image paths."
        )

    source_display = display_math(source)
    segment_display = []
    for segment in segments:
        segment_display.extend(display_math(segment["en"]))
    content_equations = [item for item in content_items if item.get("type") == "equation"]
    if len(source_display) != len(content_equations):
        fail(errors, f"Display equation count mismatch: markdown={len(source_display)} content_list={len(content_equations)}.")
    if source_display != segment_display:
        fail(errors, "Display equations in generated English segments do not exactly match source order/content.")

    placeholder_patterns = [
        r"保留_",
        r"KEEP_",
        r"ZQX[0-9]",
        r"<details>",
        r"</details>",
        r"<ol>\s*</ol>",
        r"<ul>\s*</ul>",
        r"\$\$\s*\$\$",
    ]
    segments_text = SEGMENTS_TS.read_text(encoding="utf-8")
    for pattern in placeholder_patterns:
        if re.search(pattern, segments_text):
            fail(errors, f"Found forbidden generated artifact matching {pattern!r}.")

    math_mismatches: list[int] = []
    text_mismatches: list[int] = []
    untranslated: list[int] = []
    list_mismatches: list[int] = []

    for index, (block, segment) in enumerate(zip(source_blocks, segments), start=1):
        en_html = segment["en"]
        zh_html = segment["zh"]

        source_math = all_math(block)
        if all_math(en_html) != source_math or all_math(zh_html) != source_math:
            math_mismatches.append(index)

        if html_list_item_count(en_html) != html_list_item_count(zh_html):
            list_mismatches.append(index)
        if re.search(r"<br />\n\s*(?:\d+\.|-)\s+", zh_html):
            list_mismatches.append(index)

        source_text = normalize_source_block(block)
        en_text = normalize_text(en_html)
        if source_text and source_text not in en_text and en_text not in source_text:
            text_mismatches.append(index)

        if not zh_html.strip():
            untranslated.append(index)
        elif normalize_text(zh_html) == normalize_text(en_html) and not is_nontranslated_allowed(en_html):
            untranslated.append(index)
        elif not has_chinese(zh_html) and re.search(r"[A-Za-z]{4,}", normalize_text(en_html)) and not is_nontranslated_allowed(en_html):
            untranslated.append(index)

    if math_mismatches:
        fail(errors, f"Math mismatch in segments: {math_mismatches[:12]}{'...' if len(math_mismatches) > 12 else ''}")
    if list_mismatches:
        unique = sorted(set(list_mismatches))
        fail(errors, f"List structure mismatch in segments: {unique[:12]}{'...' if len(unique) > 12 else ''}")
    if text_mismatches:
        fail(errors, f"English text mismatch in segments: {text_mismatches[:12]}{'...' if len(text_mismatches) > 12 else ''}")
    if untranslated:
        fail(errors, f"Potential untranslated zh segments: {untranslated[:12]}{'...' if len(untranslated) > 12 else ''}")

    if ORIGINAL_TITLE not in meta or ORIGINAL_URL not in meta or 'status: "translation"' not in meta:
        fail(errors, "Meta file is missing translation status, original title, or original URL.")

    print("Lecture notes audit")
    print(f"- PDF pages: {page_count}")
    print(f"- MinerU content items: {len(content_items)}")
    print(f"- Blog segments: {len(segments)}")
    print(f"- Referenced/copied images: {len(source_images)}/{len(public_images)}")
    print(
        f"- Structured images/charts: {EXPECTED_CONTENT_TYPE_COUNTS['image']}/{EXPECTED_CONTENT_TYPE_COUNTS['chart']}"
    )
    print(f"- Figure captions: {len(source_figures)}")
    print(f"- Display equations: {len(source_display)}")
    print(f"- Inline math snippets in source body: {len(inline_math('\\n\\n'.join(source_blocks)))}")
    print(f"- Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"  WARN: {warning}")

    if errors:
        print(f"- Errors: {len(errors)}")
        for error in errors:
            print(f"  ERROR: {error}")
        raise SystemExit(1)

    print("- Result: PASS")


if __name__ == "__main__":
    main()

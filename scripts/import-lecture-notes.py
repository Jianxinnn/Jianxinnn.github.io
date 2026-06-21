from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import requests
from deep_translator import GoogleTranslator
from markdown import markdown

_requests_get = requests.get


def requests_get_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", 20)
    return _requests_get(*args, **kwargs)


requests.get = requests_get_with_timeout


REPO_ROOT = Path(__file__).resolve().parents[1]
MINERU_ROOT = Path(
    "/Users/jxtang/MinerU/lecture_notes.pdf-afa87b40-318d-4443-9656-9fb8362ad1c4"
)
SOURCE_MD = MINERU_ROOT / "full.md"
SOURCE_IMAGES = MINERU_ROOT / "images"
SLUG = "flow-matching-diffusion-models-cn"
POST_ROOT = REPO_ROOT / "content" / "blog" / "posts" / SLUG
PUBLIC_ROOT = REPO_ROOT / "public" / "blog" / SLUG
PUBLIC_IMAGES = PUBLIC_ROOT / "images"
CACHE_PATH = REPO_ROOT / ".cache" / "lecture-notes-translation-cache.json"

ORIGINAL_TITLE = "An Introduction to Flow Matching and Diffusion Models"
ORIGINAL_URL = "https://diffusion.csail.mit.edu/"


PROTECTED_PATTERNS = [
    re.compile(r"(?s)```.*?```"),
    re.compile(r"(?s)\$\$.*?\$\$"),
    re.compile(r"\$[^$\n]+\$"),
    re.compile(r"!\[[^\]]*\]\([^)]+\)"),
    re.compile(r"`[^`]+`"),
    re.compile(r"https?://[^\s)]+"),
    re.compile(r"\[[0-9,\s]+\]"),
    re.compile(r"<[^>]+>"),
]


TERM_FIXES = [
    ("流量匹配", "流匹配"),
    ("流程匹配", "流匹配"),
    ("去噪扩散模型", "去噪扩散模型"),
    ("生成建模", "生成式建模"),
    ("生成模型", "生成式模型"),
    ("普通微分方程", "常微分方程"),
    ("分数函数", "score 函数"),
    ("分数匹配", "score matching"),
    ("评分函数", "score 函数"),
    ("评分匹配", "score matching"),
    ("流动和扩散模型", "流模型与扩散模型"),
    ("分类器免费指导", "无分类器引导"),
    ("无分类器指导", "无分类器引导"),
    ("香草指导", "普通引导"),
    ("指南：如何根据prompt进行调节", "引导：如何基于 prompt 条件化"),
    ("指南：如何根据 prompt 进行调节", "引导：如何基于 prompt 条件化"),
    ("指南：如何根据prompt进行条件调整", "引导：如何基于 prompt 条件化"),
    ("指南：如何根据 prompt 进行条件调整", "引导：如何基于 prompt 条件化"),
    ("指导", "引导"),
    ("噪音", "噪声"),
    ("提示", "prompt"),
    ("边际概率路径", "边缘概率路径"),
    ("边际向量场", "边缘向量场"),
    ("边际分布", "边缘分布"),
    ("潜在空间", "latent 空间"),
    ("自动编码器", "自编码器"),
    ("跳跃率", "跳转率"),
]

HEADING_FIXES = {
    "## 4.3 Score Matching": "## 4.3 Score Matching（得分匹配）",
    "## 8 References": "## 8 参考文献",
}


REFERENCE_START = re.compile(r"^## 8 References")
APPENDIX_START = re.compile(r"^## A ")
LIST_MARKER = re.compile(r"^(\s*(?:\d+\.|-)\s+)(.*)$")


def load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
      return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


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


def protect(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        token = f"ZQX{len(placeholders)}QXZ"
        placeholders[token] = match.group(0)
        return token

    protected = text
    for pattern in PROTECTED_PATTERNS:
        protected = pattern.sub(replace, protected)
    return protected, placeholders


def restore(text: str, placeholders: dict[str, str]) -> str:
    for token, value in placeholders.items():
        text = re.sub(re.escape(token), lambda _match: value, text, flags=re.I)
    return text


def display_math(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", item.strip()) for item in re.findall(r"\$\$(.*?)\$\$", text, flags=re.S)]


def inline_math(text: str) -> list[str]:
    without_display = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    return [re.sub(r"\s+", " ", item.strip()) for item in re.findall(r"(?<!\$)\$([^$\n]+)\$(?!\$)", without_display)]


def math_fragments(text: str) -> list[str]:
    return display_math(text) + inline_math(text)


def list_item_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if LIST_MARKER.match(line))


def apply_term_fixes(text: str) -> str:
    for source, target in TERM_FIXES:
        text = text.replace(source, target)
    text = text.replace("Figure ", "图 ")
    text = text.replace("Key Idea", "关键思想")
    text = text.replace("Summary", "小结")
    text = text.replace("Theorem", "定理")
    text = text.replace("Example", "例")
    text = text.replace("Remark", "注")
    text = text.replace("Proposition", "命题")
    text = text.replace("Intuition", "直观解释")
    return text


def translate_plain_text(text: str, translator: GoogleTranslator) -> str:
    if not text.strip() or not re.search(r"[A-Za-z]", text):
        return text

    leading = re.match(r"^\s*", text).group(0)
    trailing = re.search(r"\s*$", text).group(0)
    core = text[len(leading) : len(text) - len(trailing) if trailing else len(text)]
    if not core.strip():
        return text

    last_error: Exception | None = None
    for attempt in range(6):
        try:
            return leading + translator.translate(core) + trailing
        except Exception as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
            translator = GoogleTranslator(source="en", target="zh-CN")
    raise RuntimeError(f"Translation failed after retries for text fragment: {core[:120]!r}") from last_error


def translate_preserving_protected_order(body: str, translator: GoogleTranslator) -> str:
    protected, placeholders = protect(body)
    translated_parts: list[str] = []
    for part in re.split(r"(ZQX\d+QXZ)", protected):
        if not part:
            continue
        if part in placeholders:
            translated_parts.append(placeholders[part])
        else:
            translated_parts.append(translate_plain_text(part, translator))
    return apply_term_fixes("".join(translated_parts))


def translate_list_block(body: str, translator: GoogleTranslator) -> str:
    translated_lines: list[str] = []
    for line in body.splitlines():
        match = LIST_MARKER.match(line)
        if match:
            marker, item = match.groups()
            translated_lines.append(marker + translate_preserving_protected_order(item, translator).lstrip())
        else:
            translated_lines.append(translate_preserving_protected_order(line, translator))
    return "\n".join(translated_lines)


def should_skip_translation(block: str, in_references: bool) -> bool:
    stripped = block.strip()
    return (
        in_references
        or stripped.startswith("![](")
        or stripped.startswith("```")
        or stripped.startswith("$$")
        or bool(re.fullmatch(r"\$[^$]+\$", stripped, flags=re.S))
    )


def translate_block(block: str, translator: GoogleTranslator, cache: dict[str, str], in_references: bool) -> str:
    if block in HEADING_FIXES:
        return HEADING_FIXES[block]

    if should_skip_translation(block, in_references):
        return block

    digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
    if digest in cache:
        cached = apply_term_fixes(cache[digest])
        if (
            "保留_" not in cached
            and "KEEP_" not in cached
            and math_fragments(cached) == math_fragments(block)
            and list_item_count(cached) == list_item_count(block)
        ):
            return cached

    prefix = ""
    body = block
    if block.startswith("## "):
        prefix = "## "
        body = block[3:]

    translated = (
        translate_list_block(body, translator)
        if list_item_count(body)
        else translate_preserving_protected_order(body, translator)
    )
    result = prefix + translated
    cache[digest] = result
    return result


def normalize_heading(block: str) -> str:
    if not block.startswith("## "):
        return block

    title = block[3:].strip()
    if title == ORIGINAL_TITLE:
        return ""

    level = "##"
    if re.match(r"^(\d+|[A-E])\s+", title):
        level = "##"
    elif re.match(r"^(\d+\.\d+|[A-E]\.\d+)", title):
        level = "###"
    elif re.match(r"^(\d+\.\d+\.\d+)", title):
        level = "####"
    else:
        level = "###"
    return f"{level} {title}"


def rewrite_image_paths(block: str) -> str:
    def replace(match: re.Match[str]) -> str:
        filename = Path(match.group(1)).name
        return f"](/blog/{SLUG}/images/{filename})"

    return re.sub(r"\]\(images/([^)]+)\)", replace, block)


def markdown_to_html(block: str) -> str:
    block = rewrite_image_paths(normalize_heading(block))
    if not block.strip():
        return ""

    math_tokens: dict[str, str] = {}

    def stash_math(match: re.Match[str]) -> str:
        token = f"@@MATH{len(math_tokens)}@@"
        math_tokens[token] = match.group(0)
        return token

    protected = re.sub(r"(?s)\$\$.*?\$\$", stash_math, block)
    protected = re.sub(r"\$[^$\n]+\$", stash_math, protected)
    html = markdown(protected, extensions=["extra", "sane_lists"])
    for token, value in math_tokens.items():
        html = html.replace(token, value)
    html = html.replace("<img ", '<img loading="lazy" ')
    return html


def referenced_images(text: str) -> set[str]:
    return {Path(match).name for match in re.findall(r"!\[[^\]]*\]\(images/([^)]+)\)", text)}


def write_generated_files(segments: list[dict[str, str]]) -> None:
    POST_ROOT.mkdir(parents=True, exist_ok=True)
    PUBLIC_IMAGES.mkdir(parents=True, exist_ok=True)

    segments_ts = (
        'import type { BilingualSegment } from "@/components/bilingual-article";\n\n'
        "export const lectureNotesSegments = "
        + json.dumps(segments, ensure_ascii=False, indent=2)
        + " satisfies BilingualSegment[];\n"
    )
    (POST_ROOT / "segments.ts").write_text(segments_ts, encoding="utf-8")

    (POST_ROOT / "index.mdx").write_text(
        'import { BilingualArticle } from "@/components/bilingual-article";\n'
        'import { lectureNotesSegments } from "./segments";\n\n'
        "<BilingualArticle segments={lectureNotesSegments} />\n",
        encoding="utf-8",
    )

    (POST_ROOT / "meta.ts").write_text(
        'import type { BlogPostMeta } from "../../types";\n\n'
        "const meta = {\n"
        '  title: "流匹配与扩散模型导论",\n'
        '  summary: "MIT 6.S184 课程讲义的中文翻译与双语对照版本，系统介绍 flow matching、扩散模型、score matching、guidance、latent diffusion 和离散扩散语言模型。",\n'
        '  date: "2026-06-21",\n'
        '  readingTime: "84 pages",\n'
        '  sourceType: "mdx",\n'
        '  image: "/assets/visuals/profile-field.png",\n'
        '  category: "Paper notes",\n'
        '  language: "bilingual",\n'
        "  source: {\n"
        '    status: "translation",\n'
        '    label: "转载 / 翻译",\n'
        f'    originalTitle: "{ORIGINAL_TITLE}",\n'
        f'    originalUrl: "{ORIGINAL_URL}"\n'
        "  },\n"
        '  tags: ["diffusion models", "flow matching"]\n'
        "} satisfies BlogPostMeta;\n\n"
        "export default meta;\n",
        encoding="utf-8",
    )


def copy_images(text: str) -> None:
    used_images = referenced_images(text)
    PUBLIC_IMAGES.mkdir(parents=True, exist_ok=True)
    for filename in used_images:
        src = SOURCE_IMAGES / filename
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, PUBLIC_IMAGES / filename)


def main() -> None:
    source = strip_mineru_details(SOURCE_MD.read_text(encoding="utf-8"))
    copy_images(source)

    blocks = split_blocks(source)
    cache = load_cache()
    translator = GoogleTranslator(source="en", target="zh-CN")
    segments: list[dict[str, str]] = []
    in_references = False
    in_front_toc = True

    for index, block in enumerate(blocks):
        if in_front_toc:
            stripped = block.strip()
            if stripped == "## 1 Introduction":
                in_front_toc = False
            elif stripped.startswith("Peter Holderrieth") or stripped.startswith("Website:"):
                pass
            else:
                continue

        if REFERENCE_START.match(block):
            in_references = True
        elif APPENDIX_START.match(block):
            in_references = False

        zh_block = translate_block(block, translator, cache, in_references)
        en_html = markdown_to_html(block)
        zh_html = markdown_to_html(zh_block)
        if en_html and zh_html:
            segments.append({"kind": "html", "en": en_html, "zh": zh_html})

        if index % 5 == 0:
            save_cache(cache)
        if index % 20 == 0:
            print(f"translated block {index + 1}/{len(blocks)}; cache={len(cache)}", flush=True)
            time.sleep(0.4)

    save_cache(cache)
    write_generated_files(segments)
    print(f"Generated {len(segments)} bilingual segments for {SLUG}.")
    print(f"Copied {len(referenced_images(source))} referenced images.")


if __name__ == "__main__":
    main()

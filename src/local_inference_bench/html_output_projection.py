"""Project model-produced HTML into visible text and valid table cells."""

from __future__ import annotations

import re
from html.parser import HTMLParser


_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "tfoot",
        "thead",
        "tr",
        "ul",
    }
)
_CELL_TAGS = frozenset({"td", "th"})
_SECTION_TAGS = frozenset({"tbody", "tfoot", "thead"})
_TABLE_STRUCTURE_TAGS = frozenset(
    {"caption", "col", "colgroup", "table", "tr", *_CELL_TAGS, *_SECTION_TAGS}
)
_INVISIBLE_TAGS = frozenset(
    {
        "audio",
        "canvas",
        "datalist",
        "head",
        "iframe",
        "noscript",
        "object",
        "script",
        "select",
        "style",
        "template",
        "title",
        "video",
    }
)
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_WHITESPACE = re.compile(r"[\t\f\v ]+")
_DISPLAY_NONE = re.compile(
    r"(?:^|;)\s*display\s*:\s*none(?:\s*!important)?\s*(?:;|$)",
    re.IGNORECASE,
)
_VISIBILITY_HIDDEN = re.compile(
    r"(?:^|;)\s*visibility\s*:\s*(?:hidden|collapse)(?:\s*!important)?\s*(?:;|$)",
    re.IGNORECASE,
)


def project_html_visible_text(value: str) -> str:
    """Return deterministic rendered-text evidence without hidden HTML content."""

    if type(value) is not str:
        raise TypeError("HTML output must be a string")
    parser = _VisibleTextParser()
    parser.feed(value)
    parser.close()
    return _normalize_lines("".join(parser.chunks))


def extract_html_table_rows(value: str) -> list[list[list[str]]]:
    """Return only closed, structurally valid HTML tables in source order."""

    if type(value) is not str:
        raise TypeError("HTML output must be a string")
    parser = _TableParser()
    parser.feed(value)
    parser.close()
    parser.finish()
    if parser.had_malformed_table:
        return []
    return parser.tables


def _normalize_lines(value: str) -> str:
    lines = [_WHITESPACE.sub(" ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def _element_is_hidden(
    tag: str,
    attrs: list[tuple[str, str | None]],
) -> bool:
    if tag in _INVISIBLE_TAGS:
        return True
    attributes = {name.casefold(): value for name, value in attrs}
    if tag in {"details", "dialog"} and "open" not in attributes:
        # The bounded scorer does not model interactive disclosure. Suppressing
        # the whole closed container is conservative: it cannot turn hidden
        # descendants into positive recognition evidence.
        return True
    if "hidden" in attributes:
        return True
    if "popover" in attributes:
        return True
    aria_hidden = attributes.get("aria-hidden")
    if isinstance(aria_hidden, str) and aria_hidden.strip().casefold() == "true":
        return True
    style = attributes.get("style")
    return isinstance(style, str) and (
        _DISPLAY_NONE.search(style) is not None
        or _VISIBILITY_HIDDEN.search(style) is not None
    )


class _VisibilityStack:
    def __init__(self) -> None:
        self.open_elements: list[tuple[str, bool]] = []
        self.hidden_depth = 0

    def push(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        hidden = _element_is_hidden(tag, attrs)
        self.open_elements.append((tag, hidden))
        self.hidden_depth += int(hidden)

    def pop(self, tag: str) -> None:
        for index in range(len(self.open_elements) - 1, -1, -1):
            if self.open_elements[index][0] != tag:
                continue
            removed = self.open_elements[index:]
            del self.open_elements[index:]
            self.hidden_depth -= sum(int(hidden) for _, hidden in removed)
            return


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._visibility = _VisibilityStack()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        folded = tag.casefold()
        hidden = _element_is_hidden(folded, attrs)
        if folded not in _VOID_TAGS:
            self._visibility.push(folded, attrs)
        if self._visibility.hidden_depth or hidden:
            return
        self._append_start_boundary(folded)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        folded = tag.casefold()
        hidden = _element_is_hidden(folded, attrs)
        if folded not in _VOID_TAGS and hidden:
            # HTML does not permit self-closing syntax to close non-void elements.
            # Treat a suppressed ``<script/>``-style token as an opening tag so
            # malformed markup cannot expose trailing text as positive evidence.
            self._visibility.push(folded, attrs)
            return
        if self._visibility.hidden_depth or hidden:
            return
        self._append_start_boundary(folded)

    def handle_endtag(self, tag: str) -> None:
        folded = tag.casefold()
        if self._visibility.hidden_depth == 0:
            if folded in _BLOCK_TAGS:
                self.chunks.append("\n")
            elif folded in _CELL_TAGS:
                self.chunks.append(" ")
        self._visibility.pop(folded)

    def handle_data(self, data: str) -> None:
        if self._visibility.hidden_depth == 0:
            self.chunks.append(data)

    def _append_start_boundary(self, tag: str) -> None:
        if tag == "br" or tag in _BLOCK_TAGS:
            self.chunks.append("\n")
        elif tag in _CELL_TAGS:
            self.chunks.append(" ")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._visibility = _VisibilityStack()
        self._table_depth = 0
        self._valid = False
        self._rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._cell_tag: str | None = None
        self._section_tag: str | None = None
        self._in_caption = False
        self._in_colgroup = False
        self._markup_stack: list[str] = []
        self._top_level_children: list[str] = []
        self.had_malformed_table = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        folded = tag.casefold()
        self._track_markup_start(folded)
        if (
            self._table_depth
            and folded in _CELL_TAGS
            and not _cell_spans_are_default(attrs)
        ):
            self._valid = False
        hidden = _element_is_hidden(folded, attrs)
        if folded == "table" and (self._visibility.hidden_depth or hidden):
            self.had_malformed_table = True
        if folded not in _VOID_TAGS:
            self._visibility.push(folded, attrs)
        if self._visibility.hidden_depth or hidden:
            return
        self._handle_visible_start(folded)

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        folded = tag.casefold()
        hidden = _element_is_hidden(folded, attrs)
        if folded == "table" and (self._visibility.hidden_depth or hidden):
            self.had_malformed_table = True
        if self._table_depth or folded == "table":
            if folded not in _VOID_TAGS:
                self._valid = False
            self._validate_markup_child(folded)
            if folded in _CELL_TAGS and not _cell_spans_are_default(attrs):
                self._valid = False
        if folded not in _VOID_TAGS and hidden:
            self._visibility.push(folded, attrs)
            return
        if self._visibility.hidden_depth or hidden:
            return
        self._handle_visible_start(folded)

    def handle_endtag(self, tag: str) -> None:
        folded = tag.casefold()
        self._track_markup_end(folded)
        if self._visibility.hidden_depth == 0:
            self._handle_visible_end(folded)
        self._visibility.pop(folded)

    def handle_data(self, data: str) -> None:
        if self._visibility.hidden_depth or self._table_depth != 1:
            return
        if self._cell is not None:
            self._cell.append(data)
        elif not self._in_caption and data.strip():
            self._valid = False

    def finish(self) -> None:
        if self._table_depth or self._markup_stack:
            self.had_malformed_table = True
        self._reset_table()

    def _handle_visible_start(self, tag: str) -> None:
        if tag == "table":
            if self._table_depth == 0:
                self._table_depth = 1
                self._valid = True
                self._rows = []
                self._row = None
                self._cell = None
                self._cell_tag = None
                self._section_tag = None
                self._in_caption = False
                self._in_colgroup = False
                self._top_level_children = []
            else:
                self._table_depth += 1
                self._valid = False
            return
        if self._table_depth != 1:
            return
        if self._cell is not None:
            if tag == "br":
                self._cell.append(" ")
            elif tag in _TABLE_STRUCTURE_TAGS:
                self._valid = False
            return
        if self._in_caption:
            if tag in _TABLE_STRUCTURE_TAGS:
                self._valid = False
            return
        if self._in_colgroup:
            if tag != "col":
                self._valid = False
            return
        if tag == "tr":
            if self._row is not None or self._cell is not None:
                self._valid = False
            self._row = []
        elif tag in _CELL_TAGS:
            if self._row is None or self._cell is not None:
                self._valid = False
            self._cell = []
            self._cell_tag = tag
        elif self._row is not None:
            self._valid = False
        elif tag in _SECTION_TAGS:
            if self._section_tag is not None:
                self._valid = False
            else:
                self._section_tag = tag
        elif tag == "caption":
            if self._section_tag is not None or self._rows:
                self._valid = False
            self._in_caption = True
        elif tag == "colgroup":
            if self._section_tag is not None or self._rows:
                self._valid = False
            self._in_colgroup = True
        else:
            self._valid = False

    def _handle_visible_end(self, tag: str) -> None:
        if tag == "table" and self._table_depth:
            if self._table_depth > 1:
                self._table_depth -= 1
                return
            if (
                self._cell is not None
                or self._row is not None
                or self._section_tag is not None
                or self._in_caption
                or self._in_colgroup
            ):
                self._valid = False
            if not _table_child_sequence_is_valid(self._top_level_children):
                self._valid = False
            if self._valid and self._rows:
                self.tables.append(self._rows)
            elif not self._valid:
                self.had_malformed_table = True
            self._reset_table()
            return
        if self._table_depth != 1:
            return
        if self._in_caption:
            if tag == "caption":
                self._in_caption = False
            elif tag in _TABLE_STRUCTURE_TAGS:
                self._valid = False
            return
        if self._in_colgroup:
            if tag == "colgroup":
                self._in_colgroup = False
            elif tag != "col":
                self._valid = False
            return
        if tag in _CELL_TAGS:
            if self._cell is None or self._cell_tag != tag or self._row is None:
                self._valid = False
                return
            self._row.append(_WHITESPACE.sub(" ", "".join(self._cell)).strip())
            self._cell = None
            self._cell_tag = None
        elif tag == "tr":
            if self._row is None or self._cell is not None:
                self._valid = False
                return
            if not self._row:
                self._valid = False
            else:
                self._rows.append(self._row)
            self._row = None
        elif tag in _SECTION_TAGS:
            if self._section_tag != tag or self._row is not None:
                self._valid = False
            else:
                self._section_tag = None

    def _reset_table(self) -> None:
        self._table_depth = 0
        self._valid = False
        self._rows = []
        self._row = None
        self._cell = None
        self._cell_tag = None
        self._section_tag = None
        self._in_caption = False
        self._in_colgroup = False
        self._markup_stack = []
        self._top_level_children = []

    def _track_markup_start(self, tag: str) -> None:
        if self._table_depth == 0:
            if tag == "table":
                self._markup_stack = [tag]
            return
        self._validate_markup_child(tag)
        if tag not in _VOID_TAGS:
            self._markup_stack.append(tag)

    def _validate_markup_child(self, tag: str) -> None:
        if not self._markup_stack:
            return
        parent = self._markup_stack[-1]
        if parent == "table":
            allowed = tag in {"caption", "colgroup", "tr", *_SECTION_TAGS}
            if allowed:
                self._top_level_children.append(tag)
        elif parent == "colgroup":
            allowed = tag == "col"
        elif parent in _SECTION_TAGS:
            allowed = tag == "tr"
        elif parent == "tr":
            allowed = tag in _CELL_TAGS
        else:
            allowed = tag not in _TABLE_STRUCTURE_TAGS
        if not allowed:
            self._valid = False

    def _track_markup_end(self, tag: str) -> None:
        if not self._markup_stack:
            return
        if self._markup_stack[-1] == tag:
            self._markup_stack.pop()
            return
        self._valid = False
        if tag in self._markup_stack:
            matching_index = len(self._markup_stack) - 1 - self._markup_stack[::-1].index(tag)
            del self._markup_stack[matching_index:]


def _table_child_sequence_is_valid(children: list[str]) -> bool:
    """Validate conservative HTML table child cardinality and ordering."""

    phase = 0
    seen_caption = False
    seen_thead = False
    seen_tfoot = False
    row_mode: str | None = None
    for tag in children:
        if tag == "caption":
            if seen_caption or phase != 0:
                return False
            seen_caption = True
            continue
        if tag == "colgroup":
            if phase > 1:
                return False
            phase = 1
            continue
        if tag == "thead":
            if seen_thead or phase > 2:
                return False
            seen_thead = True
            phase = 2
            continue
        if tag in {"tbody", "tr"}:
            next_mode = "section" if tag == "tbody" else "direct"
            if seen_tfoot or (row_mode is not None and row_mode != next_mode):
                return False
            row_mode = next_mode
            phase = 3
            continue
        if tag == "tfoot":
            if seen_tfoot:
                return False
            seen_tfoot = True
            phase = 4
            continue
        return False
    return True


def _cell_spans_are_default(attrs: list[tuple[str, str | None]]) -> bool:
    for name, value in attrs:
        if name.casefold() not in {"colspan", "rowspan"}:
            continue
        if not isinstance(value, str) or value.strip() != "1":
            return False
    return True

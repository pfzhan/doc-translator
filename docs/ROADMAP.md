# Roadmap

Work that is planned but not done yet. When an item ships, remove it here and describe the feature in
`README.md` and `README.zh-CN.md`. Bugs belong in issues, not here.

## PDF

- **Scanned PDFs.** Pages without a text layer are refused today with a hint to run OCR first. Running OCR in the
  pipeline would need a text layer that keeps the page's glyph positions, so that erasing and writing back work as
  they do for born-digital files.
- **Tables without rules.** A table is recognized by its rules or by aligned column edges. Without either, its
  cells are joined into paragraphs and translated as running text.
- **Reading order across unclear gutters.** When the gap between two columns is narrow or interrupted by a
  figure, lines of both columns can be read as one.
- **Paragraph breaks before a capital.** A line break followed by an English capital letter still starts a new
  paragraph, which splits sentences that continue with a proper noun.
- **Renumbering objects on save.** Saving uses `garbage=1` because MuPDF 1.28 loses a reference when it renumbers
  a dictionary of exactly 101 entries (see `_save` in `app/formats/pdf_output.py`; `tests/test_pdf_output.py`
  reproduces it). Report the bug upstream with that reproduction, and go back to `garbage=2`, which keeps the
  cross-reference table compact, once a fixed MuPDF is the minimum version.
- **Long functions.** `_split_lines` (`pdf_split_lines.py`, 465 lines), `_merge_continuations`
  (`pdf_continuations.py`, 339) and `_placeholderize` (`pdf_units.py`, 316) are still single functions with
  nested state, as the page writer was before it was split into classes. Splitting them follows the same method:
  record the per-page digests of the sample outputs first, and require them to be identical afterwards.

## Other formats

- **Word.** Text in inline content controls, comments, charts and SmartArt is not translated.
- **Subtitles.** A sentence split over several cues is translated cue by cue, so the translation of each cue lacks
  the rest of the sentence. Translating the sentence once and dividing the result over its cues would fix this.
- **HTML files.** A translated page refers to its images and styles by the original relative paths, so it shows
  them only when saved next to the original's files. Copying or inlining them would make the output standalone.

## Browser extension

- **Firefox.** The Firefox build (`scripts/build_firefox.py`) passes `web-ext lint` but has not been run in
  Firefox. The browser tests in `tests/test_extension_browser.py` run only in Chromium.

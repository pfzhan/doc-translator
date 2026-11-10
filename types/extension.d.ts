// Globals of the browser extension (extension/) that the type libraries lack. The extension's files are classic
// scripts sharing one global scope, checked together by tsconfig.extension.json.

// Worker-only: background.js runs as a service worker in Chrome, where importScripts exists, and as an event page in
// Firefox, where it does not (background.js checks before calling it). Declared here rather than by adding the
// webworker lib, whose globals clash with the DOM ones the other scripts use.
declare var importScripts: ((...urls: (string | URL)[]) => void) | undefined;

interface Window {
  /** Set by content.js once it has set itself up in a frame; later injections stop there. */
  __docTranslator?: boolean;
  /** Set by autostart.js on sites translated automatically, just before content.js runs. */
  __docTranslatorAuto?: boolean;
  /** Set by selection.js: translates the current selection; later injections call it instead of setting up again. */
  __docTranslatorSelection?: () => Promise<void>;
}

interface Element {
  /** Firefox only, to extensions' content scripts: the shadow root of the element, open or closed. Chrome has
   * chrome.dom.openOrClosedShadowRoot instead. */
  readonly openOrClosedShadowRoot?: ShadowRoot | null;
}

declare namespace chrome.commands {
  /** Firefox 137 and later: opens the browser's page for the extension's shortcuts. Chrome has no such call; it
   * opens chrome://extensions/shortcuts instead. */
  const openShortcutSettings: (() => Promise<void>) | undefined;
}

// ---------- Messages between the scripts ----------

/** Content scripts to the service worker: translate texts of the page. */
interface TranslateMessage {
  type: 'dt-translate' | 'dt-translate-selected';
  texts: string[];
  /** Aligned with texts: true when the text is an HTML fragment (see richOf in content.js). Missing means none. */
  html?: boolean[];
  title: string;
}

/** selection.js to the service worker: look up a selected word or short phrase. */
interface LookupMessage {
  type: 'dt-lookup';
  text: string;
  /** The sentence the text was selected from; empty when there is none. */
  context: string;
  title: string;
}

/** A dictionary entry, as POST /api/web/lookup gives it (WordEntry in app/lookup.py). */
interface WordEntry {
  headword: string;
  phonetic: string;
  senses: { pos: string; meaning: string }[];
  in_context: string;
}

/** The service worker's answer to a LookupMessage; lang is the language the entry is written in. */
type LookupAnswer = { error: string } | { entry: WordEntry; lang: string };

/** The content style sheet, for the shadow roots content.js enters. */
interface CssMessage {
  type: 'dt-css';
}

/** The service worker's answer to a TranslateMessage. A null translation keeps the original. fatal: the error comes
 * from the settings or the server, not from these texts, so every other request would fail the same way. */
type TranslateAnswer =
  | { error: string; fatal?: boolean }
  | { translations: (string | null)[]; lang: string; langs: (string | null)[] };

/** What the content script of a frame reports after a ControlMessage. */
interface PageState {
  started: boolean;
  hidden: boolean;
}

/** The service worker to the content script of a frame. dt-sync carries the top frame's state to an iframe. */
type ControlMessage = { type: 'dt-toggle' } | { type: 'dt-start' } | ({ type: 'dt-sync' } & PageState);

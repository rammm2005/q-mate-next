"use client";

import React, { useState } from "react";

/**
 * Supported languages for syntax highlighting.
 */
type SupportedLanguage =
  | "typescript"
  | "javascript"
  | "python"
  | "php"
  | "go";

const LANGUAGE_DISPLAY: Record<string, string> = {
  typescript: "TypeScript",
  javascript: "JavaScript",
  python: "Python",
  php: "PHP",
  go: "Go",
};

export interface SourceReferenceData {
  file_path: string;
  function_name?: string | null;
  start_line: number;
  end_line: number;
  snippet: string;
  relevance: number;
}

interface SourceReferenceProps {
  source: SourceReferenceData;
  index: number;
}

/**
 * Detects the language from a file path extension.
 */
function detectLanguage(filePath: string): SupportedLanguage | null {
  const ext = filePath.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "ts":
    case "tsx":
      return "typescript";
    case "js":
    case "jsx":
      return "javascript";
    case "py":
      return "python";
    case "php":
      return "php";
    case "go":
      return "go";
    default:
      return null;
  }
}

/**
 * SourceReference component displays an individual source reference
 * with file path, function name, line range, and an expandable code snippet.
 *
 * Requirements: 13.2, 13.3
 */
export default function SourceReference({ source, index }: SourceReferenceProps) {
  const [expanded, setExpanded] = useState(false);
  const language = detectLanguage(source.file_path);
  const languageClass = language ? `language-${language}` : "";
  const languageLabel = language
    ? LANGUAGE_DISPLAY[language]
    : "";

  return (
    <div className="source-reference">
      <div
        className="source-reference-header"
        onClick={() => setExpanded(!expanded)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            setExpanded(!expanded);
          }
        }}
        aria-expanded={expanded}
        aria-label={`Source ${index + 1}: ${source.file_path}`}
      >
        <span className="source-index">[{index + 1}]</span>
        <span className="source-file-path">{source.file_path}</span>
        {source.function_name && (
          <span className="source-function-name">
            {source.function_name}
          </span>
        )}
        <span className="source-line-range">
          Lines {source.start_line}-{source.end_line}
        </span>
        {languageLabel && (
          <span className="source-language-badge">{languageLabel}</span>
        )}
        <span className="source-expand-icon">
          {expanded ? "▾" : "▸"}
        </span>
      </div>

      {expanded && source.snippet && (
        <div className="source-snippet">
          <pre>
            <code className={languageClass}>{source.snippet}</code>
          </pre>
        </div>
      )}
    </div>
  );
}

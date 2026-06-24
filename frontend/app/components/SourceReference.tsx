"use client";

import React, { useState } from "react";
import { Eye, ChevronDown, ChevronRight, FileCode } from "lucide-react";

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
  onOpenFile?: (filePath: string, line: number) => void;
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
 * SourceReference component displays an individual source reference.
 */
export default function SourceReference({ source, index, onOpenFile }: SourceReferenceProps) {
  const [expanded, setExpanded] = useState(false);
  const language = detectLanguage(source.file_path);
  const languageClass = language ? `language-${language}` : "";
  const languageLabel = language
    ? LANGUAGE_DISPLAY[language]
    : "";

  const handleOpenFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onOpenFile) {
      onOpenFile(source.file_path, source.start_line);
    }
  };

  return (
    <div className="border border-gray-200 dark:border-gray-800 rounded-lg mb-2 overflow-hidden bg-white dark:bg-darkCard transition-all duration-150">
      <div
        className="group flex items-center gap-2 px-3 py-2.5 cursor-pointer hover:bg-blue-500/5 transition-colors duration-150 flex-wrap outline-none select-none"
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
        <span className="text-xs font-bold text-blue-600 dark:text-blue-400 flex-shrink-0">[{index + 1}]</span>
        <span className="font-mono text-xs font-medium text-blue-600 dark:text-blue-400 underline decoration-transparent group-hover:decoration-blue-500 dark:group-hover:decoration-blue-400 transition-colors duration-150 truncate max-w-[200px] sm:max-w-xs md:max-w-md" title={source.file_path}>
          {source.file_path}
        </span>
        {source.function_name && (
          <span className="text-xs text-gray-500 dark:text-gray-400 italic">
            → {source.function_name}
          </span>
        )}
        <span className="text-xs text-gray-400 dark:text-gray-500 ml-auto">
          Lines {source.start_line}-{source.end_line}
        </span>
        {languageLabel && (
          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 uppercase tracking-wider">{languageLabel}</span>
        )}
        {onOpenFile && (
          <button
            className="px-2.5 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded font-medium transition-all shadow-sm hover:shadow hover:-translate-y-[1px] active:translate-y-0 active:shadow-sm flex-shrink-0 ml-2 cursor-pointer flex items-center gap-1"
            onClick={handleOpenFile}
            title="Open file viewer"
          >
            <Eye size={12} />
            View
          </button>
        )}
        <span className="text-gray-400 dark:text-gray-500 flex-shrink-0 ml-1">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
      </div>

      {expanded && source.snippet && (
        <div className="border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/80 p-3 overflow-x-auto">
          <pre className="m-0">
            <code className={`font-mono text-xs leading-relaxed ${languageClass}`}>{source.snippet}</code>
          </pre>
        </div>
      )}
    </div>
  );
}

"use client";

import React, { useState, useEffect } from "react";
import SourceReference, { SourceReferenceData } from "./SourceReference";
import RetrievalStatistics from "./RetrievalStatistics";
import {
  ChevronDown,
  HelpCircle,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Info,
  BookOpen
} from "lucide-react";

export interface AnswerCardProps {
  question: string;
  answerText: string;
  sources: SourceReferenceData[];
  confidence: number;
  onOpenFile?: (filePath: string, startLine: number, endLine: number) => void;
  onToggle?: () => void;
  mode?: string;
  comparison?: {
    bm25_sources: SourceReferenceData[];
    indobert_sources: SourceReferenceData[];
    evaluation: string;
  } | null;
  isCollapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
}

/**
 * Returns CSS classes for confidence score color coding.
 */
function getConfidenceLevel(confidence: number): string {
  if (confidence >= 0.8) return "bg-green-500/15 text-green-700 dark:text-green-400";
  if (confidence >= 0.5) return "bg-amber-500/15 text-amber-700 dark:text-amber-400";
  return "bg-red-500/15 text-red-700 dark:text-red-400";
}

/**
 * Returns a matching icon based on confidence.
 */
function ConfidenceIcon({ confidence, size = 14 }: { confidence: number; size?: number }) {
  if (confidence >= 0.8) return <CheckCircle2 size={size} className="text-green-600 dark:text-green-400" />;
  if (confidence >= 0.5) return <Info size={size} className="text-amber-600 dark:text-amber-400" />;
  return <AlertCircle size={size} className="text-red-600 dark:text-red-400" />;
}

/**
 * Renders answer text with basic markdown-like formatting.
 */
function renderFormattedAnswer(text: string): React.ReactNode[] {
  const elements: React.ReactNode[] = [];
  if (!text) return elements;
  const lines = text.split("\n");
  let inCodeBlock = false;
  let codeBlockContent = "";
  let codeBlockLanguage = "";
  let blockIndex = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check for code block delimiters
    if (line.trimStart().startsWith("```")) {
      if (!inCodeBlock) {
        // Opening code block
        inCodeBlock = true;
        codeBlockLanguage = line.trimStart().slice(3).trim();
        codeBlockContent = "";
      } else {
        // Closing code block
        inCodeBlock = false;
        const langClass = codeBlockLanguage
          ? `language-${codeBlockLanguage}`
          : "";
        elements.push(
          <pre key={`code-${blockIndex}`} className="bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 my-3 overflow-x-auto">
            <code className={langClass}>{codeBlockContent}</code>
          </pre>
        );
        blockIndex++;
        codeBlockLanguage = "";
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockContent += (codeBlockContent ? "\n" : "") + line;
      continue;
    }

    // Process inline formatting
    elements.push(
      <p key={`line-${i}`} className="mb-2 leading-relaxed text-sm sm:text-base text-neutral-800 dark:text-neutral-200">
        {renderInlineFormatting(line, i)}
      </p>
    );
  }

  // Handle unclosed code block
  if (inCodeBlock && codeBlockContent) {
    elements.push(
      <pre key={`code-unclosed-${blockIndex}`} className="bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 my-3 overflow-x-auto">
        <code className={codeBlockLanguage ? `language-${codeBlockLanguage}` : ""}>
          {codeBlockContent}
        </code>
      </pre>
    );
  }

  return elements;
}

/**
 * Renders inline formatting: bold, inline code, and [Source N] references.
 */
function renderInlineFormatting(text: string, lineIndex: number): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /(\*\*(.+?)\*\*|`([^`]+)`|\[Source\s+(\d+)\])/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let partIndex = 0;

  while ((match = pattern.exec(text)) !== null) {
    // Add preceding plain text
    if (match.index > lastIndex) {
      parts.push(
        <span key={`${lineIndex}-text-${partIndex}`}>
          {text.slice(lastIndex, match.index)}
        </span>
      );
      partIndex++;
    }

    if (match[2]) {
      // Bold text
      parts.push(
        <strong key={`${lineIndex}-bold-${partIndex}`} className="font-semibold text-neutral-900 dark:text-neutral-50">{match[2]}</strong>
      );
    } else if (match[3]) {
      // Inline code
      parts.push(
        <code key={`${lineIndex}-code-${partIndex}`} className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded font-mono text-xs sm:text-sm text-red-600 dark:text-red-400">
          {match[3]}
        </code>
      );
    } else if (match[4]) {
      // Source reference badge
      parts.push(
        <span
          key={`${lineIndex}-source-${partIndex}`}
          className="inline-flex items-center gap-1 bg-blue-500/10 text-blue-600 dark:text-blue-400 text-xs font-semibold px-1.5 py-0.5 rounded align-middle"
        >
          <BookOpen size={10} />
          Source {match[4]}
        </span>
      );
    }

    lastIndex = match.index + match[0].length;
    partIndex++;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(
      <span key={`${lineIndex}-text-end`}>{text.slice(lastIndex)}</span>
    );
  }

  // If no formatting was found, return plain text
  if (parts.length === 0) {
    parts.push(<span key={`${lineIndex}-plain`}>{text}</span>);
  }

  return parts;
}

/**
 * AnswerCard displays the AI-generated answer.
 */
export default function AnswerCard({
  question,
  answerText,
  sources,
  confidence,
  onOpenFile,
  onToggle,
  mode = "bm25",
  comparison = null,
  isCollapsed: controlledCollapsed,
  onCollapsedChange,
}: AnswerCardProps) {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<"answer" | "retrieval" | "evaluation">("answer");

  // Use controlled collapsed state if provided, otherwise use internal state
  const isCollapsed = controlledCollapsed !== undefined ? controlledCollapsed : internalCollapsed;

  useEffect(() => {
    if (onToggle && !isCollapsed) {
      setTimeout(() => {
        onToggle();
      }, 50);
    }
  }, [activeTab, onToggle, isCollapsed]);

  const confidencePercent = Math.round(confidence * 100);
  const confidenceClasses = getConfidenceLevel(confidence);

  const handleToggle = () => {
    const nextCollapsed = !isCollapsed;
    
    // Update state based on whether it's controlled or uncontrolled
    if (onCollapsedChange) {
      onCollapsedChange(nextCollapsed);
    } else {
      setInternalCollapsed(nextCollapsed);
    }
    
    if (onToggle) {
      // Wait a tiny fraction of a second for DOM layout to adjust before scrolling
      setTimeout(() => {
        onToggle();
      }, 50);
    }
  };

  const isCompareMode = mode === "compare" && comparison;

  return (
    <div className="bg-white dark:bg-darkCard border border-gray-200 dark:border-gray-800 rounded-xl shadow-sm dark:shadow-black/30 transition-all duration-200">
      <div
        className={`flex items-center justify-between px-5 py-4 bg-blue-500/[0.02] hover:bg-blue-500/5 cursor-pointer select-none transition-colors duration-200 ${isCollapsed ? "" : "border-b border-gray-200 dark:border-gray-800"}`}
        onClick={handleToggle}
      >
        <div className="flex items-start gap-3 flex-1 min-w-0">
          <span className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-50 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5">
            <HelpCircle size={15} />
          </span>
          <span className="font-semibold text-sm sm:text-base leading-relaxed break-words">{question}</span>
        </div>
        <span className="flex items-center ml-4 flex-shrink-0 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
          <ChevronDown
            size={18}
            className={`transition-transform duration-250 ease-in-out ${isCollapsed ? "-rotate-90" : ""}`}
          />
        </span>
      </div>

      {!isCollapsed && (
        <div className="px-5 py-4 animate-slide-down overflow-y-auto">
          {/* Compare Tabs */}
          {isCompareMode && (
            <div className="flex border-b border-gray-200 dark:border-gray-800 mb-4 gap-4">
              <button
                onClick={() => setActiveTab("answer")}
                className={`pb-2 text-xs sm:text-sm font-semibold border-b-2 transition-all ${activeTab === "answer"
                  ? "border-blue-500 text-blue-600 dark:text-blue-400"
                  : "border-transparent text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  }`}
              >
                Grounded Answer
              </button>
              <button
                onClick={() => setActiveTab("retrieval")}
                className={`pb-2 text-xs sm:text-sm font-semibold border-b-2 transition-all ${activeTab === "retrieval"
                  ? "border-blue-500 text-blue-600 dark:text-blue-400"
                  : "border-transparent text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  }`}
              >
                Retrieval Comparison
              </button>
              <button
                onClick={() => setActiveTab("evaluation")}
                className={`pb-2 text-xs sm:text-sm font-semibold border-b-2 transition-all ${activeTab === "evaluation"
                  ? "border-blue-500 text-blue-600 dark:text-blue-400"
                  : "border-transparent text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  }`}
              >
                AI Accuracy Evaluation
              </button>
            </div>
          )}

          {(!isCompareMode || activeTab === "answer") && (
            <>
              <div className="flex items-center gap-2 mb-3">
                <span className="flex items-center justify-center w-6 h-6 rounded-full bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-400 flex-shrink-0">
                  <Sparkles size={14} />
                </span>
                <span className={`text-xs font-semibold px-2.5 py-1 rounded-full flex items-center gap-1.5 ${confidenceClasses}`}>
                  <ConfidenceIcon confidence={confidence} size={12} />
                  Confidence: {confidencePercent}%
                </span>
              </div>

              <div className="mb-4">
                {renderFormattedAnswer(answerText)}
              </div>

              {sources.length > 0 && (
                <div className="border-t border-gray-200 dark:border-gray-800 pt-4 mt-2">
                  <h4 className="text-xs font-bold mb-3 text-gray-400 dark:text-gray-500 uppercase tracking-wider flex items-center gap-1">
                    <BookOpen size={12} />
                    Sources Used
                  </h4>
                  <div className="flex flex-col gap-2">
                    {sources.map((source, idx) => (
                      <SourceReference
                        key={idx}
                        source={source}
                        index={idx}
                        onOpenFile={onOpenFile}
                      />
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {isCompareMode && activeTab === "retrieval" && (
            <div className="pt-2">
              {/* Retrieval Statistics */}
              <RetrievalStatistics
                bm25Count={comparison.bm25_sources.length}
                indobertCount={comparison.indobert_sources.length}
              />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* BM25 Chunks */}
                <div className="border border-gray-150 dark:border-gray-800 rounded-lg p-4 bg-gray-50/50 dark:bg-white/[0.01]">
                  <h4 className="text-xs font-bold mb-3 text-blue-600 dark:text-blue-400 uppercase tracking-wider flex items-center gap-1">
                    <BookOpen size={12} />
                    BM25 (Lexical Search)
                  </h4>
                  <div className="flex flex-col gap-2">
                    {comparison.bm25_sources.length > 0 ? (
                      comparison.bm25_sources.map((source, idx) => (
                        <SourceReference
                          key={idx}
                          source={source}
                          index={idx}
                          onOpenFile={onOpenFile}
                        />
                      ))
                    ) : (
                      <span className="text-xs text-gray-400">Tidak ada file yang dicocokkan.</span>
                    )}
                  </div>
                </div>

                {/* IndoBERT Chunks */}
                <div className="border border-gray-150 dark:border-gray-800 rounded-lg p-4 bg-gray-50/50 dark:bg-white/[0.01]">
                  <h4 className="text-xs font-bold mb-3 text-purple-600 dark:text-purple-400 uppercase tracking-wider flex items-center gap-1">
                    <BookOpen size={12} />
                    IndoBERT (Semantic Search)
                  </h4>
                  <div className="flex flex-col gap-2">
                    {comparison.indobert_sources.length > 0 ? (
                      comparison.indobert_sources.map((source, idx) => (
                        <SourceReference
                          key={idx}
                          source={source}
                          index={idx}
                          onOpenFile={onOpenFile}
                        />
                      ))
                    ) : (
                      <span className="text-xs text-gray-400">Tidak ada file yang dicocokkan.</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {isCompareMode && activeTab === "evaluation" && (
            <div className="pt-2 animate-fade-in">
              <div className="bg-blue-500/5 border border-blue-500/10 rounded-xl p-5">
                <h4 className="text-xs font-bold mb-3 text-blue-600 dark:text-blue-400 uppercase tracking-wider flex items-center gap-1">
                  <BookOpen size={12} />
                  Analisis Perbandingan Akurasi AI
                </h4>
                <div className="prose dark:prose-invert max-w-none text-neutral-800 dark:text-neutral-200">
                  {renderFormattedAnswer(comparison.evaluation)}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

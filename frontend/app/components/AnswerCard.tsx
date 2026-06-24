"use client";

import React from "react";
import SourceReference, { SourceReferenceData } from "./SourceReference";

export interface AnswerCardProps {
  question: string;
  answerText: string;
  sources: SourceReferenceData[];
  confidence: number;
  onOpenFile?: (filePath: string, line: number) => void;
}

/**
 * Returns a CSS class name for confidence score color coding.
 */
function getConfidenceLevel(confidence: number): string {
  if (confidence >= 0.8) return "confidence-high";
  if (confidence >= 0.5) return "confidence-medium";
  return "confidence-low";
}

/**
 * Renders answer text with basic markdown-like formatting.
 * Handles:
 * - Code blocks (triple backticks with optional language)
 * - Inline code (single backticks)
 * - Bold text (**text**)
 * - Line breaks
 * - [Source N] references rendered as badges
 */
function renderFormattedAnswer(text: string): React.ReactNode[] {
  const elements: React.ReactNode[] = [];
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
          <pre key={`code-${blockIndex}`} className="answer-code-block">
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
      <p key={`line-${i}`} className="answer-line">
        {renderInlineFormatting(line, i)}
      </p>
    );
  }

  // Handle unclosed code block
  if (inCodeBlock && codeBlockContent) {
    elements.push(
      <pre key={`code-unclosed-${blockIndex}`} className="answer-code-block">
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
  // Match **bold**, `inline code`, and [Source N] references
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
        <strong key={`${lineIndex}-bold-${partIndex}`}>{match[2]}</strong>
      );
    } else if (match[3]) {
      // Inline code
      parts.push(
        <code key={`${lineIndex}-code-${partIndex}`} className="inline-code">
          {match[3]}
        </code>
      );
    } else if (match[4]) {
      // Source reference badge
      parts.push(
        <span
          key={`${lineIndex}-source-${partIndex}`}
          className="source-badge"
        >
          [Source {match[4]}]
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
 * AnswerCard displays the AI-generated answer with:
 * - Formatted answer text (markdown-like)
 * - Source references as clickable expandable elements
 * - Code snippets with syntax highlighting
 * - Confidence score badge
 *
 * Requirements: 13.2, 13.3
 */
export default function AnswerCard({
  question,
  answerText,
  sources,
  confidence,
  onOpenFile,
}: AnswerCardProps) {
  const confidencePercent = Math.round(confidence * 100);
  const confidenceLevel = getConfidenceLevel(confidence);

  return (
    <div className="answer-card">
      <div className="answer-question">
        <span className="question-icon">Q</span>
        <span className="question-text">{question}</span>
      </div>

      <div className="answer-body">
        <div className="answer-header">
          <span className="answer-icon">A</span>
          <span className={`confidence-badge ${confidenceLevel}`}>
            Confidence: {confidencePercent}%
          </span>
        </div>

        <div className="answer-content">
          {renderFormattedAnswer(answerText)}
        </div>

        {sources.length > 0 && (
          <div className="answer-sources">
            <h4 className="sources-heading">Sources</h4>
            {sources.map((source, idx) => (
              <SourceReference
                key={idx}
                source={source}
                index={idx}
                onOpenFile={onOpenFile}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

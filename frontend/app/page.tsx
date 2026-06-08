"use client";

import React, { useState, useRef, useEffect, FormEvent } from "react";
import AnswerCard, { AnswerCardProps } from "./components/AnswerCard";

/**
 * Maximum number of Q&A pairs to keep in session history.
 * Requirement 13.4: at least 50 most recent pairs.
 */
const MAX_HISTORY_SIZE = 50;

/**
 * Represents a single question-and-answer pair in the history.
 */
interface QAPair {
  id: string;
  question: string;
  answerText: string;
  sources: AnswerCardProps["sources"];
  confidence: number;
  timestamp: Date;
}

/**
 * API response structure from backend POST /api/query.
 */
interface QueryResponse {
  answer: string;
  sources: Array<{
    file_path: string;
    function_name?: string | null;
    start_line: number;
    end_line: number;
    snippet: string;
    relevance: number;
  }>;
  confidence: number;
  metadata: Record<string, unknown>;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<QAPair[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to latest message when history changes
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, isLoading]);

  /**
   * Submit question to the backend API and add result to history.
   */
  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || trimmedQuestion.length > 1000) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": "user-session-key",
        },
        body: JSON.stringify({ question: trimmedQuestion }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || `Request failed with status ${response.status}`
        );
      }

      const data: QueryResponse = await response.json();

      const newPair: QAPair = {
        id: `qa-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        question: trimmedQuestion,
        answerText: data.answer,
        sources: data.sources,
        confidence: data.confidence,
        timestamp: new Date(),
      };

      setHistory((prev) => {
        const updated = [...prev, newPair];
        // Keep only the most recent MAX_HISTORY_SIZE items
        if (updated.length > MAX_HISTORY_SIZE) {
          return updated.slice(updated.length - MAX_HISTORY_SIZE);
        }
        return updated;
      });

      setQuestion("");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  /**
   * Retry the last failed question without re-entering it.
   * Requirement 13.6.
   */
  function handleRetry() {
    if (question.trim()) {
      setError(null);
      handleSubmit(new Event("submit") as unknown as FormEvent);
    }
  }

  /**
   * Handle keyboard shortcut (Ctrl+Enter or Cmd+Enter to submit).
   */
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleSubmit(e as unknown as FormEvent);
    }
  }

  return (
    <main className="chat-container">
      <header className="chat-header">
        <h1>CodeQ-Mate</h1>
        <p>Context-aware question answering for software repositories</p>
      </header>

      {/* Chat history area - scrollable */}
      <div className="chat-history">
        {history.length === 0 && !isLoading && (
          <div className="chat-empty-state">
            <p>Ask a question about your codebase to get started.</p>
          </div>
        )}

        {history.map((pair) => (
          <AnswerCard
            key={pair.id}
            question={pair.question}
            answerText={pair.answerText}
            sources={pair.sources}
            confidence={pair.confidence}
          />
        ))}

        {isLoading && (
          <div className="loading-indicator">
            <div className="loading-spinner" />
            <span>Searching codebase and generating answer...</span>
          </div>
        )}

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠</span>
            <span>{error}</span>
            <button
              className="retry-button"
              onClick={handleRetry}
              type="button"
            >
              Retry
            </button>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Query input area */}
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your code... (Ctrl+Enter to submit)"
            maxLength={1000}
            rows={2}
            disabled={isLoading}
            aria-label="Question input"
          />
          <div className="input-footer">
            <span className="char-count">
              {question.length}/1000
            </span>
            <button
              type="submit"
              className="submit-button"
              disabled={isLoading || !question.trim()}
            >
              {isLoading ? "Sending..." : "Ask"}
            </button>
          </div>
        </div>
      </form>
    </main>
  );
}

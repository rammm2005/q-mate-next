"use client";

import React, { useState, useRef, useEffect, FormEvent } from "react";
import AnswerCard, { AnswerCardProps } from "./components/AnswerCard";
import FileTree, { FileTreeNode } from "./components/FileTree";
import FileViewer from "./components/FileViewer";

const MAX_HISTORY_SIZE = 50;

interface QAPair {
  id: string;
  question: string;
  answerText: string;
  sources: AnswerCardProps["sources"];
  confidence: number;
}

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

interface IngestResponse {
  status: string;
  repo_name: string;
  total_files: number;
  total_chunks: number;
  languages: string[];
  error?: string;
}

interface RepoStatus {
  is_indexed: boolean;
  repo_name: string;
  total_chunks: number;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState<QAPair[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ingest state
  const [githubUrl, setGithubUrl] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResponse | null>(null);
  const [repoStatus, setRepoStatus] = useState<RepoStatus | null>(null);
  const [fileTree, setFileTree] = useState<FileTreeNode[]>([]);
  const [fileTreeRepoName, setFileTreeRepoName] = useState("");
  
  // File viewer state
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, isLoading]);

  // Check status on load
  useEffect(() => {
    fetch("/api/status")
      .then((r) => r.json())
      .then((data: RepoStatus) => {
        setRepoStatus(data);
        if (data.is_indexed) fetchFileTree();
      })
      .catch(() => {});
  }, []);

  function fetchFileTree() {
    fetch("/api/filetree")
      .then((r) => r.json())
      .then((data) => {
        setFileTree(data.tree || []);
        setFileTreeRepoName(data.repo_name || "");
      })
      .catch(() => {});
  }

  async function handleIngest(e: FormEvent) {
    e.preventDefault();
    const url = githubUrl.trim();
    if (!url) return;

    setIsIngesting(true);
    setIngestResult(null);
    setError(null);

    try {
      const response = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ github_url: url }),
      });

      const data: IngestResponse = await response.json();
      setIngestResult(data);

      if (data.status === "success") {
        setRepoStatus({
          is_indexed: true,
          repo_name: data.repo_name,
          total_chunks: data.total_chunks,
        });
        setHistory([]); // Clear old history for new repo
        fetchFileTree(); // Load file tree
      }
    } catch (err) {
      setError("Failed to connect to backend. Make sure the server is running.");
    } finally {
      setIsIngesting(false);
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || trimmed.length > 1000) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Request failed (${response.status})`);
      }

      const data: QueryResponse = await response.json();

      const newPair: QAPair = {
        id: `qa-${Date.now()}`,
        question: trimmed,
        answerText: data.answer,
        sources: data.sources,
        confidence: data.confidence,
      };

      setHistory((prev) => {
        const updated = [...prev, newPair];
        return updated.length > MAX_HISTORY_SIZE
          ? updated.slice(updated.length - MAX_HISTORY_SIZE)
          : updated;
      });

      setQuestion("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsLoading(false);
    }
  }

  function handleRetry() {
    if (question.trim()) {
      setError(null);
      handleSubmit(new Event("submit") as unknown as FormEvent);
    }
  }

  const isRepoReady = repoStatus?.is_indexed ?? false;

  return (
    <div className="app-layout">
      {/* File Tree Sidebar */}
      {isRepoReady && fileTree.length > 0 && (
        <aside className="sidebar">
          <FileTree
            tree={fileTree}
            repoName={fileTreeRepoName}
            onFileClick={(path) => {
              setSelectedFilePath(path); // Open file viewer instead
            }}
          />
        </aside>
      )}

      <main className="chat-container">
      <header className="chat-header">
        <h1>CodeQ-Mate</h1>
        <p>Context-aware question answering for software repositories</p>
      </header>

      {/* Repository Ingestion Section */}
      <div className="ingest-section">
        <form className="ingest-form" onSubmit={handleIngest}>
          <input
            type="text"
            className="ingest-input"
            value={githubUrl}
            onChange={(e) => setGithubUrl(e.target.value)}
            placeholder="Paste GitHub URL (e.g. https://github.com/pallets/flask)"
            disabled={isIngesting}
          />
          <button
            type="submit"
            className="ingest-button"
            disabled={isIngesting || !githubUrl.trim()}
          >
            {isIngesting ? "Cloning..." : "Index Repository"}
          </button>
        </form>

        {isIngesting && (
          <div className="ingest-progress">
            <div className="loading-spinner" />
            <span>Cloning repository and indexing code... This may take a minute.</span>
          </div>
        )}

        {ingestResult && ingestResult.status === "success" && (
          <div className="ingest-success">
            ✅ Indexed <strong>{ingestResult.repo_name}</strong>: {ingestResult.total_files} files, {ingestResult.total_chunks} code chunks
            {ingestResult.languages.length > 0 && (
              <span> ({ingestResult.languages.join(", ")})</span>
            )}
          </div>
        )}

        {ingestResult && ingestResult.status === "error" && (
          <div className="ingest-error">
            ❌ {ingestResult.error}
          </div>
        )}

        {isRepoReady && !isIngesting && !ingestResult && (
          <div className="ingest-success">
            📦 Repository <strong>{repoStatus!.repo_name}</strong> indexed ({repoStatus!.total_chunks} chunks ready)
          </div>
        )}
      </div>

      {/* Chat History */}
      <div className="chat-history">
        {!isRepoReady && history.length === 0 && !isLoading && (
          <div className="chat-empty-state">
            <p>👆 Start by indexing a GitHub repository above, then ask questions about the code.</p>
          </div>
        )}

        {isRepoReady && history.length === 0 && !isLoading && (
          <div className="chat-empty-state">
            <p>Repository ready! Ask a question about the code below.</p>
          </div>
        )}

        {history.map((pair) => (
          <AnswerCard
            key={pair.id}
            question={pair.question}
            answerText={pair.answerText}
            sources={pair.sources}
            confidence={pair.confidence}
            onOpenFile={(filePath, line) => {
              setSelectedFilePath(filePath);
            }}
          />
        ))}

        {isLoading && (
          <div className="loading-indicator">
            <div className="loading-spinner" />
            <span>Searching codebase...</span>
          </div>
        )}

        {error && (
          <div className="error-message">
            <span>⚠ {error}</span>
            <button className="retry-button" onClick={handleRetry} type="button">
              Retry
            </button>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      {/* Query Input */}
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <textarea
            className="chat-input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
                e.preventDefault();
                handleSubmit(e as unknown as FormEvent);
              }
            }}
            placeholder={
              isRepoReady
                ? "Ask about the code... (Ctrl+Enter to submit)"
                : "Index a repository first..."
            }
            maxLength={1000}
            rows={2}
            disabled={isLoading || !isRepoReady}
          />
          <div className="input-footer">
            <span className="char-count">{question.length}/1000</span>
            <button
              type="submit"
              className="submit-button"
              disabled={isLoading || !question.trim() || !isRepoReady}
            >
              {isLoading ? "Searching..." : "Ask"}
            </button>
          </div>
        </div>
      </form>
    </main>
    
    {/* File Viewer Modal */}
    {selectedFilePath && (
      <FileViewer
        filePath={selectedFilePath}
        onClose={() => setSelectedFilePath(null)}
      />
    )}
    </div>
  );
}

"use client";

import React, { useState, useRef, useEffect, FormEvent } from "react";
import AnswerCard, { AnswerCardProps } from "./components/AnswerCard";
import FileTree, { FileTreeNode } from "./components/FileTree";
import FileViewer from "./components/FileViewer";
import {
  Search,
  FolderGit,
  Loader2,
  RotateCcw,
  AlertTriangle,
  PanelLeftClose,
  PanelLeft,
  Database,
  Brain,
  GitCompare,
  CheckCircle2,
  Info,
  X
} from "lucide-react";

const GithubIcon = ({ size = 18 }: { size?: number }) => (
  <svg
    viewBox="0 0 24 24"
    width={size}
    height={size}
    stroke="currentColor"
    strokeWidth="2"
    fill="none"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

const MAX_HISTORY_SIZE = 50;

interface ComparisonData {
  bm25_sources: Array<{
    file_path: string;
    function_name?: string | null;
    start_line: number;
    end_line: number;
    snippet: string;
    relevance: number;
  }>;
  indobert_sources: Array<{
    file_path: string;
    function_name?: string | null;
    start_line: number;
    end_line: number;
    snippet: string;
    relevance: number;
  }>;
  evaluation: string;
}

interface QAPair {
  id: string;
  question: string;
  answerText: string;
  sources: AnswerCardProps["sources"];
  confidence: number;
  mode?: string;
  comparison?: ComparisonData | null;
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
  comparison?: ComparisonData | null;
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
  const [expandedAnswerId, setExpandedAnswerId] = useState<string | null>(null);

  // Toast notification state
  interface ToastItem {
    id: string;
    message: string;
    type: "success" | "error" | "info";
  }
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  useEffect(() => {
    const handleToast = (e: Event) => {
      const customEvent = e as CustomEvent<{ message: string; type?: "success" | "error" | "info" }>;
      const { message, type = "info" } = customEvent.detail;
      const id = `toast-${Date.now()}-${Math.random()}`;
      
      setToasts((prev) => [...prev, { id, message, type }]);
      
      setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, 3000);
    };

    window.addEventListener("app-toast" as any, handleToast);
    return () => window.removeEventListener("app-toast" as any, handleToast);
  }, []);

  // Ingest state
  const [githubUrl, setGithubUrl] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResponse | null>(null);
  const [repoStatus, setRepoStatus] = useState<RepoStatus | null>(null);
  const [fileTree, setFileTree] = useState<FileTreeNode[]>([]);
  const [fileTreeRepoName, setFileTreeRepoName] = useState("");
  
  // Sidebar state
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // Retrieval mode state
  const [retrievalMode, setRetrievalMode] = useState<"bm25" | "indobert" | "compare">("bm25");

  // File viewer state
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [selectedStartLine, setSelectedStartLine] = useState<number | undefined>(undefined);
  const [selectedEndLine, setSelectedEndLine] = useState<number | undefined>(undefined);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const chatHistoryRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const scrollToBottom = () => {
      if (chatHistoryRef.current) {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
      }
    };

    // Only scroll on new messages, not on layout changes
    if (isLoading || history.length > 0) {
      scrollToBottom();
    }
  }, [history.length, isLoading]); // Removed layoutChangeTrigger to prevent unnecessary scrolls

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

    // Dispatch info toast
    window.dispatchEvent(new CustomEvent("app-toast", {
      detail: { message: "Cloning and indexing repository...", type: "info" }
    }));

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
        setSelectedFilePath(null); // Reset open file viewer to prevent showing old repository files
        fetchFileTree(); // Load file tree

        // Success toast
        window.dispatchEvent(new CustomEvent("app-toast", {
          detail: { message: `Successfully indexed ${data.repo_name}!`, type: "success" }
        }));
      } else {
        // Error toast
        window.dispatchEvent(new CustomEvent("app-toast", {
          detail: { message: `Ingestion failed: ${data.error}`, type: "error" }
        }));
      }
    } catch (err) {
      setError("Failed to connect to backend. Make sure the server is running.");
      window.dispatchEvent(new CustomEvent("app-toast", {
        detail: { message: "Failed to connect to backend.", type: "error" }
      }));
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
        body: JSON.stringify({ question: trimmed, mode: retrievalMode }),
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
        mode: retrievalMode,
        comparison: data.comparison,
      };

      setHistory((prev) => {
        const updated = [...prev, newPair];
        return updated.length > MAX_HISTORY_SIZE
          ? updated.slice(updated.length - MAX_HISTORY_SIZE)
          : updated;
      });

      // Auto-expand the newly added answer
      setExpandedAnswerId(newPair.id);

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
    <div className="flex h-screen overflow-hidden bg-gray-50 dark:bg-darkBg text-neutral-900 dark:text-neutral-100 font-sans transition-colors duration-200">
      {/* File Tree Sidebar */}
      {isRepoReady && fileTree.length > 0 && (
        <aside className={`${isSidebarOpen ? "w-[280px] min-w-[280px]" : "w-0 overflow-hidden border-r-0"} border-r border-gray-200 dark:border-gray-800 transition-all duration-300 bg-white dark:bg-darkCard flex flex-col`}>
          <FileTree
            tree={fileTree}
            repoName={fileTreeRepoName}
            onFileClick={(path) => {
              setSelectedFilePath(path);
              setSelectedStartLine(undefined);
              setSelectedEndLine(undefined);
            }}
          />
        </aside>
      )}

      <main className="flex flex-col h-full flex-1 min-w-0 max-w-[900px] mx-auto px-4 sm:px-6 overflow-hidden">
        {/* Header */}
        <header className="py-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-3">
            {isRepoReady && fileTree.length > 0 && (
              <button
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                className="p-2 rounded-lg border border-gray-200 dark:border-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-gray-600 dark:text-gray-400 cursor-pointer"
                title={isSidebarOpen ? "Collapse Sidebar" : "Expand Sidebar"}
              >
                {isSidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeft size={18} />}
              </button>
            )}
            <div className="text-left">
              <h1 className="text-lg sm:text-xl font-bold flex items-center gap-2">
                <Database className="text-blue-600 dark:text-blue-400" size={20} />
                CodeQ-Mate
              </h1>
              <p className="text-gray-500 dark:text-gray-400 text-xs">Context-aware question answering for software repositories</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="p-2 rounded-lg border border-gray-200 dark:border-gray-800 hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 hover:text-neutral-900 dark:hover:text-neutral-100 transition-colors"
              title="GitHub"
            >
              <GithubIcon size={18} />
            </a>
          </div>
        </header>

        {/* Repository Ingestion Section */}
        <div className="py-4 border-b border-gray-200 dark:border-gray-800 flex-shrink-0">
          <form className="flex gap-2" onSubmit={handleIngest}>
            <input
              type="text"
              className="flex-1 px-3 py-2 border border-gray-200 dark:border-gray-800 rounded-lg text-sm bg-white dark:bg-darkCard outline-none transition-all focus:border-blue-500 focus:ring-1 focus:ring-blue-500 placeholder-gray-400 dark:placeholder-gray-500"
              value={githubUrl}
              onChange={(e) => setGithubUrl(e.target.value)}
              placeholder="Paste GitHub URL (e.g. https://github.com/pallets/flask)"
              disabled={isIngesting}
            />
            <button
              type="submit"
              className="px-5 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap flex items-center gap-1.5 cursor-pointer shadow-sm hover:shadow"
              disabled={isIngesting || !githubUrl.trim()}
            >
              {isIngesting ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Cloning...
                </>
              ) : (
                <>
                  <FolderGit size={16} />
                  Index Repository
                </>
              )}
            </button>
          </form>

          {isIngesting && (
            <div className="flex items-center gap-2 mt-3 text-xs text-gray-500 dark:text-gray-400">
              <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              <span>Cloning repository and indexing code... This may take a minute.</span>
            </div>
          )}

          {ingestResult && ingestResult.status === "success" && (
            <div className="mt-3 p-2.5 bg-green-500/10 border border-green-500/20 rounded-md text-xs sm:text-sm text-green-600 dark:text-green-400 flex items-center gap-2">
              <span>✅ Indexed <strong>{ingestResult.repo_name}</strong>: {ingestResult.total_files} files, {ingestResult.total_chunks} code chunks {ingestResult.languages.length > 0 && `(${ingestResult.languages.join(", ")})`}</span>
            </div>
          )}

          {ingestResult && ingestResult.status === "error" && (
            <div className="mt-3 p-2.5 bg-red-500/10 border border-red-500/20 rounded-md text-xs sm:text-sm text-red-600 dark:text-red-400 flex items-center gap-2">
              <AlertTriangle size={16} />
              <span>{ingestResult.error}</span>
            </div>
          )}

          {isRepoReady && !isIngesting && !ingestResult && (
            <div className="mt-3 p-2.5 bg-green-500/10 border border-green-500/20 rounded-md text-xs sm:text-sm text-green-600 dark:text-green-400">
              📦 Repository <strong>{repoStatus!.repo_name}</strong> indexed ({repoStatus!.total_chunks} chunks ready)
            </div>
          )}
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto pr-2 flex flex-col gap-6 min-h-0" ref={chatHistoryRef}>
          {!isRepoReady && history.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full text-center p-8 text-gray-500 dark:text-gray-400">
              <FolderGit className="w-12 h-12 mb-4 text-blue-500/80 animate-pulse" />
              <p className="text-sm font-medium">Start by indexing a GitHub repository above, then ask questions about the code.</p>
            </div>
          )}

          {isRepoReady && history.length === 0 && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full text-center p-8 text-gray-500 dark:text-gray-400">
              <Database className="w-12 h-12 mb-4 text-blue-500/80" />
              <p className="text-sm font-medium">Repository ready! Ask a question about the code below.</p>
            </div>
          )}

          {history.map((pair) => (
            <AnswerCard
              key={pair.id}
              question={pair.question}
              answerText={pair.answerText}
              sources={pair.sources}
              confidence={pair.confidence}
              mode={pair.mode}
              comparison={pair.comparison}
              onOpenFile={(filePath, startLine, endLine) => {
                setSelectedFilePath(filePath);
                setSelectedStartLine(startLine);
                setSelectedEndLine(endLine);
              }}
              isCollapsed={expandedAnswerId !== null && expandedAnswerId !== pair.id}
              onCollapsedChange={(collapsed) => {
                setExpandedAnswerId(collapsed ? null : pair.id);
              }}
            />
          ))}

          {isLoading && (
            <div className="flex items-center gap-3 p-4 bg-gray-100/50 dark:bg-white/5 rounded-lg border border-gray-100 dark:border-gray-800 text-gray-500 dark:text-gray-400 text-sm">
              <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
              <span>Searching codebase...</span>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-2 px-4 py-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-600 dark:text-red-400 text-sm">
              <AlertTriangle size={16} className="text-red-500" />
              <span>⚠ {error}</span>
              <button
                className="ml-auto px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-xs font-medium transition-colors flex items-center gap-1 cursor-pointer"
                onClick={handleRetry}
                type="button"
              >
                <RotateCcw size={12} />
                Retry
              </button>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Query Input */}
        <form className="py-4 border-t border-gray-200 dark:border-gray-800 flex-shrink-0" onSubmit={handleSubmit}>
          <div className="border border-gray-200 dark:border-gray-800 rounded-xl overflow-hidden bg-white dark:bg-darkCard transition-all focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500 shadow-sm">
            <textarea
              className="w-full px-4 py-3 border-none outline-none resize-none bg-transparent text-sm leading-relaxed placeholder-gray-400 dark:placeholder-gray-500 text-neutral-950 dark:text-neutral-50"
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
             <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-3 py-2 border-t border-gray-100 dark:border-gray-800/80 bg-gray-50/50 dark:bg-white/[0.01]">
              <div className="flex items-center flex-wrap gap-2.5">
                <div className="flex items-center gap-1.5 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
                  <button
                    type="button"
                    onClick={() => setRetrievalMode("bm25")}
                    className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all flex items-center gap-1 cursor-pointer select-none ${
                      retrievalMode === "bm25"
                        ? "bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm"
                        : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                    }`}
                  >
                    <Search size={14} /> BM25
                  </button>
                  <button
                    type="button"
                    onClick={() => setRetrievalMode("indobert")}
                    className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all flex items-center gap-1 cursor-pointer select-none ${
                      retrievalMode === "indobert"
                        ? "bg-white dark:bg-gray-700 text-purple-600 dark:text-purple-400 shadow-sm"
                        : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                    }`}
                  >
                    <Brain size={14} /> IndoBERT
                  </button>
                  <button
                    type="button"
                    onClick={() => setRetrievalMode("compare")}
                    className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all flex items-center gap-1 cursor-pointer select-none ${
                      retrievalMode === "compare"
                        ? "bg-white dark:bg-gray-700 text-indigo-600 dark:text-indigo-400 shadow-sm"
                        : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                    }`}
                  >
                    <GitCompare size={14} /> Compare
                  </button>
                </div>
                <span className="text-xs text-gray-400 dark:text-gray-500 font-medium">{question.length}/1000</span>
              </div>
              <button
                type="submit"
                className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-xs font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-1 cursor-pointer"
                disabled={isLoading || !question.trim() || !isRepoReady}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search size={12} />
                    Ask
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      </main>
      {/* File Viewer Modal */}
      {selectedFilePath && (
        <FileViewer
          filePath={selectedFilePath}
          onClose={() => {
            setSelectedFilePath(null);
            setSelectedStartLine(undefined);
            setSelectedEndLine(undefined);
          }}
          startLine={selectedStartLine}
          endLine={selectedEndLine}
        />
      )}

      {/* Toast Notifications */}
      <div className="fixed bottom-5 right-5 z-[10000] flex flex-col gap-2 max-w-sm w-full pointer-events-none">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className="pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-lg border border-gray-200 dark:border-gray-800 bg-white/95 dark:bg-darkCard/95 shadow-xl backdrop-blur-md animate-slide-in-right transition-all duration-300"
          >
            {toast.type === "success" && <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0" />}
            {toast.type === "error" && <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />}
            {toast.type === "info" && <Info className="w-5 h-5 text-blue-500 flex-shrink-0" />}
            <span className="text-sm font-medium text-neutral-800 dark:text-neutral-200 flex-1">{toast.message}</span>
            <button
              onClick={() => setToasts((prev) => prev.filter((t) => t.id !== toast.id))}
              className="p-0.5 rounded-md hover:bg-gray-150 dark:hover:bg-gray-800 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors cursor-pointer"
            >
              <X size={14} />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

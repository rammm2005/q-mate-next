"use client";

import React, { useState, useEffect } from "react";
import {
  X,
  FileCode,
  Navigation,
  Copy,
  Download,
  ExternalLink,
  Loader2,
  AlertTriangle,
  Eye,
  Code2,
  BookOpen
} from "lucide-react";

interface FileViewerProps {
  filePath: string;
  onClose: () => void;
  initialLine?: number;
  startLine?: number;
  endLine?: number;
}

interface FileContent {
  file_path: string;
  content: string;
  language: string;
  total_lines: number;
  size_bytes?: number;
  from_chunks?: boolean;
}

interface EditorUrls {
  vscode: string;
  vscode_insiders: string;
  intellij: string;
  sublime: string;
  atom: string;
}

/**
 * Parses markdown inline formatting (bold, code, links).
 */
function renderInlineMarkdownFormatting(text: string, lineIndex: number): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /(\*\*(.+?)\*\*|`([^`]+)`)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let partIndex = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <span key={`${lineIndex}-text-${partIndex}`}>
          {text.slice(lastIndex, match.index)}
        </span>
      );
      partIndex++;
    }

    if (match[2]) {
      parts.push(
        <strong key={`${lineIndex}-bold-${partIndex}`} className="font-bold text-neutral-900 dark:text-neutral-50">
          {match[2]}
        </strong>
      );
    } else if (match[3]) {
      parts.push(
        <code key={`${lineIndex}-code-${partIndex}`} className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded font-mono text-xs text-red-600 dark:text-red-400">
          {match[3]}
        </code>
      );
    }

    lastIndex = match.index + match[0].length;
    partIndex++;
  }

  if (lastIndex < text.length) {
    parts.push(
      <span key={`${lineIndex}-text-end`}>{text.slice(lastIndex)}</span>
    );
  }

  if (parts.length === 0) {
    parts.push(<span key={`${lineIndex}-plain`}>{text}</span>);
  }

  return parts;
}

/**
 * Renders raw Markdown content as formatted HTML structure.
 */
function renderFormattedMarkdown(text: string): React.ReactNode[] {
  const elements: React.ReactNode[] = [];
  if (!text) return elements;
  const lines = text.split("\n");
  let inCodeBlock = false;
  let codeBlockContent = "";
  let codeBlockLanguage = "";
  let blockIndex = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Code block check
    if (line.trimStart().startsWith("```")) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeBlockLanguage = line.trimStart().slice(3).trim();
        codeBlockContent = "";
      } else {
        inCodeBlock = false;
        elements.push(
          <pre key={`md-code-${blockIndex}`} className="bg-gray-100 dark:bg-gray-800/80 border border-gray-250 dark:border-gray-800 rounded-lg p-3 my-3 overflow-x-auto font-mono text-xs sm:text-sm">
            <code>{codeBlockContent}</code>
          </pre>
        );
        blockIndex++;
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockContent += (codeBlockContent ? "\n" : "") + line;
      continue;
    }

    // Headers
    if (line.startsWith("#")) {
      const match = line.match(/^(#{1,6})\s+(.*)$/);
      if (match) {
        const level = match[1].length;
        const headingText = match[2];
        const headingClasses = 
          level === 1 ? "text-2xl font-bold mt-6 mb-3 border-b pb-2 text-neutral-905 dark:text-neutral-50" :
          level === 2 ? "text-xl font-bold mt-5 mb-2.5 text-neutral-905 dark:text-neutral-50" :
          level === 3 ? "text-lg font-bold mt-4 mb-2 text-neutral-905 dark:text-neutral-50" :
          "text-base font-semibold mt-3 mb-2 text-neutral-905 dark:text-neutral-50";
        
        elements.push(
          React.createElement(`h${level}`, { key: `md-h-${i}`, className: headingClasses }, headingText)
        );
        continue;
      }
    }

    // Blockquote
    if (line.startsWith(">")) {
      const quoteText = line.slice(1).trim();
      elements.push(
        <blockquote key={`md-bq-${i}`} className="border-l-4 border-blue-500/50 dark:border-blue-400/50 pl-4 py-1 my-3 text-gray-500 dark:text-gray-400 italic bg-blue-500/[0.02] rounded-r">
          {quoteText}
        </blockquote>
      );
      continue;
    }

    // Horizontal rule
    if (line.trim() === "---" || line.trim() === "***" || line.trim() === "___") {
      elements.push(<hr key={`md-hr-${i}`} className="my-5 border-gray-250 dark:border-gray-800" />);
      continue;
    }

    // List items
    if (line.trim().startsWith("- ") || line.trim().startsWith("* ")) {
      const itemText = line.trim().slice(2);
      elements.push(
        <li key={`md-li-${i}`} className="ml-4 list-disc text-sm sm:text-base text-neutral-800 dark:text-neutral-200 py-0.5">
          {renderInlineMarkdownFormatting(itemText, i)}
        </li>
      );
      continue;
    }

    if (/^\d+\.\s+/.test(line.trim())) {
      const dotIndex = line.trim().indexOf(".");
      const itemText = line.trim().slice(dotIndex + 1).trim();
      elements.push(
        <li key={`md-oli-${i}`} className="ml-4 list-decimal text-sm sm:text-base text-neutral-800 dark:text-neutral-200 py-0.5">
          {renderInlineMarkdownFormatting(itemText, i)}
        </li>
      );
      continue;
    }

    // Paragraph
    if (line.trim()) {
      elements.push(
        <p key={`md-p-${i}`} className="mb-2 leading-relaxed text-sm sm:text-base text-neutral-800 dark:text-neutral-200">
          {renderInlineMarkdownFormatting(line, i)}
        </p>
      );
    }
  }

  return elements;
}

/**
 * Modal component for viewing file content (read-only)
 */
export default function FileViewer({ filePath, onClose, initialLine = 1, startLine, endLine }: FileViewerProps) {
  const [content, setContent] = useState<FileContent | null>(null);
  const [editorUrls, setEditorUrls] = useState<EditorUrls | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentLine, setCurrentLine] = useState(initialLine);
  
  // viewMode can be "code" or "preview"
  const [viewMode, setViewMode] = useState<"code" | "preview">("code");

  // Keyboard shortcut: ESC to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  useEffect(() => {
    loadFileContent();
    loadEditorUrls();
    
    // Scroll to highlighted line after content loads
    if (startLine) {
      setTimeout(() => {
        const lineElement = document.getElementById(`line-${startLine}`);
        if (lineElement) {
          lineElement.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 300);
    }
  }, [filePath]);

  const loadFileContent = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/file?path=${encodeURIComponent(filePath)}&action=content`);
      if (!response.ok) {
        throw new Error(`Failed to load file: ${response.status}`);
      }
      const data = await response.json();
      setContent(data);

      // Determine initial view mode based on file extension
      const ext = filePath.split(".").pop()?.toLowerCase();
      if (ext === "md" || ext === "ipynb") {
        setViewMode("preview");
      } else {
        setViewMode("code");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load file");
    } finally {
      setLoading(false);
    }
  };

  const loadEditorUrls = async () => {
    try {
      const response = await fetch(`/api/file?path=${encodeURIComponent(filePath)}&action=open&line=${currentLine}`);
      if (response.ok) {
        const data = await response.json();
        setEditorUrls(data.editor_urls);
      }
    } catch (err) {
      console.error("Failed to load editor URLs:", err);
    }
  };

  const openInEditor = (editorType: keyof EditorUrls) => {
    if (!editorUrls) return;
    const url = editorUrls[editorType];
    window.location.href = url;
  };

  const downloadFile = () => {
    if (!content) return;
    const blob = new Blob([content.content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filePath.split("/").pop() || "file.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const copyToClipboard = () => {
    if (!content) return;
    navigator.clipboard.writeText(content.content);
    window.dispatchEvent(new CustomEvent("app-toast", {
      detail: { message: "Content copied to clipboard!", type: "success" }
    }));
  };

  const jumpToLine = () => {
    const lineInput = prompt(`Jump to line (1-${content?.total_lines || 1}):`, String(currentLine));
    if (lineInput) {
      const line = parseInt(lineInput, 10);
      if (line >= 1 && line <= (content?.total_lines || 1)) {
        setCurrentLine(line);
        const lineElement = document.getElementById(`line-${line}`);
        if (lineElement) {
          lineElement.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    }
  };

  const isMarkdown = filePath.endsWith(".md");
  const isIPYNB = filePath.endsWith(".ipynb");
  const supportsPreview = isMarkdown || isIPYNB;

  const renderNotebook = (jsonContent: string) => {
    try {
      const notebook = JSON.parse(jsonContent);
      const cells = notebook.cells || [];

      return (
        <div className="p-6 bg-white dark:bg-darkCard space-y-6 max-w-4xl mx-auto font-sans leading-relaxed">
          {cells.map((cell: any, index: number) => {
            const source = Array.isArray(cell.source) ? cell.source.join("") : cell.source || "";

            if (cell.cell_type === "markdown") {
              return (
                <div key={index} className="prose dark:prose-invert max-w-none pb-2 border-b border-gray-100 dark:border-gray-800/80">
                  {renderFormattedMarkdown(source)}
                </div>
              );
            } else if (cell.cell_type === "code") {
              const outputs = cell.outputs || [];
              return (
                <div key={index} className="space-y-2 border border-gray-250 dark:border-gray-800 rounded-lg overflow-hidden bg-gray-50/50 dark:bg-white/[0.01]">
                  <div className="flex items-center justify-between px-3 py-1.5 bg-gray-100 dark:bg-gray-800/80 border-b border-gray-250 dark:border-gray-800 text-[10px] text-gray-500 font-mono select-none">
                    <span>In [{cell.execution_count || " "}]</span>
                    <span>Python Cell</span>
                  </div>
                  <pre className="p-4 overflow-x-auto text-xs sm:text-sm font-mono bg-gray-150/10 dark:bg-black/10 m-0">
                    <code>{source}</code>
                  </pre>
                  {outputs.length > 0 && (
                    <div className="border-t border-gray-250 dark:border-gray-800 bg-gray-50/30 dark:bg-black/[0.05] p-3 space-y-1">
                      <div className="text-[10px] text-gray-400 font-mono uppercase tracking-wider mb-1 select-none">Outputs</div>
                      {outputs.map((out: any, oIdx: number) => {
                        if (out.output_type === "stream" || out.output_type === "execute_result") {
                          const text = Array.isArray(out.text)
                            ? out.text.join("")
                            : Array.isArray(out.data?.["text/plain"])
                            ? out.data["text/plain"].join("")
                            : out.text || out.data?.["text/plain"] || "";
                          return (
                            <pre key={oIdx} className="text-xs font-mono text-gray-700 dark:text-gray-300 m-0 overflow-x-auto whitespace-pre-wrap leading-relaxed">
                              {text}
                            </pre>
                          );
                        }
                        if (out.output_type === "error") {
                          const ename = out.ename || "Error";
                          const evalue = out.evalue || "";
                          return (
                            <pre key={oIdx} className="text-xs font-mono text-red-600 dark:text-red-400 m-0 overflow-x-auto whitespace-pre-wrap bg-red-500/5 p-2 rounded">
                              {ename}: {evalue}
                            </pre>
                          );
                        }
                        return null;
                      })}
                    </div>
                  )}
                </div>
              );
            }
            return null;
          })}
        </div>
      );
    } catch (e: any) {
      return (
        <div className="p-6 text-red-650 dark:text-red-400 text-xs">
          Failed to parse Jupyter Notebook JSON: {e.message}. Switch to Source View to inspect.
        </div>
      );
    }
  };

  return (
    <div className="fixed inset-0 bg-black/75 backdrop-blur-[4px] z-[9999] flex items-center justify-center p-4 sm:p-8 animate-fade-in" onClick={onClose}>
      <div className="bg-white dark:bg-darkCard rounded-xl shadow-2xl w-full max-w-[1200px] max-h-[90vh] flex flex-col animate-slide-up border border-gray-200 dark:border-gray-800 overflow-hidden" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-800 bg-gray-100 dark:bg-gray-800">
          <div className="flex items-center gap-2.5 font-semibold text-sm sm:text-base overflow-hidden">
            <FileCode size={18} className="text-blue-500 flex-shrink-0" />
            <span className="font-mono truncate" title={filePath}>{filePath}</span>
            {content?.from_chunks && (
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-amber-500/15 text-amber-700 dark:text-amber-400 border border-amber-500/30" title="Reconstructed from indexed chunks">
                From Chunks
              </span>
            )}
          </div>
          <button
            className="p-1 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors flex items-center justify-center cursor-pointer"
            onClick={onClose}
            title="Close (Esc)"
          >
            <X size={20} />
          </button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center justify-between px-6 py-3 border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-darkCard flex-wrap gap-2">
          {/* Left section: Info & View Mode toggle for Markdown/Jupyter */}
          <div className="flex items-center gap-4 flex-wrap">
            {content && (
              <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                {content.language} • {content.total_lines} lines
                {content.size_bytes && ` • ${(content.size_bytes / 1024).toFixed(1)} KB`}
              </span>
            )}

            {supportsPreview && content && (
              <div className="flex items-center bg-gray-100 dark:bg-gray-800 p-0.5 rounded-lg border border-gray-200/50 dark:border-gray-750">
                <button
                  onClick={() => setViewMode("preview")}
                  className={`px-3 py-1 rounded-md text-xs font-semibold flex items-center gap-1 transition-all cursor-pointer ${
                    viewMode === "preview"
                      ? "bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm"
                      : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  }`}
                >
                  <Eye size={12} />
                  Preview
                </button>
                <button
                  onClick={() => setViewMode("code")}
                  className={`px-3 py-1 rounded-md text-xs font-semibold flex items-center gap-1 transition-all cursor-pointer ${
                    viewMode === "code"
                      ? "bg-white dark:bg-gray-700 text-blue-600 dark:text-blue-400 shadow-sm"
                      : "text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                  }`}
                >
                  <Code2 size={12} />
                  Source Code
                </button>
              </div>
            )}
          </div>

          {/* Right section: Toolbar actions */}
          <div className="flex items-center gap-2 flex-wrap">
            {viewMode === "code" && (
              <button onClick={jumpToLine} className="px-3 py-1.5 border border-gray-200 dark:border-gray-800 rounded-md bg-white dark:bg-darkCard text-neutral-900 dark:text-neutral-100 hover:bg-gray-100 dark:hover:bg-gray-800 hover:border-blue-500 transition-all duration-200 text-xs whitespace-nowrap cursor-pointer flex items-center gap-1.5" title="Jump to line">
                <Navigation size={12} className="rotate-45" />
                Go to Line
              </button>
            )}
            <button onClick={copyToClipboard} className="px-3 py-1.5 border border-gray-200 dark:border-gray-800 rounded-md bg-white dark:bg-darkCard text-neutral-900 dark:text-neutral-100 hover:bg-gray-100 dark:hover:bg-gray-800 hover:border-blue-500 transition-all duration-200 text-xs whitespace-nowrap cursor-pointer flex items-center gap-1.5" title="Copy content">
              <Copy size={12} />
              Copy
            </button>
            <button onClick={downloadFile} className="px-3 py-1.5 border border-gray-200 dark:border-gray-800 rounded-md bg-white dark:bg-darkCard text-neutral-900 dark:text-neutral-100 hover:bg-gray-100 dark:hover:bg-gray-800 hover:border-blue-500 transition-all duration-200 text-xs whitespace-nowrap cursor-pointer flex items-center gap-1.5" title="Download file">
              <Download size={12} />
              Download
            </button>
            {editorUrls && (
              <div className="relative group">
                <button className="px-3 py-1.5 border border-blue-600 rounded-md bg-blue-600 text-white hover:bg-blue-700 hover:border-blue-700 transition-all duration-200 text-xs whitespace-nowrap cursor-pointer flex items-center gap-1">
                  Open in Editor ▾
                </button>
                <div className="hidden group-hover:block absolute top-full right-0 mt-1 bg-white dark:bg-darkCard border border-gray-200 dark:border-gray-800 rounded-md shadow-xl min-w-[180px] z-50">
                  <button className="w-full px-4 py-2.5 text-left border-none bg-transparent text-xs hover:bg-gray-100 dark:hover:bg-gray-800 text-neutral-900 dark:text-neutral-100 transition-colors duration-200 first:rounded-t-md last:rounded-b-md flex items-center justify-between cursor-pointer" onClick={() => openInEditor("vscode")}>
                    Visual Studio Code
                    <ExternalLink size={10} className="text-gray-400" />
                  </button>
                  <button className="w-full px-4 py-2.5 text-left border-none bg-transparent text-xs hover:bg-gray-100 dark:hover:bg-gray-800 text-neutral-900 dark:text-neutral-100 transition-colors duration-200 flex items-center justify-between cursor-pointer" onClick={() => openInEditor("vscode_insiders")}>
                    VS Code Insiders
                    <ExternalLink size={10} className="text-gray-400" />
                  </button>
                  <button className="w-full px-4 py-2.5 text-left border-none bg-transparent text-xs hover:bg-gray-100 dark:hover:bg-gray-800 text-neutral-900 dark:text-neutral-100 transition-colors duration-200 flex items-center justify-between cursor-pointer" onClick={() => openInEditor("intellij")}>
                    IntelliJ IDEA
                    <ExternalLink size={10} className="text-gray-400" />
                  </button>
                  <button className="w-full px-4 py-2.5 text-left border-none bg-transparent text-xs hover:bg-gray-100 dark:hover:bg-gray-800 text-neutral-900 dark:text-neutral-100 transition-colors duration-200 flex items-center justify-between cursor-pointer" onClick={() => openInEditor("sublime")}>
                    Sublime Text
                    <ExternalLink size={10} className="text-gray-400" />
                  </button>
                  <button className="w-full px-4 py-2.5 text-left border-none bg-transparent text-xs hover:bg-gray-100 dark:hover:bg-gray-800 text-neutral-900 dark:text-neutral-100 transition-colors duration-200 first:rounded-t-md last:rounded-b-md flex items-center justify-between cursor-pointer" onClick={() => openInEditor("atom")}>
                    Atom
                    <ExternalLink size={10} className="text-gray-400" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-auto bg-gray-50 dark:bg-darkBg custom-scrollbar">
          {loading && (
            <div className="flex flex-col items-center justify-center py-16 px-8 gap-4 text-gray-500 dark:text-gray-400">
              <Loader2 className="w-10 h-10 animate-spin text-blue-500" />
              <p className="text-sm">Loading file content...</p>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center py-16 px-8 gap-4 text-red-600 dark:text-red-400">
              <AlertTriangle size={36} className="text-red-500" />
              <p className="text-sm font-semibold">Failed to load file: {error}</p>
              <button onClick={loadFileContent} className="px-6 py-2 border border-blue-600 rounded-md bg-blue-600 text-white cursor-pointer text-sm hover:bg-blue-700 transition-colors duration-200">
                Retry
              </button>
            </div>
          )}

          {content && !loading && !error && (
            <>
              {/* Code/Source View Mode */}
              {viewMode === "code" && (
                <div className="flex font-mono text-xs sm:text-sm leading-relaxed bg-white dark:bg-darkCard">
                  <div className="py-4 pl-4 pr-2 text-right text-gray-450 dark:text-gray-500 select-none border-r border-gray-200 dark:border-gray-850 bg-black/[0.01] dark:bg-white/[0.01] min-w-[60px] font-mono">
                    {Array.from({ length: content.total_lines }, (_, i) => {
                      const lineNum = i + 1;
                      const isHighlighted = startLine && endLine && lineNum >= startLine && lineNum <= endLine;
                      return (
                        <div 
                          key={lineNum} 
                          id={`line-${lineNum}`} 
                          className={`px-2 text-xs transition-colors ${
                            isHighlighted 
                              ? "bg-yellow-200 dark:bg-yellow-900/40 text-yellow-900 dark:text-yellow-200 font-bold" 
                              : ""
                          }`}
                        >
                          {lineNum}
                        </div>
                      );
                    })}
                  </div>
                  <pre className={`flex-1 py-4 px-6 m-0 overflow-x-auto whitespace-pre bg-white dark:bg-darkCard language-${content.language} custom-scrollbar`}>
                    <code>
                      {content.content.split('\n').map((line, i) => {
                        const lineNum = i + 1;
                        const isHighlighted = startLine && endLine && lineNum >= startLine && lineNum <= endLine;
                        return (
                          <div 
                            key={lineNum} 
                            className={`${
                              isHighlighted 
                                ? "bg-yellow-100 dark:bg-yellow-900/20 border-l-4 border-yellow-500 dark:border-yellow-400 pl-2 -ml-2" 
                                : ""
                            }`}
                          >
                            {line}
                          </div>
                        );
                      })}
                    </code>
                  </pre>
                </div>
              )}

              {/* Formatted Preview Mode for Markdown & Jupyter Notebooks */}
              {viewMode === "preview" && (
                <div className="bg-white dark:bg-darkCard min-h-full">
                  {isMarkdown && (
                    <div className="p-6 max-w-4xl mx-auto prose dark:prose-invert">
                      {renderFormattedMarkdown(content.content)}
                    </div>
                  )}
                  {isIPYNB && renderNotebook(content.content)}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-gray-200 dark:border-gray-800 bg-gray-100 dark:bg-gray-800 text-center">
          <span className="text-xs text-gray-500 dark:text-gray-400">
            💡 This is a read-only preview. Use "Open in Editor" to make changes.
          </span>
        </div>
      </div>
    </div>
  );
}

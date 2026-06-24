"use client";

import React, { useState, useEffect } from "react";

interface FileViewerProps {
  filePath: string;
  onClose: () => void;
  initialLine?: number;
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
 * Modal component for viewing file content (read-only)
 * with "Open in Editor" buttons
 */
export default function FileViewer({ filePath, onClose, initialLine = 1 }: FileViewerProps) {
  const [content, setContent] = useState<FileContent | null>(null);
  const [editorUrls, setEditorUrls] = useState<EditorUrls | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentLine, setCurrentLine] = useState(initialLine);

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
      // Silently fail - editor URLs are optional
      console.error("Failed to load editor URLs:", err);
    }
  };

  const openInEditor = (editorType: keyof EditorUrls) => {
    if (!editorUrls) return;
    const url = editorUrls[editorType];
    // Try to open the URL
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
    alert("Content copied to clipboard!");
  };

  const jumpToLine = () => {
    const lineInput = prompt(`Jump to line (1-${content?.total_lines || 1}):`, String(currentLine));
    if (lineInput) {
      const line = parseInt(lineInput, 10);
      if (line >= 1 && line <= (content?.total_lines || 1)) {
        setCurrentLine(line);
        // Scroll to line
        const lineElement = document.getElementById(`line-${line}`);
        if (lineElement) {
          lineElement.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    }
  };

  return (
    <div className="file-viewer-overlay" onClick={onClose}>
      <div className="file-viewer-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="file-viewer-header">
          <div className="file-viewer-title">
            <span className="file-icon">📄</span>
            <span className="file-path">{filePath}</span>
            {content?.from_chunks && (
              <span className="badge badge-warning" title="Reconstructed from indexed chunks">
                From Chunks
              </span>
            )}
          </div>
          <button className="close-button" onClick={onClose} title="Close (Esc)">
            ✕
          </button>
        </div>

        {/* Toolbar */}
        <div className="file-viewer-toolbar">
          <div className="toolbar-left">
            {content && (
              <>
                <span className="file-info">
                  {content.language} • {content.total_lines} lines
                  {content.size_bytes && ` • ${(content.size_bytes / 1024).toFixed(1)} KB`}
                </span>
              </>
            )}
          </div>
          <div className="toolbar-right">
            <button onClick={jumpToLine} className="toolbar-btn" title="Jump to line">
              🎯 Go to Line
            </button>
            <button onClick={copyToClipboard} className="toolbar-btn" title="Copy content">
              📋 Copy
            </button>
            <button onClick={downloadFile} className="toolbar-btn" title="Download file">
              ⬇️ Download
            </button>
            {editorUrls && (
              <div className="editor-dropdown">
                <button className="toolbar-btn primary">🚀 Open in Editor ▾</button>
                <div className="editor-menu">
                  <button onClick={() => openInEditor("vscode")}>Visual Studio Code</button>
                  <button onClick={() => openInEditor("vscode_insiders")}>VS Code Insiders</button>
                  <button onClick={() => openInEditor("intellij")}>IntelliJ IDEA</button>
                  <button onClick={() => openInEditor("sublime")}>Sublime Text</button>
                  <button onClick={() => openInEditor("atom")}>Atom</button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="file-viewer-content">
          {loading && (
            <div className="file-viewer-loading">
              <div className="spinner"></div>
              <p>Loading file content...</p>
            </div>
          )}

          {error && (
            <div className="file-viewer-error">
              <p>❌ {error}</p>
              <button onClick={loadFileContent} className="retry-btn">
                Retry
              </button>
            </div>
          )}

          {content && !loading && !error && (
            <div className="code-viewer">
              <div className="line-numbers">
                {Array.from({ length: content.total_lines }, (_, i) => (
                  <div key={i + 1} id={`line-${i + 1}`} className="line-number">
                    {i + 1}
                  </div>
                ))}
              </div>
              <pre className={`code-content language-${content.language}`}>
                <code>{content.content}</code>
              </pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="file-viewer-footer">
          <span className="help-text">
            💡 This is a read-only preview. Use "Open in Editor" to make changes.
          </span>
        </div>
      </div>
    </div>
  );
}

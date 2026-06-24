"use client";

import React, { useState } from "react";

export interface FileTreeNode {
  name: string;
  type: "file" | "folder";
  path: string;
  children?: FileTreeNode[];
}

interface FileTreeProps {
  tree: FileTreeNode[];
  repoName: string;
  onFileClick?: (path: string) => void;
}

/** Get file icon based on extension */
function getFileIcon(name: string): string {
  const ext = name.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "py": return "🐍";
    case "ts":
    case "tsx": return "🔷";
    case "js":
    case "jsx": return "🟨";
    case "go": return "🔵";
    case "php": return "🐘";
    case "json": return "📋";
    case "md": return "📝";
    case "css":
    case "scss": return "🎨";
    case "html": return "🌐";
    case "yaml":
    case "yml": return "⚙️";
    default: return "📄";
  }
}

/** Single tree node (file or folder) */
function TreeNode({
  node,
  depth,
  onFileClick,
}: {
  node: FileTreeNode;
  depth: number;
  onFileClick?: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(depth < 2);

  if (node.type === "file") {
    return (
      <div
        className="tree-file"
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => onFileClick?.(node.path)}
        title={node.path}
      >
        <span className="tree-icon">{getFileIcon(node.name)}</span>
        <span className="tree-name">{node.name}</span>
      </div>
    );
  }

  return (
    <div className="tree-folder-wrapper">
      <div
        className="tree-folder"
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => setExpanded(!expanded)}
      >
        <span className="tree-arrow">{expanded ? "▾" : "▸"}</span>
        <span className="tree-icon">📁</span>
        <span className="tree-name">{node.name}</span>
        {node.children && (
          <span className="tree-count">{node.children.length}</span>
        )}
      </div>
      {expanded && node.children && (
        <div className="tree-children">
          {node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              onFileClick={onFileClick}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/** File tree sidebar component */
export default function FileTree({ tree, repoName, onFileClick }: FileTreeProps) {
  const [collapsed, setCollapsed] = useState(false);

  if (!tree || tree.length === 0) return null;

  return (
    <div className={`file-tree-panel ${collapsed ? "collapsed" : ""}`}>
      <div className="file-tree-header" onClick={() => setCollapsed(!collapsed)}>
        <span className="file-tree-toggle">{collapsed ? "▸" : "▾"}</span>
        <span className="file-tree-title">📦 {repoName}</span>
      </div>
      {!collapsed && (
        <div className="file-tree-content">
          {tree.map((node) => (
            <TreeNode
              key={node.path}
              node={node}
              depth={0}
              onFileClick={onFileClick}
            />
          ))}
        </div>
      )}
    </div>
  );
}

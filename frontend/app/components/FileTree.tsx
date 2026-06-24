"use client";

import React, { useState } from "react";
import { Folder, FileText, FileCode, ChevronRight, ChevronDown, Package } from "lucide-react";

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

/** Get color-coded file icon based on extension */
function getFileIcon(name: string) {
  const ext = name.split(".").pop()?.toLowerCase();
  switch (ext) {
    case "py": return <FileCode size={14} className="text-green-600 dark:text-green-400" />;
    case "ts":
    case "tsx": return <FileCode size={14} className="text-blue-500 dark:text-blue-400" />;
    case "js":
    case "jsx": return <FileCode size={14} className="text-amber-500 dark:text-amber-400" />;
    case "go": return <FileCode size={14} className="text-cyan-600 dark:text-cyan-400" />;
    case "php": return <FileCode size={14} className="text-purple-500 dark:text-purple-400" />;
    case "json": return <FileText size={14} className="text-rose-500 dark:text-rose-400" />;
    case "md": return <FileText size={14} className="text-teal-600 dark:text-teal-400" />;
    case "css":
    case "scss": return <FileCode size={14} className="text-pink-500 dark:text-pink-400" />;
    case "html": return <FileCode size={14} className="text-orange-500 dark:text-orange-400" />;
    case "yaml":
    case "yml": return <FileText size={14} className="text-neutral-500 dark:text-neutral-400" />;
    default: return <FileText size={14} className="text-neutral-400" />;
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
        className="flex items-center gap-2 py-1.5 px-2 cursor-pointer text-xs transition-colors duration-100 rounded hover:bg-blue-500/10 dark:hover:bg-blue-500/5 mx-2 select-none"
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => onFileClick?.(node.path)}
        title={node.path}
      >
        <span className="flex-shrink-0">{getFileIcon(node.name)}</span>
        <span className="truncate text-neutral-700 dark:text-neutral-300 font-mono">{node.name}</span>
      </div>
    );
  }

  return (
    <div>
      <div
        className="flex items-center gap-1.5 py-2 px-2 cursor-pointer text-xs font-semibold transition-colors duration-100 rounded hover:bg-blue-500/5 mx-2 select-none"
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-gray-400 dark:text-gray-500 flex-shrink-0">
          {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </span>
        <Folder size={14} className="text-blue-500/80 dark:text-blue-400/80 flex-shrink-0" />
        <span className="truncate text-neutral-800 dark:text-neutral-200">{node.name}</span>
        {node.children && (
          <span className="ml-auto text-[10px] text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-800/60 px-1.5 py-0.5 rounded-full">{node.children.length}</span>
        )}
      </div>
      {expanded && node.children && (
        <div className="animate-slide-down">
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
    <div className="w-full flex flex-col h-full bg-white dark:bg-darkCard select-none">
      <div
        className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-800 cursor-pointer font-bold text-xs sticky top-0 bg-white dark:bg-darkCard hover:bg-blue-500/[0.02] z-10 select-none text-neutral-800 dark:text-neutral-200"
        onClick={() => setCollapsed(!collapsed)}
      >
        <div className="flex items-center gap-2 truncate">
          <Package size={14} className="text-blue-600 dark:text-blue-400" />
          <span className="truncate font-bold tracking-wide">{repoName}</span>
        </div>
        <span className="text-gray-400">
          {collapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
        </span>
      </div>
      {!collapsed && (
        <div className="py-2 overflow-y-auto flex-1 custom-scrollbar">
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

"use client";

import React from "react";
import { BarChart3, Hash, Target, TrendingUp } from "lucide-react";

export interface RetrievalStatisticsProps {
  bm25Count: number;
  indobertCount: number;
}

/**
 * RetrievalStatistics displays comparison metrics for BM25 and IndoBERT retrievers.
 */
export default function RetrievalStatistics({
  bm25Count,
  indobertCount,
}: RetrievalStatisticsProps) {
  const total = bm25Count + indobertCount;
  const bm25Percent = total > 0 ? Math.round((bm25Count / total) * 100) : 0;
  const indobertPercent = total > 0 ? Math.round((indobertCount / total) * 100) : 0;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6 p-4 bg-gradient-to-r from-blue-50/50 to-purple-50/50 dark:from-blue-900/10 dark:to-purple-900/10 rounded-xl border border-gray-200 dark:border-gray-800">
      {/* Total Chunks */}
      <div className="flex items-center gap-3 p-3 bg-white/80 dark:bg-darkCard/80 rounded-lg border border-gray-100 dark:border-gray-800/50 backdrop-blur-sm">
        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500 text-white">
          <BarChart3 size={20} />
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">Total Chunks</div>
          <div className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">{total}</div>
        </div>
      </div>

      {/* BM25 Stats */}
      <div className="flex items-center gap-3 p-3 bg-white/80 dark:bg-darkCard/80 rounded-lg border border-gray-100 dark:border-gray-800/50 backdrop-blur-sm">
        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400">
          <Hash size={20} />
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">BM25 (Lexical)</div>
          <div className="flex items-baseline gap-2">
            <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">{bm25Count}</div>
            <div className="text-xs font-semibold text-blue-600/60 dark:text-blue-400/60">({bm25Percent}%)</div>
          </div>
        </div>
      </div>

      {/* IndoBERT Stats */}
      <div className="flex items-center gap-3 p-3 bg-white/80 dark:bg-darkCard/80 rounded-lg border border-gray-100 dark:border-gray-800/50 backdrop-blur-sm">
        <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-purple-500/10 text-purple-600 dark:text-purple-400">
          <TrendingUp size={20} />
        </div>
        <div className="flex-1">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">IndoBERT (Semantic)</div>
          <div className="flex items-baseline gap-2">
            <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">{indobertCount}</div>
            <div className="text-xs font-semibold text-purple-600/60 dark:text-purple-400/60">({indobertPercent}%)</div>
          </div>
        </div>
      </div>
    </div>
  );
}

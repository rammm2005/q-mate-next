import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/status`, {
      cache: "no-store",
    });
    if (!response.ok) {
      return NextResponse.json(
        { is_indexed: false, repo_name: "", total_chunks: 0 },
        { status: 200 }
      );
    }
    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { is_indexed: false, repo_name: "", total_chunks: 0 },
      { status: 200 }
    );
  }
}

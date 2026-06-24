import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/filetree`);
    if (!response.ok) {
      return NextResponse.json({ tree: [], repo_name: "" });
    }
    const data = await response.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ tree: [], repo_name: "" });
  }
}

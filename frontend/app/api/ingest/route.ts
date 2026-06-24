import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const backendResponse = await fetch(`${BACKEND_URL}/api/ingest`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": "default-key",
      },
      body: JSON.stringify(body),
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => null);
      return NextResponse.json(
        errorData || { detail: `Backend error: ${backendResponse.status}` },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      {
        status: "error",
        repo_name: "",
        total_files: 0,
        total_chunks: 0,
        languages: [],
        error: "Backend service is unavailable. Make sure the FastAPI server is running on port 8000.",
      },
      { status: 200 }
    );
  }
}

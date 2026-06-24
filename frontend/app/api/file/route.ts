import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

/**
 * GET /api/file?path=...&action=content|open&line=1
 * 
 * - action=content: Get file content for preview
 * - action=open: Get editor URLs to open file
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const filePath = searchParams.get("path");
  const action = searchParams.get("action") || "content";
  const line = searchParams.get("line") || "1";

  if (!filePath) {
    return NextResponse.json(
      { detail: "Missing 'path' query parameter" },
      { status: 400 }
    );
  }

  try {
    let endpoint = "";
    if (action === "content") {
      endpoint = `/api/file/content?file_path=${encodeURIComponent(filePath)}`;
    } else if (action === "open") {
      endpoint = `/api/file/open?file_path=${encodeURIComponent(filePath)}&line=${line}`;
    } else {
      return NextResponse.json(
        { detail: "Invalid action. Use 'content' or 'open'" },
        { status: 400 }
      );
    }

    const backendResponse = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => null);
      return NextResponse.json(
        { detail: errorData?.detail || `Backend error: ${backendResponse.status}` },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json(
      { detail: "Backend service is unavailable" },
      { status: 503 }
    );
  }
}

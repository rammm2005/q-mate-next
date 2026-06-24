import { NextRequest, NextResponse } from "next/server";

/**
 * Next.js API route that proxies requests to the FastAPI backend.
 * This avoids CORS issues by keeping requests on the same origin.
 */

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const backendResponse = await fetch(`${BACKEND_URL}/api/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": request.headers.get("X-API-Key") || "default-key",
      },
      body: JSON.stringify(body),
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
    // Backend is not running or unreachable
    return NextResponse.json(
      { detail: "Backend service is unavailable. Make sure the FastAPI server is running on port 8000." },
      { status: 503 }
    );
  }
}

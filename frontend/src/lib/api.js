const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

/**
 * Sends the uploaded PDF + the user-defined target schema
 * to the backend and returns the extraction result.
 *
 * @param {File} file
 * @param {{name: string, description: string}[]} fields
 */
export async function runExtraction(file, fields) {
  const form = new FormData();

  form.append("file", file);

  form.append(
    "target_schema",
    JSON.stringify(
      fields.map((f) => ({
        name: f.name,
        description: f.description,
      }))
    )
  );

  let response;

  try {
    response = await fetch(`${API_BASE}/api/extract`, {
      method: "POST",
      body: form,
    });
  } catch (err) {
    throw new ApiError(
      `Could not reach the extraction service at ${API_BASE}. Is the backend running?`,
      0
    );
  }

  let payload = null;

  try {
    payload = await response.json();
  } catch (err) {
    // No JSON body
  }

  if (!response.ok) {
    const detail =
      payload?.detail ||
      `Request failed with status ${response.status}.`;

    throw new ApiError(detail, response.status);
  }

  return payload;
}

export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    return response.ok;
  } catch (err) {
    return false;
  }
}
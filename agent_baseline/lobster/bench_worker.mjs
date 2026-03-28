/**
 * Lobster benchmark worker – a minimal Node.js pipeline that calls an
 * OpenAI-compatible chat endpoint (Groq, vLLM, etc.).
 *
 * Input  (JSON on stdin):
 *   { base_url, api_key, model, system_prompt, question }
 *
 * Output (JSON on stdout):
 *   { content: "..." }  or  { error: "..." }
 */

import { createInterface } from "node:readline/promises";

async function readStdin() {
  const rl = createInterface({ input: process.stdin });
  const lines = [];
  for await (const line of rl) lines.push(line);
  return lines.join("\n");
}

async function callChat({ base_url, api_key, model, system_prompt, question }) {
  const url = `${base_url.replace(/\/+$/, "")}/chat/completions`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${api_key}`,
    },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: system_prompt },
        { role: "user", content: question },
      ],
      max_tokens: 4096,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 500)}`);
  }

  const json = await res.json();
  return {
    content: json.choices?.[0]?.message?.content ?? "",
    usage: json.usage ?? {},
  };
}

try {
  const data = JSON.parse(await readStdin());
  const result = await callChat(data);
  process.stdout.write(JSON.stringify(result) + "\n");
} catch (err) {
  process.stdout.write(JSON.stringify({ error: String(err.message ?? err) }) + "\n");
  process.exit(1);
}

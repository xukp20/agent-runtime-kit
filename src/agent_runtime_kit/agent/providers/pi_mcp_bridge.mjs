import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { Type } from "@earendil-works/pi-ai";

const manifestPath = process.env.ARK_PI_MCP_MANIFEST;
const runtimeRoot = process.env.ARK_PI_MCP_RUNTIME_ROOT;
if (!manifestPath || !runtimeRoot) {
  throw new Error("ARK Pi MCP bridge requires ARK_PI_MCP_MANIFEST and ARK_PI_MCP_RUNTIME_ROOT");
}

const moduleUrl = (relative) =>
  pathToFileURL(join(runtimeRoot, "node_modules", "@modelcontextprotocol", "sdk", "dist", "esm", relative)).href;

const [
  { Client },
  { StdioClientTransport },
  { StreamableHTTPClientTransport },
  { SSEClientTransport },
  { ToolListChangedNotificationSchema },
] =
  await Promise.all([
    import(moduleUrl("client/index.js")),
    import(moduleUrl("client/stdio.js")),
    import(moduleUrl("client/streamableHttp.js")),
    import(moduleUrl("client/sse.js")),
    import(moduleUrl("types.js")),
  ]);

const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
if (manifest.schema_version !== 1 || !Array.isArray(manifest.servers)) {
  throw new Error("invalid ARK Pi MCP bridge manifest");
}

const states = new Map();

class McpToolResultError extends Error {}

function safeName(value) {
  const normalized = String(value).replace(/[^A-Za-z0-9_-]/g, "_").replace(/^[-_]+/, "");
  return normalized || "unnamed";
}

function headers(server) {
  const result = { ...(server.http_headers || {}) };
  for (const [name, envName] of Object.entries(server.env_http_headers || {})) {
    if (process.env[envName]) result[name] = process.env[envName];
  }
  if (server.bearer_token_env_var && process.env[server.bearer_token_env_var]) {
    result.Authorization = `Bearer ${process.env[server.bearer_token_env_var]}`;
  }
  return result;
}

function stdioEnvironment(server) {
  const env = { ...process.env, ...(server.env || {}) };
  for (const name of server.env_vars || []) {
    if (process.env[name] !== undefined) env[name] = process.env[name];
  }
  return env;
}

function makeTransport(server) {
  if (server.transport === "stdio") {
    if (!server.command) throw new Error(`MCP server ${server.name} has no stdio command`);
    return new StdioClientTransport({
      command: server.command,
      args: server.args || [],
      cwd: server.cwd || undefined,
      env: stdioEnvironment(server),
      stderr: "pipe",
    });
  }
  if (!server.url) throw new Error(`MCP server ${server.name} has no URL`);
  const requestHeaders = headers(server);
  if (server.transport === "sse") {
    return new SSEClientTransport(new URL(server.url), {
      requestInit: { headers: requestHeaders },
      eventSourceInit: { fetch: (url, init) => fetch(url, { ...init, headers: requestHeaders }) },
    });
  }
  return new StreamableHTTPClientTransport(new URL(server.url), {
    requestInit: { headers: requestHeaders },
    reconnectionOptions: {
      maxReconnectionDelay: 5000,
      initialReconnectionDelay: 250,
      reconnectionDelayGrowFactor: 2,
      maxRetries: 2,
    },
  });
}

async function connect(server) {
  const state = states.get(server.name) || {
    server,
    client: null,
    transport: null,
    tools: [],
    onToolsChanged: null,
  };
  if (state.client) return state;
  const client = new Client({ name: "ark-pi-mcp-bridge", version: "1.0.0" });
  const transport = makeTransport(server);
  await client.connect(transport, { timeout: (server.startup_timeout_sec || 30) * 1000 });
  const listed = await client.listTools();
  state.client = client;
  state.transport = transport;
  state.tools = listed.tools || [];
  states.set(server.name, state);
  installListChangedHandler(state);
  return state;
}

function installListChangedHandler(state) {
  if (!state.client || !state.onToolsChanged) return;
  state.client.setNotificationHandler(ToolListChangedNotificationSchema, async () => {
    try {
      const listed = await state.client.listTools();
      state.tools = listed.tools || [];
      await state.onToolsChanged(state.tools);
    } catch {
      // Keep the last known tool registry. A transport error will still use
      // the normal one-shot reconnect path on the next invocation.
    }
  });
}

async function disconnect(state) {
  try {
    if (state.client) await state.client.close();
  } finally {
    state.client = null;
    state.transport = null;
  }
}

function enabled(server, tool) {
  if (Array.isArray(server.enabled_tools) && !server.enabled_tools.includes(tool.name)) return false;
  if (Array.isArray(server.disabled_tools) && server.disabled_tools.includes(tool.name)) return false;
  return true;
}

function piContent(content) {
  const result = [];
  for (const item of content || []) {
    if (item.type === "text") result.push({ type: "text", text: item.text });
    else if (item.type === "image") result.push({ type: "image", data: item.data, mimeType: item.mimeType });
    else result.push({ type: "text", text: JSON.stringify(item) });
  }
  return result.length ? result : [{ type: "text", text: "" }];
}

export default async function (pi) {
  const owners = new Map();
  for (const server of manifest.servers) {
    if (server.enabled === false) continue;
    let state;
    try {
      state = await connect(server);
    } catch (error) {
      if (server.required) throw error;
      continue;
    }
    const register = (tool) => {
      if (!enabled(server, tool)) return;
      const name = `mcp__${safeName(server.name)}__${safeName(tool.name)}`;
      const owner = `${server.name}/${tool.name}`;
      if (owners.has(name) && owners.get(name) !== owner) {
        throw new Error(`duplicate projected MCP tool name: ${name}`);
      }
      owners.set(name, owner);
      pi.registerTool({
        name,
        label: `${server.name}: ${tool.title || tool.name}`,
        description: tool.description || `MCP tool ${server.name}/${tool.name}`,
        parameters: Type.Unsafe(tool.inputSchema || { type: "object", properties: {} }),
        async execute(_toolCallId, params, signal) {
          let current = await connect(server);
          if (!current.tools.some((candidate) => candidate.name === tool.name)) {
            throw new Error(`MCP tool is no longer advertised: ${server.name}/${tool.name}`);
          }
          try {
            const result = await current.client.callTool(
              { name: tool.name, arguments: params },
              undefined,
              { signal, timeout: (server.tool_timeout_sec || 60) * 1000 },
            );
            if (result.isError) {
              throw new McpToolResultError(`MCP ${server.name}/${tool.name} failed: ${JSON.stringify(result.content || [])}`);
            }
            return {
              content: piContent(result.content),
              details: { mcpServer: server.name, mcpTool: tool.name, structuredContent: result.structuredContent },
            };
          } catch (firstError) {
            if (signal?.aborted || firstError instanceof McpToolResultError) throw firstError;
            await disconnect(current);
            current = await connect(server);
            const result = await current.client.callTool(
              { name: tool.name, arguments: params },
              undefined,
              { signal, timeout: (server.tool_timeout_sec || 60) * 1000 },
            );
            if (result.isError) {
              throw new McpToolResultError(`MCP ${server.name}/${tool.name} failed: ${JSON.stringify(result.content || [])}`);
            }
            return {
              content: piContent(result.content),
              details: { mcpServer: server.name, mcpTool: tool.name, structuredContent: result.structuredContent, reconnected: true },
            };
          }
        },
      });
    };
    for (const tool of state.tools) register(tool);
    state.onToolsChanged = async (tools) => {
      for (const tool of tools) register(tool);
    };
    installListChangedHandler(state);
  }

  pi.on("session_shutdown", async () => {
    await Promise.allSettled([...states.values()].map(disconnect));
  });
}

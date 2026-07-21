import {
  fauxAssistantMessage,
  fauxProvider,
  fauxToolCall,
} from "@earendil-works/pi-ai";

export default function (pi) {
  const provider = fauxProvider({
    provider: "ark-mcp-probe",
    api: "ark-probe-api",
    tokensPerSecond: 2000,
    models: [{
      id: "faux-mcp",
      name: "ARK MCP Probe Faux",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 4096,
    }],
  });
  provider.setResponses([
    fauxAssistantMessage(
      fauxToolCall("mcp__demo__echo", { marker: "ARK_PI_MCP_OK" }, { id: "mcp-1" }),
      { stopReason: "toolUse" },
    ),
    fauxAssistantMessage("ARK_PI_MCP_DONE"),
  ]);
  const model = provider.models[0];
  pi.registerProvider("ark-mcp-probe", {
    name: "ARK MCP Probe Faux",
    baseUrl: model.baseUrl,
    apiKey: "probe-no-network",
    api: provider.api,
    streamSimple: provider.provider.streamSimple,
    models: [{
      id: model.id,
      name: model.name,
      reasoning: model.reasoning,
      input: model.input,
      cost: model.cost,
      contextWindow: model.contextWindow,
      maxTokens: model.maxTokens,
    }],
  });
}

